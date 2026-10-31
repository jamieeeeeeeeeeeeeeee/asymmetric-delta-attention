"""
This script checks if the chunked kernel behaves on your hardware.
It should be ran before big training runs to identify any env issues
- does chunk_ada match the recursion
- does having vector beta be a multiple of the ones vector (so ~scalar beta)
  have the same as chunk_kda with actual scalar beta
- does each config build and have expected param count and run
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / '3rdparty/flash-linear-attention'))

import fla
from fla.ops.ada import chunk_ada
from fla.ops.kda import chunk_kda
from transformers import AutoConfig, AutoModelForCausalLM

MODELS = ('ada_r8', 'kda', 'precond_kda', 'gdn2', 'fg2_gdn')
INV_SOFTPLUS = (math.log(math.expm1(1e-3)), math.log(math.expm1(0.1)))

def emit(out, record):
    line = json.dumps(record)
    print(line, flush=True)
    if out:
        with open(out, 'a') as f:
            f.write(line + '\n')

def reference(q, k, v, g, beta, A_log, dt_bias):
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    H, K = q.shape[2], q.shape[3]
    g = -A_log.double().exp().view(1, 1, H, 1) * F.softplus(g + dt_bias.double().view(H, K))
    if beta.dim() == 3:
        beta = beta.unsqueeze(-1)
    state = k.new_zeros(q.shape[0], H, K, v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp().unsqueeze(-1)
        error = v[:, t] - (state * k[:, t].unsqueeze(-1)).sum(-2)
        state = state + (beta[:, t] * k[:, t]).unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append((state * (q[:, t] * K ** -0.5).unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, 1), state

def relative(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)).item()


def run(fn, data, do, dht):
    inputs = {n: t.detach().clone().requires_grad_() for n, t in data.items()}
    o, s = fn(**inputs)
    loss = (o.double() * do.double()).sum() + (s.double() * dht.double()).sum()
    grads = torch.autograd.grad(loss, tuple(inputs.values()))
    return dict(output=o.detach(), final_state=s.detach(), **{'grad_' + n: g.detach() for n, g in zip(inputs, grads)})

def kernel(op):
    return lambda **x: op(**x, output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)

def inputs(dim, dtype, scalar_beta, length=256, heads=2):
    torch.manual_seed(dim)
    shape = (1, length, heads, dim)
    beta = 0.1 + 0.8 * torch.rand(shape[:3] if scalar_beta else shape, device='cuda')
    q = torch.randn(shape, device='cuda', dtype=dtype)
    k = torch.randn(shape, device='cuda', dtype=dtype)
    v = torch.randn(shape, device='cuda', dtype=dtype)
    g = torch.randn(shape, device='cuda', dtype=dtype)
    A_log = torch.empty(heads, device='cuda').uniform_(0, 16).log()
    dt_bias = torch.empty(heads * dim, device='cuda').uniform_(*INV_SOFTPLUS)
    data = dict(q=q, k=k, v=v, g=g, beta=beta.to(dtype), A_log=A_log, dt_bias=dt_bias)
    return data, torch.randn(shape, device='cuda'), torch.randn(1, heads, dim, dim, device='cuda')

def op_case(dim, dtype):
    limit = 0.06 if dtype == torch.bfloat16 else 0.006
    data, do, dht = inputs(dim, dtype, scalar_beta=False)
    ada = run(kernel(chunk_ada), data, do, dht)
    exact = run(reference, data, do, dht)
    ada_err = {n: relative(ada[n], exact[n]) for n in ada}
    data, do, dht = inputs(dim, dtype, scalar_beta=True)
    kda = run(kernel(chunk_kda), data, do, dht)
    exact = run(reference, data, do, dht)
    kda_err = {n: relative(kda[n], exact[n]) for n in kda}
    finite = all(torch.isfinite(t).all().item() for t in ada.values())
    passed = finite and all(e < max(limit, 1.5 * kda_err[n]) for n, e in ada_err.items())
    return dict(kind='op_fp64', head_dim=dim, dtype=str(dtype), limit=limit, passed=passed, ada_errors=ada_err, upstream_kda_errors=kda_err)

def reduction_case(dim, dtype):
    data, do, dht = inputs(dim, dtype, scalar_beta=True)
    kda = run(kernel(chunk_kda), data, do, dht)
    vec = dict(data, beta=data['beta'].unsqueeze(-1).expand(*data['beta'].shape, dim).contiguous())
    ada = run(kernel(chunk_ada), vec, do, dht)
    ada['grad_beta'] = ada['grad_beta'].sum(-1)
    errors = {n: relative(ada[n], kda[n]) for n in kda}
    limit = 0.06 if dtype == torch.bfloat16 else 0.006
    return dict(kind='ada_equals_kda_for_constant_beta', head_dim=dim, dtype=str(dtype), limit=limit, passed=max(errors.values()) < limit, errors=errors)

def stats(t):
    t = t.detach().float()
    return dict(min=t.min().item(), max=t.max().item(), mean=t.mean().item(), zero_frac=(t == 0).float().mean().item())

def model_case(name):
    config = AutoConfig.from_pretrained(REPO / 'configs' / f'{name}.json')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
        model.apply(lambda m: setattr(m, '_is_hf_initialized', False))
    model.to_empty(device='cuda')
    with torch.no_grad():
        model.post_init()
    params = sum(p.numel() for p in model.parameters())
    attn = [layer.attn for layer in model.model.layers]
    init = {}
    for p in ('dt_bias', 'A_log', 'dt_bias_atk', 'A_log_atk'):
        if hasattr(attn[0], p):
            init[p] = stats(torch.cat([getattr(a, p).flatten() for a in attn]))
    if 'A_log' in init:
        init['A'] = stats(torch.cat([a.A_log.exp().flatten() for a in attn]))
    dt = init.get('dt_bias')
    dt_ok = dt is not None and dt['zero_frac'] == 0 and INV_SOFTPLUS[0] - 1e-3 <= dt['min'] and dt['max'] <= INV_SOFTPLUS[1] + 1e-3
    init_ok = dt_ok and 0 <= init['A']['min'] and init['A']['max'] <= 16
    finite_params = all(torch.isfinite(p).all().item() for p in model.parameters())

    model.train()
    x = torch.randint(0, config.vocab_size, (1, 2048), device='cuda')
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
    finite_grads = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    passed = init_ok and finite_params and finite_grads and 9 < loss.item() < 12
    record = dict(kind='model', config=name, model_type=config.model_type, params=params, init=init, loss=loss.item(), peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30, passed=passed)
    del model, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('configs', nargs='*')
    args = parser.parse_args()
    configs = args.configs or [f'{m}_{s}' for s in ('1B', '340M') for m in MODELS if (REPO / 'configs' / f'{m}_{s}.json').exists()]
    expected = str(REPO / '3rdparty/flash-linear-attention')
    assert fla.__file__.startswith(expected), f'wrong fla: {fla.__file__}'
    emit(args.out, dict(kind='env', fla=fla.__file__, torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    cases = [(f, dim, dtype) for f in (op_case, reduction_case) for dim in (120, 128) for dtype in (torch.float32, torch.bfloat16)]
    cases += [(model_case, name) for name in configs]
    failures = 0
    for f, *a in cases:
        try:
            record = f(*a)
        except Exception as e:
            record = dict(kind=f.__name__, args=[str(x) for x in a], passed=False, error=repr(e))
            torch.cuda.empty_cache()
        failures += not record['passed']
        emit(args.out, record)
    emit(args.out, dict(kind='summary', failures=failures))
    sys.exit(1 if failures else 0)

if __name__ == '__main__':
    main()
