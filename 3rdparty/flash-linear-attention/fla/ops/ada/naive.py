# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_recurrent_ada(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(
        lambda x: x.to(torch.float32),
        [q, k, v, g, beta]
    )
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state

    o = torch.zeros_like(v)

    for i in range(T):
        q_i, k_i, v_i, g_i, beta_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]

        k_precond = beta_i * k_i

        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', k_precond, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

    if not output_final_state:
        S = None
    return o.to(dtype), S
