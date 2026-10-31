# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.ada import chunk_ada

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class ADAttention(nn.Module):
    """
    Asymmetric Delta Attention (ADA) layer implementation.

    Each layer contains around ``hidden_size * key_dim * 2 + hidden_size * value_dim * 3`` parameters
    (q/k projections + v/g/o projections), plus the low-rank f_proj and b_proj bottlenecks.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`,
            where `num_v_heads` must be divisible by `num_heads`. Default: `None`.
        mode (str, Optional):
            Which ADA kernel to use.
            Currently available: `chunk`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            `Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues <https://arxiv.org/abs/2411.12537>`_
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        safe_gate (bool, Optional):
            Whether the kernel can assume the gate values (in log space) are in ``[lower_bound, 0)``
            and use M=16 TensorCore acceleration. Requires ``lower_bound`` to be set.
            See :func:`chunk_ada` for details. Default: ``False``.
        lower_bound (float, Optional):
            Lower bound for the forget gate in log space. Clamps the gate output to ``[lower_bound, 0)``.
            With ``-5``, the per-step decay ``exp(g) ≈ 0.0067`` at minimum — negligible impact on quality.
            See :func:`chunk_ada` for details. Default: ``None``.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
        beta_rank (int, Optional):
            Rank of the per-dimension beta projection: -1 full rank, 0 one scalar per head, r > 0 rank-r bottleneck.
            Default: -1.
        use_decay_gate (bool, Optional):
            Whether to use a decay gate. Default: `True`.
        use_scalar_gate (bool, Optional):
            Whether the decay gate is one scalar per head instead of per dimension. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use the sigmoid output gate. Default: `True`.
        use_scalar_dt_bias (bool, Optional):
            Whether dt_bias is one scalar per head instead of per dimension. Default: `False`.
        dt_init (str, Optional):
            Initialisation of dt_bias, `gdn` (log-uniform in [1e-3, 0.1]) or `zeros`. Default: `gdn`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        beta_rank: int = -1,
        use_decay_gate: bool = True,
        use_scalar_gate: bool = False,
        use_output_gate: bool = True,
        use_scalar_dt_bias: bool = False,
        dt_init: str = "gdn",
        **kwargs,
    ) -> ADAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        self.beta_rank = beta_rank
        self.use_decay_gate = use_decay_gate
        self.use_scalar_gate = use_scalar_gate
        self.use_output_gate = use_output_gate
        self.use_scalar_dt_bias = use_scalar_dt_bias

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ["chunk"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_decay_gate:
            # KDA gate projection (diagonal - per dimension, 2-layer MLP)
            if self.use_scalar_gate:
                self.f_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            else:
                self.f_proj = nn.Sequential(
                    nn.Linear(hidden_size, self.head_v_dim, bias=False),
                    nn.Linear(self.head_v_dim, self.num_v_heads * self.head_k_dim, bias=False),
                )

            if safe_gate:
                self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
            else:
                self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)))
            self.A_log._no_weight_decay = True
            dt_bias_size = self.num_v_heads if use_scalar_dt_bias else self.num_v_heads * self.head_k_dim
            if dt_init == "gdn":
                dt = torch.exp(
                    torch.rand(dt_bias_size) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            elif dt_init == "zeros":
                self.dt_bias = nn.Parameter(torch.zeros(dt_bias_size, dtype=torch.float32))
            else:
                raise ValueError(f"Unsupported dt_init `{dt_init}`. Expected 'zeros' or 'gdn'.")
            self.dt_bias._no_weight_decay = True
            self.dt_bias._is_hf_initialized = True

        if self.beta_rank == -1:
            self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            nn.init.zeros_(self.b_proj.weight)
        elif self.beta_rank == 0:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        elif self.beta_rank > 0:
            self.b_proj = nn.Sequential(
                nn.Linear(hidden_size, int(self.beta_rank), bias=False),
                nn.Linear(int(self.beta_rank), self.key_dim, bias=False),
            )
        else:
            raise ValueError(f"Unsupported beta_rank `{self.beta_rank}`.")

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        if self.use_output_gate:
            self.g_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_v_dim, bias=False),
                nn.Linear(self.head_v_dim, self.value_dim, bias=True),
            )
            self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        mode = self.mode

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.beta_rank == 0:
            beta = rearrange(beta, "... h -> ... h 1").expand(-1, -1, -1, self.head_k_dim)
        else:
            beta = rearrange(beta, "... (h d) -> ... h d", d=self.head_k_dim)
        beta = beta.contiguous()

        if self.use_decay_gate:
            g = self.f_proj(hidden_states)
            if self.use_scalar_gate:
                g = g.unsqueeze(-1).expand(-1, -1, -1, self.head_k_dim).contiguous()
            else:
                g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
            dt_bias = (
                self.dt_bias.repeat_interleave(self.head_k_dim)
                if self.use_scalar_dt_bias else self.dt_bias
            )
        else:
            g = torch.zeros_like(q)
            dt_bias = None

        # for multi-value attention, we repeat the inputs for simplicity.
        if self.num_v_heads > self.num_heads:
            q, k = (repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads) for x in (q, k))
            beta = repeat(beta, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
            if self.use_decay_gate:
                g = repeat(g, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None

        if mode == "chunk":
            o, recurrent_state = chunk_ada(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=self.use_decay_gate,
                A_log=self.A_log if self.use_decay_gate else None,
                dt_bias=dt_bias,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_output_gate:
            o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
