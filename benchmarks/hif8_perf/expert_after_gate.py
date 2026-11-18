# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Shared expert computation after either routed or source-side gate dispatch."""
import torch
from deepspeed.module_inject.auto_ep_layer import _AllToAllV
from deepspeed.moe.ep_experts import apply_expert_activation
from fused_gate_dispatch import PrequantizedFrozenGMM


def expert_after_gate(layer, gate, tokens, payload, scale, inverse, plan, config):
    experts = layer.experts
    offsets = plan.local_counts_by_source.sum(0).cumsum(0).to(torch.int64)
    gate.add_(experts.lora1(tokens, offsets))
    up = (experts.grouped_mm(tokens, experts.w3, offsets, trans_b=True)
          if config is None else PrequantizedFrozenGMM.apply(tokens, payload, scale, experts.w3, offsets, config))
    up.add_(experts.lora3(tokens, offsets))
    hidden = apply_expert_activation(gate,
                                     up,
                                     experts.activation,
                                     experts.activation_alpha,
                                     experts.activation_limit,
                                     fused=False)
    del gate, up, tokens, payload, scale
    output = experts.grouped_mm(hidden, experts.w2, offsets, trans_b=True)
    output.add_(experts.lora2(hidden, offsets))
    return _AllToAllV.apply(layer.ep_group, output.index_select(0, inverse), plan.output_splits, plan.input_splits)
