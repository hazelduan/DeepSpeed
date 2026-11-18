# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Experiment-only expert adapters, installed after AutoEP has sharded base weights.

All routed and shared experts execute; only rank-r adapters train. This module is
deliberately separate from the production ZeRO-2 repair.
"""
import copy
import functools
import os
import types

import torch
from torch import nn
from torch.nn import functional as F

from deepspeed.moe.ep_experts import apply_expert_activation
from deepspeed.ops.npu.grouped_mm import grouped_mm


class GroupedAdapter(nn.Module):

    def __init__(self, weight, rank, alpha):
        super().__init__()
        experts, outputs, inputs = weight.shape
        self.a = nn.Parameter(weight.new_empty(experts, rank, inputs))
        self.b = nn.Parameter(weight.new_zeros(experts, outputs, rank))
        nn.init.normal_(self.a, std=1 / inputs**0.5)
        self.scale = alpha / rank

    def forward(self, x, offsets):
        inner = grouped_mm(x, self.a, offsets, trans_b=True)
        return grouped_mm(inner, self.b, offsets, trans_b=True).mul_(self.scale)


class LinearAdapter(nn.Module):

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.a, std=1 / base.in_features**0.5)
        self.scale = alpha / rank

    def forward(self, x):
        delta = F.linear(F.linear(x, self.a), self.b).mul_(self.scale)
        return self.base(x).add_(delta)


def expert_forward(self, x, counts):
    offsets = counts.cumsum(0).to(torch.int64)
    valid = int(offsets[-1].item())
    tokens = x[:valid]
    gate = self.grouped_mm(tokens, self.w1, offsets, trans_b=True)
    gate.add_(self.lora1(tokens, offsets))
    up = self.grouped_mm(tokens, self.w3, offsets, trans_b=True)
    up.add_(self.lora3(tokens, offsets))
    hidden = apply_expert_activation(gate,
                                     up,
                                     self.activation,
                                     self.activation_alpha,
                                     self.activation_limit,
                                     fused=False)
    output = self.grouped_mm(hidden, self.w2, offsets, trans_b=True)
    output.add_(self.lora2(hidden, offsets))
    # F.pad clones even when all padding widths are zero. Large routed
    # buffers must not acquire that redundant copy on the usual path.
    if valid == x.shape[0]:
        return output
    return F.pad(output, (0, 0, 0, x.shape[0] - valid))


def attach_adapters(layer, rank):
    from deepspeed.checkpoint.autoep_affine import autoep_placement_to_affine_map
    from deepspeed.checkpoint.constants import AFFINE_MAP, AUTOEP_PARAM_LOGICAL_SHAPE, DS_AUTOEP_UC_META

    alpha = 2 * rank
    experts = layer.experts
    for index in (1, 2, 3):
        base = getattr(experts, f'w{index}')
        base.requires_grad_(False)
        adapter = GroupedAdapter(base, rank, alpha)
        for parameter in adapter.parameters():
            parameter.allreduce = False
            parameter.group_name = layer.ep_group_name
            parameter.ds_zero_placement_family = 'autoep_expert'
            parameter.ds_zero_partition_group_name = layer.ep_group_name
            metadata = copy.deepcopy(getattr(base, DS_AUTOEP_UC_META))
            shape = layer._expert_logical_shape(parameter)
            metadata[AUTOEP_PARAM_LOGICAL_SHAPE] = shape
            metadata[AFFINE_MAP] = autoep_placement_to_affine_map(layer.expert_placement_descriptor, shape).to_dict()
            setattr(parameter, DS_AUTOEP_UC_META, metadata)
        experts.add_module(f'lora{index}', adapter)
    experts.forward = types.MethodType(expert_forward, experts)
    if layer.shared_experts is not None:
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            base = getattr(layer.shared_experts, name)
            adapter = LinearAdapter(base, rank, alpha)
            for parameter in adapter.parameters():
                parameter.allreduce = True
                parameter.ds_zero_placement_family = 'replicated'
            setattr(layer.shared_experts, name, adapter)


def install():
    import deepspeed
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
    rank = int(os.environ.get('EXPERT_LORA_RANK', '8'))
    if rank not in (8, 16):
        raise ValueError('This experiment validates LoRA ranks 8 and 16')
    original_init = AutoEPMoELayer.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        attach_adapters(self, rank)

    AutoEPMoELayer.__init__ = init
    original_initialize = deepspeed.initialize

    @functools.wraps(original_initialize)
    def initialize(*args, **kwargs):
        if args:
            raise RuntimeError('Experiment expects keyword-only deepspeed.initialize')
        if kwargs.get('optimizer') is not None:
            raise RuntimeError('Configure the optimizer through DeepSpeed JSON for adapter discovery')
        # Iterate only after AutoEP replacement. Frozen packed weights must not
        # create MoE optimizer groups which become empty inside ZeRO.
        kwargs['model_parameters'] = (p for p in kwargs['model'].parameters() if p.requires_grad)
        return original_initialize(**kwargs)

    deepspeed.initialize = initialize


if os.environ.get('EXPERT_LORA_ENABLE') == '1':
    install()
