# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Combined EP/LoRA experiment, independent of the official repair checkout.

Expert IDs are renamed together with router rows and all expert parameters.
The model function and executed expert count are preserved. The placement is
fixed before optimizer steps; live migration and checkpoint resume are excluded.
"""
import functools
import itertools
import json
import os
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.ops.npu.grouped_mm import grouped_mm


def inner_scaled_grouped(self, x, offsets):
    inner = grouped_mm(x, self.a, offsets, trans_b=True).mul_(self.scale)
    return grouped_mm(inner, self.b, offsets, trans_b=True)


def inner_scaled_linear(self, x):
    inner = F.linear(x, self.a).mul_(self.scale)
    return self.base(x).add_(F.linear(inner, self.b))


def install_inner_scales(layer):
    # External plugin loaders may use their own module names. Bind the actual
    # instances rather than importing and re-running their installer modules.
    for i in (1, 2, 3):
        adapter = getattr(layer.experts, f'lora{i}')
        adapter.forward = types.MethodType(inner_scaled_grouped, adapter)
    if layer.shared_experts is not None:
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            adapter = getattr(layer.shared_experts, name)
            adapter.forward = types.MethodType(inner_scaled_linear, adapter)


def remap_expert_tensor(tensor, permutation, rank, world, group):
    local = tensor.shape[0]
    inverse = torch.argsort(permutation)
    owned = torch.arange(rank * local, (rank + 1) * local)
    destinations = inverse[owned] // local
    send_order = torch.argsort(destinations, stable=True)
    send = torch.bincount(destinations, minlength=world).tolist()
    wanted = permutation[rank * local:(rank + 1) * local]
    recv = torch.bincount(wanted // local, minlength=world).tolist()
    # Sender packs preserve ascending original IDs inside each destination.
    recv_order = torch.searchsorted(wanted.sort().values, wanted)
    # AutoEP replacement precedes optimizer construction and may still own
    # CPU tensors. Stage one parameter at a time through the NPU for HCCL.
    work = tensor if tensor.device.type == 'npu' else tensor.to('npu')
    packed = work.index_select(0, send_order.to(work.device)).contiguous()
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, output_split_sizes=recv, input_split_sizes=send, group=group)
    return received.index_select(0, recv_order.to(work.device)).to(tensor.device)


def restripe_experts(layer, permutation):
    world = layer.ep_size
    rank = dist.get_rank(group=layer.ep_group)
    local = layer.num_local_experts
    if permutation.shape != (world *
                             local, ) or not torch.equal(permutation.sort().values, torch.arange(world * local)):
        raise ValueError('Expert placement must be a permutation of every expert')
    if layer.router.num_expert_groups is not None:
        raise ValueError('The experiment does not rename node-limited expert groups')
    if layer.expert_bias is not None:
        raise ValueError('The experiment does not migrate dynamically updated expert biases')
    experts = layer.experts
    tensors = [getattr(experts, f'w{i}') for i in (1, 2, 3)]
    tensors.extend(parameter for i in (1, 2, 3) for parameter in getattr(experts, f'lora{i}').parameters())
    with torch.no_grad():
        for value in tensors:
            value.copy_(remap_expert_tensor(value, permutation, rank, world, layer.ep_group))
        index = permutation.to(layer.router.gate.weight.device)
        layer.router.gate.weight.copy_(layer.router.gate.weight.index_select(0, index))
        if layer.router.gate.bias is not None:
            layer.router.gate.bias.copy_(layer.router.gate.bias.index_select(0, index))
    install_router_mapping(layer.router, permutation)
    layer._system_new_to_original = permutation


def restore_original_logit_order(module, args, output):
    return output.index_select(-1, module._system_original_to_physical)


def map_router_choices(module, args, output):
    scores, indices, counts = output
    gate = module.gate
    mapped = gate._system_original_to_physical.index_select(0, indices.reshape(-1)).reshape_as(indices)
    return scores, mapped, counts.index_select(0, gate._system_physical_to_original)


def install_router_mapping(router, permutation):
    # Restore original column order before softmax/top-k. Renaming expert IDs
    # must not change floating-point reductions or which expert wins a tie.
    device = router.gate.weight.device
    router.gate.register_buffer('_system_original_to_physical', torch.argsort(permutation).to(device))
    router.gate.register_buffer('_system_physical_to_original', permutation.to(device))
    router.gate.register_forward_hook(restore_original_logit_order)
    router.register_forward_hook(map_router_choices)


def balanced_placement(counts, world, local):
    """Greedy placement with exactly local experts on every rank."""
    if len(counts) != world * local:
        raise ValueError('Routing profile has the wrong expert count')
    bins = [[] for _ in range(world)]
    loads = [0] * world
    for expert in sorted(range(len(counts)), key=lambda e: (-counts[e], e)):
        target = min((r for r in range(world) if len(bins[r]) < local), key=lambda r: (loads[r], r))
        bins[target].append(expert)
        loads[target] += counts[expert]
    return torch.tensor([expert for bucket in bins for expert in sorted(bucket)])


def measured_dispatch(self, routed_input, plan, global_counts):
    self._system_last_counts = global_counts.detach()
    return self._system_dispatch(routed_input, plan, global_counts)


original_init = AutoEPMoELayer.__init__
layer_indices = itertools.count()
profile_path = os.environ.get('SYSTEM_ROUTING_PROFILE')
routing_profile = json.loads(Path(profile_path).read_text()) if profile_path else None


@functools.wraps(original_init)
def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    index = next(layer_indices)
    if routing_profile is None:
        permutation = torch.arange(self.num_experts).reshape(self.num_local_experts, self.ep_size).T.flatten()
    else:
        permutation = balanced_placement(routing_profile['layers'][str(index)]['counts_original'], self.ep_size,
                                         self.num_local_experts)
    restripe_experts(self, permutation)
    self._system_layer_index = index
    install_inner_scales(self)
    self._system_dispatch = self.experimental_expert_dispatch
    self.experimental_expert_dispatch = types.MethodType(measured_dispatch, self)


AutoEPMoELayer.__init__ = init
