# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Isolated LoRA experiment: native fused dispatch, reusable quantized input.

Requires the small dispatch-hook patch in the isolated DeepSpeed source copy.
The official ZeRO-2 repair checkout is unchanged.
"""
import functools
import types

import torch
import torch.distributed as dist
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from fused_gate_dispatch import FusedGateDispatch
from expert_after_gate import expert_after_gate


def dispatch_experts(self, routed_input, plan, global_counts):
    experts = self.experts
    config = getattr(experts.grouped_mm, 'config', None)
    sent = global_counts.reshape(self.ep_size, self.num_local_experts).cpu().tolist()
    received = plan.local_counts_by_source.cpu().tolist()
    if not hasattr(self, '_fused_gate_hcom'):
        self._fused_gate_hcom = self.ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(dist.get_rank())
    gate, tokens, payload, scale, inverse = FusedGateDispatch.apply(routed_input, experts.w1, self.ep_group,
                                                                    self._fused_gate_hcom, sent, received, config)
    return expert_after_gate(self, gate, tokens, payload, scale, inverse, plan, config)


original_init = AutoEPMoELayer.__init__


@functools.wraps(original_init)
def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self.ep_size != 8 or self.folding_group_handles is not None:
        raise ValueError('This isolated fusion experiment requires single-node EP8 without TP folding')
    if not all(hasattr(self.experts, f'lora{i}') for i in (1, 2, 3)):
        raise ValueError('Load the validated expert LoRA plugin before the fusion plugin')
    self.experimental_expert_dispatch = types.MethodType(dispatch_experts, self)


AutoEPMoELayer.__init__ = init
