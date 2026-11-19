# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Extend the combined stack with source quantization and native FP32 ID sort."""
import functools
import types
import torch
from deepspeed import comm as dist
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from source_gate_dispatch import SourceGateDispatch
from source_pack import sorted_assignments
from expert_after_gate import expert_after_gate


def source_dispatch(self, x, order, plan, counts):
    self._system_last_counts = counts.detach()
    config = getattr(self.experts.grouped_mm, 'config', None)
    sent = counts.reshape(self.ep_size, self.num_local_experts).cpu().tolist()
    received = plan.local_counts_by_source.cpu().tolist()
    if not hasattr(self, '_fused_gate_hcom'):
        self._fused_gate_hcom = self.ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(dist.get_rank())
    gate, tokens, payload, scale, inverse = SourceGateDispatch.apply(x, self.experts.w1, self.ep_group,
                                                                     self._fused_gate_hcom, sent, received, config,
                                                                     order, self.top_k)
    return expert_after_gate(self, gate, tokens, payload, scale, inverse, plan, config, (sent, received))


original_init = AutoEPMoELayer.__init__


@functools.wraps(original_init)
def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self.score_apply != 'post' or self.comm_backend != 'comm':
        raise ValueError('Source packing currently requires post-scored EP communication')
    self.experimental_sort_assignments = functools.partial(sorted_assignments, num_experts=self.num_experts)
    self.experimental_source_dispatch = types.MethodType(source_dispatch, self)


AutoEPMoELayer.__init__ = init
