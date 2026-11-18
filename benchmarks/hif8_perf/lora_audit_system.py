# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Audit the combined optimization stack and actual expert-load imbalance."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.distributed as dist
import torch_npu
import torchao_npu
from swift import callbacks_map
from lora_audit_fused_v2 import FingerprintedFusedAudit
from fused_gate_dispatch import PAYLOAD_STATISTICS


class SystemAudit(FingerprintedFusedAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        assert Path(torchao_npu.__file__).resolve().is_relative_to(Path(os.environ['HIF8_PERF_AO_ROOT']).resolve())
        self.routing_totals = torch.zeros(48, 256, device='npu', dtype=torch.int64)
        self.routing_steps = 0
        for name, layer in self.layers:
            index = int(re.search(r'\.layers\.(\d+)\.', name).group(1))
            assert index == layer._system_layer_index
            if self.engine.hifloat8_enabled():
                cfg = layer.experts.grouped_mm.config
                layer.experts.grouped_mm.config = replace(cfg, reuse_quantized_weight=True)
        for name in ('npu_quantize', 'npu_dtype_cast'):
            original = getattr(torch_npu, name)

            def counted(*a, _original=original, _name=name, **kw):
                self.counts[_name] += 1
                operand = a[0] if a else kw['input']
                self.counts[_name + '_input_elements'] += operand.numel()
                return _original(*a, **kw)

            setattr(torch_npu, name, counted)
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['system_stack'] = dict(
            payload_reuse=True,
            expert_placement='profile-greedy' if os.environ.get('SYSTEM_ROUTING_PROFILE') else 'interleaved',
            lora_scale_on_inner=True,
            checkpoint_use_reentrant=False,
            gate_dispatch_fusion=True,
            up_payload_reuse=True,
            buffer_reuse=True,
            preserve_original_router_reductions_and_ties=True)
        manifest['expert_new_to_original'] = {
            name: layer._system_new_to_original.tolist()
            for name, layer in self.layers
        }
        manifest['ao_source'] = torchao_npu.__file__
        profile = os.environ.get('SYSTEM_ROUTING_PROFILE')
        manifest['routing_profile_sha256'] = hashlib.sha256(
            Path(profile).read_bytes()).hexdigest() if profile else None
        for source in (Path(__file__), Path(__file__).with_name('system_stack.py'),
                       Path(torchao_npu.__file__).parent / 'features/training/hifloat8/config.py',
                       Path(torchao_npu.__file__).parent / 'features/training/hifloat8/hifloat8_gmm.py'):
            manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))

    def on_step_begin(self, *args, **kwargs):
        PAYLOAD_STATISTICS.clear()
        return super().on_step_begin(*args, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        counts = torch.stack([layer._system_last_counts.to(torch.int64) for _, layer in self.layers])
        dist.all_reduce(counts)
        loads = counts.reshape(48, 8, 32).sum(-1)
        ratios = loads.max(1).values.float() / loads.float().mean(1).clamp_min(1)
        original = torch.stack([
            row.index_select(0,
                             torch.argsort(layer._system_new_to_original).to(row.device))
            for row, (_, layer) in zip(counts, self.layers)
        ])
        original_loads = original.reshape(48, 8, 32).sum(-1)
        old_ratios = original_loads.max(1).values.float() / original_loads.float().mean(1).clamp_min(1)
        self.routing_totals.add_(original)
        self.routing_steps += 1
        record = dict(step=state.global_step,
                      rank=self.rank,
                      mean_layer_max_over_mean=float(ratios.mean()),
                      max_layer_max_over_mean=float(ratios.max()),
                      original_contiguous_mean_max_over_mean=float(old_ratios.mean()),
                      original_contiguous_max_max_over_mean=float(old_ratios.max()),
                      payload_statistics=dict(PAYLOAD_STATISTICS),
                      per_layer_received_rows=loads.cpu().tolist())
        if self.engine.hifloat8_enabled():
            assert PAYLOAD_STATISTICS['fused_weight_payload_reuses'] >= 48 * 2
        with (self.root / f'system_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        return super().on_step_end(args, state, control, **kwargs)

    def on_train_end(self, *args, **kwargs):
        result = super().on_train_end(*args, **kwargs)
        if self.rank == 0:
            counts = self.routing_totals.cpu().tolist()
            profile = dict(steps=self.routing_steps,
                           layers={
                               str(i): dict(counts_original=row)
                               for i, row in enumerate(counts)
                           })
            (self.root / 'routing_profile.json').write_text(json.dumps(profile, indent=2))
        return result


callbacks_map['expert_lora_audit'] = SystemAudit
