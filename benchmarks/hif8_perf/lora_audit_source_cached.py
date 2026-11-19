# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Combined source stack plus bounded cross-step frozen Linear payload reuse."""
import hashlib
import json
import os
from pathlib import Path
from swift import callbacks_map
from lora_audit_source import SourceAudit
from frozen_linear_payload_cache import FrozenLinearPayloadCache


class CachedSourceAudit(SourceAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        budget = int(os.environ.get('DENSE_LINEAR_CACHE_BYTES', str(4 * 2**30)))
        self.linear_cache = None
        if self.engine.hifloat8_enabled() and budget:
            self.linear_cache = FrozenLinearPayloadCache(self.model, budget)
            self.linear_cache.install()
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['system_stack']['dense_linear_payload_cache_budget_bytes'] = budget
        manifest['dense_linear_cache_initial'] = self.linear_cache.statistics() if self.linear_cache else None
        for filename in ('lora_audit_source_cached.py', 'frozen_linear_payload_cache.py'):
            source = Path(__file__).with_name(filename)
            manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))

    def on_step_end(self, args, state, control, **kwargs):
        stats = self.linear_cache.statistics() if self.linear_cache else None
        quantized = self.counts['npu_quantize_input_elements']
        wire = self.counts['npu_alltoallv_quant_gmm_input_elements']
        if self.engine.hifloat8_enabled():
            from fc2_combine import COUNTS

            quantized -= COUNTS["down_grad_quantized_elements"]
            wire -= COUNTS["down_grad_quantized_elements"]
            assert quantized * self.layers[0][1].top_k == wire, 'Source tokens were redundantly quantized'
        if stats:
            assert stats['bytes'] <= stats['budget_bytes']
        row = dict(step=state.global_step,
                   rank=self.rank,
                   dense_cache=stats,
                   quantized_source_elements=quantized,
                   dispatched_payload_elements=wire)
        with (self.root / f'source_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return super().on_step_end(args, state, control, **kwargs)

    def on_train_end(self, *args, **kwargs):
        try:
            return super().on_train_end(*args, **kwargs)
        finally:
            if self.linear_cache:
                self.linear_cache.uninstall()


callbacks_map['expert_lora_audit'] = CachedSourceAudit
