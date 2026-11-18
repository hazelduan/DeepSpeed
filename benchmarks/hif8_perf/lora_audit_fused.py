# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Keep the existing objective and audit the experimental fused operator calls."""
import hashlib
import json
from pathlib import Path

import torch_npu
from swift import callbacks_map
from lora_audit_v2 import ExpertLoRAAudit


class FusedExpertLoRAAudit(ExpertLoRAAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        assert all(hasattr(layer, 'experimental_expert_dispatch') for _, layer in self.layers)
        for name in ('npu_alltoallv_gmm', 'npu_alltoallv_quant_gmm'):
            original = getattr(torch_npu, name)

            def counted(*a, _original=original, _name=name, **kw):
                self.counts[_name] += 1
                operand = a[0] if a else kw['gmm_x']
                self.counts[_name + '_input_elements'] += operand.numel()
                return _original(*a, **kw)

            setattr(torch_npu, name, counted)
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['fused_gate_dispatch'] = True
        manifest['gate_input_scale'] = 'EP-wide' if self.engine.hifloat8_enabled() else None
        manifest['lora_input'] = 'decoded-dispatched-HiFloat8' if self.engine.hifloat8_enabled() else 'original-BF16'
        for filename in ('lora_audit_fused.py', 'fused_gate_dispatch.py', 'fused_gate_lora_plugin.py',
                         'expert_after_gate.py'):
            source = Path(__file__).with_name(filename)
            manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))

    def on_step_end(self, *args, **kwargs):
        name = 'npu_alltoallv_quant_gmm' if self.engine.hifloat8_enabled() else 'npu_alltoallv_gmm'
        assert self.counts[name] >= 48, 'The fused dispatch path was not exercised on all layers'
        return super().on_step_end(*args, **kwargs)


callbacks_map['expert_lora_audit'] = FusedExpertLoRAAudit
