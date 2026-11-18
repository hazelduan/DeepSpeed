# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Source-side packing on top of the combined-stack audit."""
import hashlib
import json
from pathlib import Path
from swift import callbacks_map
from lora_audit_system_v2 import CanonicalSystemAudit


class SourceAudit(CanonicalSystemAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        assert all(hasattr(layer, 'experimental_source_dispatch') for _, layer in self.layers)
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['system_stack'].update(source_quant_before_duplication=True,
                                        native_expert_sort=True,
                                        fp32_route_gradient_restore=True)
        for filename in ('lora_audit_source.py', 'source_gate_dispatch.py', 'source_pack.py',
                         'system_source_stack.py'):
            source = Path(__file__).with_name(filename)
            manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))


callbacks_map['expert_lora_audit'] = SourceAudit
