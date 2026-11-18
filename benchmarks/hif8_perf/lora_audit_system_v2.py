# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Combined-stack audit with canonical initial expert identities."""
import hashlib
import json
from pathlib import Path
from swift import callbacks_map
from lora_audit_system import SystemAudit
from canonical_provenance import record_initial


class CanonicalSystemAudit(SystemAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        record_initial(self)
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['source_sha256'][__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))


callbacks_map['expert_lora_audit'] = CanonicalSystemAudit
