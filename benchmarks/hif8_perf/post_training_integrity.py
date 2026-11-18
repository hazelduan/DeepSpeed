# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Attach complete frozen-weight verification to either experiment arm."""
import hashlib
import json
from pathlib import Path
from swift import callbacks_map
from frozen_integrity import verify_frozen_weights

BaseAudit = callbacks_map['expert_lora_audit']


class PostTrainingIntegrityAudit(BaseAudit):

    def on_train_begin(self, *args, **kwargs):
        result = super().on_train_begin(*args, **kwargs)
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['post_training_frozen_integrity_required'] = True
        for source in (Path(__file__), Path(__file__).with_name('frozen_integrity.py')):
            manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))
        return result

    def on_train_end(self, *args, **kwargs):
        verify_frozen_weights(self)
        return super().on_train_end(*args, **kwargs)


callbacks_map['expert_lora_audit'] = PostTrainingIntegrityAudit
