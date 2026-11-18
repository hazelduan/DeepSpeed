# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Record monotonic step intervals for auditing shared-resource measurements."""
import hashlib
import json
from pathlib import Path
from swift import callbacks_map

BaseAudit = callbacks_map['expert_lora_audit']


class ResourceWindowAudit(BaseAudit):

    def on_train_begin(self, *args, **kwargs):
        result = super().on_train_begin(*args, **kwargs)
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['resource_step_windows'] = True
        manifest['source_sha256'][__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))
        return result

    def on_step_end(self, args, state, control, **kwargs):
        start = self.previous_end if self.previous_end is not None else self.begin
        result = super().on_step_end(args, state, control, **kwargs)
        with (self.root / f'resource_steps_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(
                json.dumps({
                    'step': state.global_step,
                    'start_monotonic': start,
                    'end_monotonic': self.previous_end
                }) + '\n')
        return result


callbacks_map['expert_lora_audit'] = ResourceWindowAudit
