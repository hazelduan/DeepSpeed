# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Add initial frozen-tensor fingerprints to the matched fusion experiment."""
import hashlib
import json
import os
from pathlib import Path

import torch
from swift import callbacks_map
from lora_audit_fused import FusedExpertLoRAAudit


def fingerprint(tensor):
    with torch.no_grad():
        full = tensor.numel() <= 65536 or os.environ.get('PRECISION_FULL_FINGERPRINTS') == '1'
        if full:
            sample = tensor.detach().cpu().contiguous()
        else:
            # Index before copying: a reshape of a strided packed expert weight
            # could otherwise materialize the entire large BF16 tensor.
            samples = []
            for position in (0, 1, 2):
                prefix = tuple(0 if position == 0 else (n // 2 if position == 1 else n - 1) for n in tensor.shape[:-1])
                row = tensor[prefix]
                start = 0 if position == 0 else (max(0,
                                                     row.numel() // 2 -
                                                     16) if position == 1 else max(0,
                                                                                   row.numel() - 32))
                samples.append(row[start:start + 32])
            sample = torch.cat(samples).detach().contiguous().cpu()
        digest = hashlib.sha256(sample.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest, full=full)


class FingerprintedFusedAudit(FusedExpertLoRAAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        tensors = {'parameter:' + name: p for name, p in self.model.named_parameters() if not p.requires_grad}
        tensors.update({'buffer:' + name: p for name, p in self.model.named_buffers()})
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['frozen_tensor_fingerprints'] = {name: fingerprint(value) for name, value in tensors.items()}
        source = Path(__file__)
        manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))


callbacks_map['expert_lora_audit'] = FingerprintedFusedAudit
