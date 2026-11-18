# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Increase only HiFloat8's exact causal-CE block size to reduce launch overhead."""
import hashlib
import json
import os
from pathlib import Path

from swift import callbacks_map
from deepspeed.runtime.chunked_cross_entropy import ChunkedCausalLMLoss

BaseAudit = callbacks_map['expert_lora_audit']


class HiFloat8LossBlockAudit(BaseAudit):

    def on_train_begin(self, *args, **kwargs):
        result = super().on_train_begin(*args, **kwargs)
        assert self.engine.hifloat8_enabled()
        loss = self.model.loss_function
        assert isinstance(loss, ChunkedCausalLMLoss) and loss.backend == 'torch'
        before = loss.block_rows
        block_rows = int(os.environ.get('HIF8_CE_BLOCK_ROWS', '512'))
        assert block_rows in (128, 256, 512, 1024)
        loss.block_rows = block_rows
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['hif8_loss_block_rows'] = block_rows
        manifest['previous_loss_block_rows'] = before
        manifest['source_sha256'][__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))
        return result


callbacks_map['expert_lora_audit'] = HiFloat8LossBlockAudit
