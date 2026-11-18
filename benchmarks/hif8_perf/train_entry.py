# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Experiment-only memory budget before Swift materializes the model."""
from pathlib import Path
import json
import os
import runpy
import sys

import torch
import torch_npu  # noqa: F401 (registers the NPU backend)

local = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])
torch.npu.set_device(local)
torch.npu.set_compile_mode(jit_compile=False)
fraction = float(os.environ.get('HIF8_MEMORY_FRACTION', '0.85'))
torch.npu.set_per_process_memory_fraction(fraction, local)
free, total = torch.npu.mem_get_info(local)
out = Path(os.environ['HIF8_RUN_OUTPUT'])
record = {
    'rank': rank,
    'physical_card': os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')[local],
    'fraction_cap': fraction,
    'free_bytes': free,
    'total_bytes': total
}
(out / f'memory_preflight_rank{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
minimum = float(os.environ.get('HIF8_MIN_FREE_GIB', '105'))
if free < minimum * 1024**3:
    raise RuntimeError(f'Only {free/1024**3:.2f}GiB free; model preflight requires {minimum}GiB')
# Match the successful historical probe: initialize and exercise HCCL before
# Swift imports/materializes the model. This preflight is outside training timing.
import deepspeed
import deepspeed.comm as dist

deepspeed.init_distributed(dist_backend='hccl')
probe = torch.tensor([rank + 1.], device='npu')
dist.all_reduce(probe)
assert probe.item() == int(os.environ['WORLD_SIZE']) * (int(os.environ['WORLD_SIZE']) + 1) / 2
record['premodel_allreduce'] = True
(out / f'memory_preflight_rank{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
del probe
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
