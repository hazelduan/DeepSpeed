# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--node-rank', type=int, required=True)
parser.add_argument('--nnodes', type=int, default=2)
parser.add_argument('--local-world', type=int, default=4)
parser.add_argument('--master-addr', required=True)
parser.add_argument('--master-port', type=int, required=True)
parser.add_argument('--log-dir', type=Path, required=True)
parser.add_argument('script')
parser.add_argument('script_args', nargs=argparse.REMAINDER)
args = parser.parse_args()
args.log_dir.mkdir(parents=True, exist_ok=False)
children = []
streams = []


def stop():
    for process in children:
        if process.poll() is None: process.terminate()
    for process in children:
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def interrupted(signum, frame):
    stop()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
try:
    for local_rank in range(args.local_world):
        rank = args.node_rank * args.local_world + local_rank
        env = os.environ.copy()
        env.update(RANK=str(rank),
                   LOCAL_RANK=str(local_rank),
                   WORLD_SIZE=str(args.nnodes * args.local_world),
                   LOCAL_WORLD_SIZE=str(args.local_world),
                   GROUP_RANK=str(args.node_rank),
                   GROUP_WORLD_SIZE=str(args.nnodes),
                   MASTER_ADDR=args.master_addr,
                   MASTER_PORT=str(args.master_port),
                   TORCH_DEVICE_BACKEND_AUTOLOAD='1')
        stream = (args.log_dir / f'rank{rank}.log').open('w')
        streams.append(stream)
        children.append(
            subprocess.Popen([sys.executable, args.script, *args.script_args],
                             env=env,
                             stdin=subprocess.DEVNULL,
                             stdout=stream,
                             stderr=subprocess.STDOUT))
    ranks = {str(args.node_rank * args.local_world + i): p.pid for i, p in enumerate(children)}
    (args.log_dir / 'processes.json'
     ).write_text(json.dumps({
         'parent': os.getpid(),
         'node_rank': args.node_rank,
         'ranks': ranks
     }, indent=2) + '\n')
    while True:
        codes = [p.poll() for p in children]
        if any(code is not None and code != 0 for code in codes): raise RuntimeError(f'Worker failure {codes}')
        if all(code == 0 for code in codes): break
        time.sleep(0.5)
    (args.log_dir / 'exit.json').write_text(json.dumps({'workers': codes}) + '\n')
finally:
    stop()
    for stream in streams:
        stream.close()
