# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Record activation of the HiFloat8 FC2 and production offload paths."""
import hashlib
import json
from pathlib import Path

from swift import callbacks_map
from torchao.hifloat8 import HiFloat8Linear

from fc2_combine import COUNTS
from native_weighted_restore import install

Base = callbacks_map["expert_lora_audit"]


class Fc2Audit(Base):

    def on_train_begin(self, *args, **kwargs):
        result = super().on_train_begin(*args, **kwargs)
        assert len(self.layers) == 48
        linears = [
            layer for layer in self.model.modules() if type(layer) is HiFloat8Linear and not layer.weight.requires_grad
        ]
        assert len(linears) == 132
        assert all(layer.weight.device.type == "cpu" and layer._host_weight_payload is not None for layer in linears)
        install()

        path = self.root / f"manifest_rank{self.rank}.json"
        data = json.loads(path.read_text())
        data["production_offload"] = {
            "weights": len(linears),
            "host_bytes": sum(layer.weight.numel() * layer.weight.element_size() for layer in linears),
        }
        data["fc2_experiment"] = dict(fused=True, fused_dx=True, native_weighted_restore=True)
        for name in ("fc2_audit.py", "fc2_combine.py", "fc2_dx_dispatch.py", "native_weighted_restore.py",
                     "system_source_stack.py", "expert_after_gate.py"):
            file = Path(__file__).with_name(name)
            data["source_sha256"][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        path.write_text(json.dumps(data, indent=2))
        return result

    def on_step_begin(self, *args, **kwargs):
        COUNTS.clear()
        return super().on_step_begin(*args, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        assert COUNTS["fused_down_return"] >= 48 and COUNTS["fused_down_input_grad"] == 48, COUNTS
        assert COUNTS["native_weighted_restore"] >= 48, COUNTS
        with (self.root / f"fc2_rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=state.global_step, counts=dict(COUNTS))) + "\n")
        return super().on_step_end(args, state, control, **kwargs)


callbacks_map["expert_lora_audit"] = Fc2Audit
