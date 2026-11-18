# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Bounded, same-stream payload cache for registered immutable dense weights.

Only converted HiFloat8Linear parameters are eligible. Activations and
trainable weights always use the normal quantizer. The cache is process-local
and excluded from checkpoints; version changes invalidate retained payloads.
"""
from collections import OrderedDict
import importlib
import weakref
import torch


class FrozenLinearPayloadCache:

    def __init__(self, model, budget_bytes=4 * 2**30):
        self.module = importlib.import_module('torchao_npu.features.training.hifloat8.hifloat8_linear')
        self.original = self.module._quantize_data
        self.budget = budget_bytes
        self.registry = {}
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.hits = self.misses = self.evictions = self.invalidations = self.cross_stream_bypasses = 0
        for layer in model.modules():
            if isinstance(layer, self.module.HiFloat8Linear) and not layer.weight.requires_grad:
                self.registry[self.identity(layer.weight)] = weakref.ref(layer.weight)
        self.active = False

    @staticmethod
    def identity(tensor):
        return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device

    def statistics(self):
        return dict(registered_weights=len(self.registry),
                    entries=len(self.entries),
                    bytes=self.bytes,
                    peak_bytes=self.peak_bytes,
                    budget_bytes=self.budget,
                    hits=self.hits,
                    misses=self.misses,
                    evictions=self.evictions,
                    invalidations=self.invalidations,
                    cross_stream_bypasses=self.cross_stream_bypasses)

    def install(self):
        if self.active:
            raise RuntimeError('Dense weight payload cache is already installed')

        def quantize(tensor, *, dst_type_max=None):
            reference = self.registry.get(self.identity(tensor))
            owner = reference() if reference is not None else None
            if owner is None or owner.requires_grad or tensor.requires_grad:
                return self.original(tensor, dst_type_max=dst_type_max)
            key = id(owner), dst_type_max
            stream = torch.npu.current_stream(tensor.device)
            entry = self.entries.get(key)
            if entry is not None and entry['version'] == owner._version:
                if entry['stream'] != stream:
                    self.cross_stream_bypasses += 1
                    return self.original(tensor, dst_type_max=dst_type_max)
                self.entries.move_to_end(key)
                self.hits += 1
                return entry['value']
            if entry is not None:
                self.bytes -= entry['bytes']
                self.entries.pop(key)
                self.invalidations += 1
            self.misses += 1
            value = self.original(tensor, dst_type_max=dst_type_max)
            size = sum(t.numel() * t.element_size() for t in value)
            if size <= self.budget:
                while self.bytes + size > self.budget:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old['bytes']
                    self.evictions += 1
                self.entries[key] = dict(version=owner._version, stream=stream, value=value, bytes=size)
                self.bytes += size
                self.peak_bytes = max(self.peak_bytes, self.bytes)
            return value

        self.wrapper = quantize
        self.module._quantize_data = quantize
        self.active = True

    def uninstall(self):
        if self.active:
            if self.module._quantize_data is not self.wrapper:
                raise RuntimeError('Another wrapper replaced the dense quantizer')
            self.module._quantize_data = self.original
            self.active = False
