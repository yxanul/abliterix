from __future__ import annotations

import os

import torch

from abliterix.core.vllm_backend import VLLMGenerator


class _DummyLLM:
    def __init__(self):
        self.reset_prefix_cache_calls = 0

    def reset_prefix_cache(self):
        self.reset_prefix_cache_calls += 1


def _make_generator(tmp_path) -> VLLMGenerator:
    gen = object.__new__(VLLMGenerator)
    gen._lora_disabled = False
    gen._adapter_dir = os.path.join(tmp_path, "current")
    gen._lora_max_rank = 1
    gen._adapter_id = 0
    gen._adapter_name = "steering_0"
    gen._lora_target_modules = []
    gen.llm = _DummyLLM()
    return gen


def test_save_adapter_assigns_fresh_lora_request_identity_per_trial(tmp_path):
    """vLLM caches LoRA adapters by request identity. Abliterix rewrites the
    same tmpfs adapter path every trial, so the request id/name must change or
    later trials can reuse the first loaded adapter."""

    gen = _make_generator(tmp_path)
    weights = {
        "layers.0.self_attn.o_proj": (
            torch.ones(1, 3, dtype=torch.float32),
            torch.ones(4, 1, dtype=torch.float32),
        )
    }

    path_1 = gen.save_adapter(weights, ["o_proj"], "dummy/base")
    args_1 = gen._lora_request_args(path_1)
    path_2 = gen.save_adapter(weights, ["o_proj"], "dummy/base")
    args_2 = gen._lora_request_args(path_2)

    assert path_1 == path_2
    assert args_1["lora_path"] == path_1
    assert args_2["lora_path"] == path_2
    assert args_1["lora_int_id"] == 1
    assert args_2["lora_int_id"] == 2
    assert args_1["lora_name"] == "steering_1"
    assert args_2["lora_name"] == "steering_2"
    assert args_1["lora_name"] != args_2["lora_name"]
    assert gen.llm.reset_prefix_cache_calls == 2
