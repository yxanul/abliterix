from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from abliterix.core.vllm_backend import ProjectionCache
from abliterix.types import DecayKernel, SteeringProfile


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(3, 4, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer()])


class _ToyEngine:
    def __init__(self) -> None:
        self.model = _ToyModel()
        self.transformer_layers = list(self.model.layers)

    def steerable_modules(self, layer_idx: int):
        layer = self.model.layers[layer_idx]
        return {"attn.o_proj": [layer.o_proj]}


def test_projection_cache_build_accepts_plain_linear_without_base_layer():
    engine = _ToyEngine()
    steering_vectors = torch.randn(2, 4)

    cache = ProjectionCache.build(engine, steering_vectors)

    info = cache.projections[0]["attn.o_proj"]
    assert info["module_path"] == "layers.0.o_proj"
    assert info["direction"] == "output"
    assert info["vW_all"].shape == (2, 3)
    assert cache.target_modules == ["o_proj"]


def test_projection_cache_builds_rank_k_lora_for_multi_direction_vectors():
    engine = _ToyEngine()
    steering_vectors = torch.randn(3, 2, 4)

    cache = ProjectionCache.build(engine, steering_vectors)
    info = cache.projections[0]["attn.o_proj"]

    assert info["vW_all"].shape == (3, 2, 3)

    lora_weights = cache.build_lora_weights(
        {
            "attn.o_proj": SteeringProfile(
                max_weight=2.0,
                max_weight_position=0.0,
                min_weight=2.0,
                min_weight_distance=1.0,
            )
        },
        vector_index=0.5,  # Ignored for multi-direction vectors.
        config=SimpleNamespace(
            steering=SimpleNamespace(decay_kernel=DecayKernel.LINEAR)
        ),
    )

    lora_a, lora_b = lora_weights["layers.0.o_proj"]
    assert lora_a.shape == (3, 3)
    assert lora_b.shape == (4, 3)


def test_projection_cache_steerable_components_excludes_zero_companions():
    cache = ProjectionCache()
    cache.projections = {
        0: {
            "attn.o_proj": {"vW_all": torch.empty(2, 3)},
            "mlp.down_proj": {"experts": []},
            "moe.expert_gate": {"companions": []},
            "moe.expert_up": {"companions": []},
        }
    }

    assert cache.steerable_components() == ["attn.o_proj", "mlp.down_proj"]


def test_projection_cache_safetensors_dequants_compressed_tensors_weight_scale(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    weight_f32 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scale = torch.tensor([[0.5], [2.0]])
    save_file(
        {
            "model.layers.0.self_attn.o_proj.weight": weight_f32.to(
                torch.float8_e4m3fn
            ),
            "model.layers.0.self_attn.o_proj.weight_scale": scale,
        },
        model_dir / "model.safetensors",
    )
    (model_dir / "config.json").write_text(
        """
        {
          "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized"
          }
        }
        """,
        encoding="utf-8",
    )

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=1,
                quantization_config={
                    "quant_method": "compressed-tensors",
                    "format": "float-quantized",
                },
            )
        ),
    )

    config = SimpleNamespace(
        model=SimpleNamespace(model_id=str(model_dir), trust_remote_code=False),
        steering=SimpleNamespace(disabled_components=set()),
    )
    steering_vectors = torch.eye(2)

    cache = ProjectionCache.build_from_safetensors(config, steering_vectors)

    info = cache.projections[0]["attn.o_proj"]
    expected_w = weight_f32 * scale
    torch.testing.assert_close(info["vW_all"], expected_w)
    assert info["module_path"] == "model.layers.0.self_attn.o_proj"
    assert info["direction"] == "output"
    assert cache.target_modules == ["o_proj"]


def test_projection_cache_safetensors_preserves_multi_direction_projection_shape(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    save_file(
        {
            "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 3),
        },
        model_dir / "model.safetensors",
    )
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=1)
        ),
    )

    config = SimpleNamespace(
        model=SimpleNamespace(model_id=str(model_dir), trust_remote_code=False),
        steering=SimpleNamespace(disabled_components=set()),
    )
    steering_vectors = torch.randn(3, 2, 4)

    cache = ProjectionCache.build_from_safetensors(config, steering_vectors)

    info = cache.projections[0]["attn.o_proj"]
    assert info["vW_all"].shape == (3, 2, 3)
    assert info["direction"] == "output"
