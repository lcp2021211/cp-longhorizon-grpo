from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from verl.models.transformers.qwen3_5 import _ensure_text_rotary_buffers_on_device, _get_input_embeds
from verl.workers.fsdp_workers import _align_lora_adapter_dtype, get_vl_model_vision_tower


class _FakeVisual(torch.nn.Module):
    def __init__(self, *, trainable: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=trainable)
        self.called = False

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, pixel_values, grid_thw):
        self.called = True
        return SimpleNamespace(pooler_output=self.weight.expand(1, 2))


class _FakeModel(torch.nn.Module):
    def __init__(self, *, vision_trainable: bool):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 2)
        self.visual = _FakeVisual(trainable=vision_trainable)
        self.config = SimpleNamespace(
            image_token_id=30,
            video_token_id=31,
            vision_config=SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=2),
        )

    def get_input_embeddings(self):
        return self.embedding


def test_text_only_forward_skips_frozen_vision_tower():
    model = _FakeModel(vision_trainable=False)

    result = _get_input_embeds(model, torch.tensor([[1, 2, 3]]))

    assert result["inputs_embeds"].shape == (1, 3, 2)
    assert model.visual.called is False


def test_text_only_forward_keeps_dummy_pass_for_trainable_vision_tower():
    model = _FakeModel(vision_trainable=True)

    _get_input_embeds(model, torch.tensor([[1, 2, 3]]))

    assert model.visual.called is True


def test_vision_tower_lookup_unwraps_peft_style_layers():
    visual = _FakeVisual(trainable=True)
    transformer = SimpleNamespace(visual=visual)
    lora_model = SimpleNamespace(model=transformer)
    peft_model = SimpleNamespace(model=lora_model, base_model=lora_model)

    assert get_vl_model_vision_tower(peft_model) is visual


def test_lora_adapter_dtype_is_aligned_before_fsdp2_wrapping():
    module = torch.nn.Module()
    module.register_parameter(
        "frozen_base", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
    )
    module.register_parameter("lora_adapter", torch.nn.Parameter(torch.ones(2, dtype=torch.float32)))

    aligned_count, target_dtype = _align_lora_adapter_dtype(module)

    assert aligned_count == 1
    assert target_dtype == torch.bfloat16
    assert module.frozen_base.dtype == torch.bfloat16
    assert module.frozen_base.requires_grad is False
    assert module.lora_adapter.dtype == torch.bfloat16
    assert module.lora_adapter.requires_grad is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to reproduce CPU-offloaded RoPE buffers")
def test_text_rotary_buffers_follow_cuda_inputs():
    config = Qwen3_5TextConfig(hidden_size=32, num_attention_heads=4, head_dim=8)
    rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    language_model = SimpleNamespace(rotary_emb=rotary_emb)
    device = torch.device("cuda")

    _ensure_text_rotary_buffers_on_device(language_model, device)
    cos, sin = rotary_emb(
        torch.zeros((1, 3, 32), device=device),
        torch.arange(3, device=device).unsqueeze(0),
    )

    assert all(buffer.device.type == "cuda" for buffer in rotary_emb.buffers())
    assert cos.device.type == "cuda"
    assert sin.device.type == "cuda"


def test_rollout_preloads_qwen35_base_before_lora_sync():
    project_dir = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_dir / "configs/train/grpo/agentic_ablation_tau2.yaml").read_text(encoding="utf-8")
    )

    assert config["actor_rollout_ref"]["rollout"]["load_format"] == "safetensors"
