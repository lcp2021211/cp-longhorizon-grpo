import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM

from verl.utils.checkpoint.lora_checkpoint import (
    COMPACT_LORA_METADATA,
    CompactLoraCheckpointError,
    prepare_compact_lora_resume,
    save_lora_adapter_checkpoint,
)


def _config(tmp_path: Path):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": str(tmp_path / "base"), "lora_rank": 2, "lora_adapter_path": None},
                "actor": {
                    "checkpoint": {
                        "save_contents": ["optimizer", "extra"],
                        "load_contents": ["optimizer", "extra"],
                    }
                },
            },
            "trainer": {
                "resume_mode": "auto",
                "resume_from_path": None,
                "default_local_dir": str(tmp_path / "checkpoints"),
            },
        }
    )


def _write_state_files(actor_dir: Path):
    actor_dir.mkdir(parents=True, exist_ok=True)
    (actor_dir / "fsdp_config.json").write_text(json.dumps({"world_size": 1}), encoding="utf-8")
    torch.save({"state": {}}, actor_dir / "optim_world_size_1_rank_0.pt")
    torch.save({"rng": {}}, actor_dir / "extra_state_world_size_1_rank_0.pt")


def _tiny_lora_model(seed: int = 123):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    base = LlamaForCausalLM(config)
    base_state = copy.deepcopy(base.state_dict())
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    return get_peft_model(base, peft_config), config, base_state


def _trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _assert_nested_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_state_equal(left_item, right_item)
    else:
        assert left == right


def test_adapter_plus_optimizer_round_trip_is_exact(tmp_path):
    model, config, base_state = _tiny_lora_model()
    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=1e-3)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    expected_adapter = {
        key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()
    }
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    actor_dir = tmp_path / "actor"
    save_lora_adapter_checkpoint(
        actor_path=actor_dir,
        lora_params=expected_adapter,
        peft_config=model.peft_config["default"],
        base_model_path=str(tmp_path / "base"),
        save_contents=["optimizer", "extra"],
        compact=False,
    )

    restored_base = LlamaForCausalLM(config)
    restored_base.load_state_dict(base_state)
    restored = PeftModel.from_pretrained(restored_base, actor_dir / "lora_adapter", is_trainable=True)
    restored_optimizer = torch.optim.AdamW(_trainable_parameters(restored), lr=1e-3)
    restored_optimizer.load_state_dict(expected_optimizer)

    restored_adapter = get_peft_model_state_dict(restored)
    assert expected_adapter.keys() == restored_adapter.keys()
    for key, value in expected_adapter.items():
        torch.testing.assert_close(value, restored_adapter[key], rtol=0, atol=0)
    _assert_nested_state_equal(expected_optimizer, restored_optimizer.state_dict())

    # The next update must also be identical, proving optimizer state is bound
    # to the same trainable adapter parameters after reconstruction.
    for current_model, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        next_loss = current_model(input_ids=input_ids, labels=input_ids).loss
        next_loss.backward()
        current_optimizer.step()
        current_optimizer.zero_grad()
    for key, value in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(value, get_peft_model_state_dict(restored)[key], rtol=0, atol=0)


def test_prepare_resume_injects_validated_adapter(tmp_path):
    config = _config(tmp_path)
    actor_dir = tmp_path / "checkpoints" / "global_step_50" / "actor"
    model, _, _ = _tiny_lora_model()
    _write_state_files(actor_dir)
    save_lora_adapter_checkpoint(
        actor_path=actor_dir,
        lora_params=get_peft_model_state_dict(model),
        peft_config=model.peft_config["default"],
        base_model_path=config.actor_rollout_ref.model.path,
        save_contents=["optimizer", "extra"],
        compact=True,
    )
    (actor_dir.parent / "data.pt").write_bytes(b"dataloader")
    (actor_dir.parent.parent / "latest_checkpointed_iteration.txt").write_text("50", encoding="utf-8")

    adapter_path = prepare_compact_lora_resume(config)

    assert adapter_path == str(actor_dir / "lora_adapter")
    assert config.actor_rollout_ref.model.lora_adapter_path == adapter_path


def test_prepare_resume_rejects_corrupt_adapter(tmp_path):
    config = _config(tmp_path)
    actor_dir = tmp_path / "checkpoints" / "global_step_50" / "actor"
    model, _, _ = _tiny_lora_model()
    _write_state_files(actor_dir)
    save_lora_adapter_checkpoint(
        actor_path=actor_dir,
        lora_params=get_peft_model_state_dict(model),
        peft_config=model.peft_config["default"],
        base_model_path=config.actor_rollout_ref.model.path,
        save_contents=["optimizer", "extra"],
        compact=True,
    )
    (actor_dir.parent / "data.pt").write_bytes(b"dataloader")
    (actor_dir.parent.parent / "latest_checkpointed_iteration.txt").write_text("50", encoding="utf-8")
    with (actor_dir / "lora_adapter" / "adapter_model.safetensors").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(CompactLoraCheckpointError, match="size mismatch"):
        prepare_compact_lora_resume(config)


def test_prepare_resume_rejects_inexact_content_config(tmp_path):
    config = _config(tmp_path)
    config.actor_rollout_ref.actor.checkpoint.save_contents = ["optimizer"]
    config.actor_rollout_ref.actor.checkpoint.load_contents = ["optimizer"]

    with pytest.raises(CompactLoraCheckpointError, match="exactly"):
        prepare_compact_lora_resume(config)


def test_metadata_is_committed_for_compact_checkpoint(tmp_path):
    model, _, _ = _tiny_lora_model()
    actor_dir = tmp_path / "actor"
    _write_state_files(actor_dir)
    save_lora_adapter_checkpoint(
        actor_path=actor_dir,
        lora_params=get_peft_model_state_dict(model),
        peft_config=model.peft_config["default"],
        base_model_path="base-model",
        save_contents=["optimizer", "extra"],
        compact=True,
    )
    metadata = json.loads((actor_dir / COMPACT_LORA_METADATA).read_text(encoding="utf-8"))
    assert metadata["format"] == "verl-compact-lora-v1"
    assert metadata["base_model_path"] == "base-model"
    assert metadata["adapter_size_bytes"] > 0
