import copy
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import LlamaConfig, LlamaForCausalLM

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.lora_checkpoint import save_lora_adapter_checkpoint
from verl.utils.fsdp_utils import apply_fsdp2, layered_summon_lora_params


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 checkpoint test requires CUDA")


def _build_base(config, state):
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.load_state_dict(state)
    return model.cuda()


def _wrap_fsdp2(model, mesh):
    apply_fsdp2(
        model,
        {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            ),
        },
        {},
    )
    return model


def _trainable(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        left = left.full_tensor() if hasattr(left, "full_tensor") else left
        right = right.full_tensor() if hasattr(right, "full_tensor") else right
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_compact_lora_fsdp2_save_and_resume_is_exact():
    if not dist.is_initialized():
        rendezvous = Path(tempfile.mkdtemp()) / "dist_init"
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    torch.cuda.set_device(0)
    mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("fsdp",))

    torch.manual_seed(29)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    initial = LlamaForCausalLM(config).to(torch.bfloat16).state_dict()
    initial = {key: value.cpu().clone() for key, value in initial.items()}
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = _wrap_fsdp2(get_peft_model(_build_base(config, initial), lora_config), mesh)
    optimizer = torch.optim.AdamW(_trainable(model), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    checkpoint_config = OmegaConf.create(
        {"save_contents": ["optimizer", "extra"], "load_contents": ["optimizer", "extra"]}
    )
    actor_dir = Path(tempfile.mkdtemp()) / "actor"
    manager = FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        checkpoint_config=checkpoint_config,
    )
    manager.save_checkpoint(str(actor_dir), global_step=1)
    expected_adapter = layered_summon_lora_params(model)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    save_lora_adapter_checkpoint(
        actor_path=actor_dir,
        lora_params=expected_adapter,
        peft_config=model.peft_config["default"],
        base_model_path="tiny-base",
        save_contents=["optimizer", "extra"],
        compact=True,
    )

    restored = PeftModel.from_pretrained(
        _build_base(config, initial), actor_dir / "lora_adapter", is_trainable=True
    )
    restored = _wrap_fsdp2(restored, mesh)
    restored_optimizer = torch.optim.AdamW(_trainable(restored), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.9)
    restored_manager = FSDPCheckpointManager(
        model=restored,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        checkpoint_config=checkpoint_config,
    )
    restored_manager.load_checkpoint(str(actor_dir))

    restored_adapter = layered_summon_lora_params(restored)
    assert expected_adapter.keys() == restored_adapter.keys()
    for key, value in expected_adapter.items():
        torch.testing.assert_close(value, restored_adapter[key], rtol=0, atol=0)
    _assert_nested_equal(expected_optimizer, restored_optimizer.state_dict())
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    dist.destroy_process_group()
