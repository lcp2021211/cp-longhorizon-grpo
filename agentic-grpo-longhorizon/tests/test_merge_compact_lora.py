import importlib.util
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test" / "merge_fsdp_to_hf.py"


def _load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_fsdp_to_hf", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compact_lora_export_matches_adapter_model(tmp_path):
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    base_dir = tmp_path / "base"
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(base_dir, safe_serialization=True)

    trainable = get_peft_model(
        AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    with torch.no_grad():
        for name, parameter in trainable.named_parameters():
            if "lora_" in name:
                parameter.uniform_(-0.1, 0.1)

    actor_dir = tmp_path / "actor"
    adapter_dir = actor_dir / "lora_adapter"
    trainable.save_pretrained(adapter_dir)
    (actor_dir / "compact_lora_checkpoint.json").write_text(
        json.dumps({"format": "verl-compact-lora-v1", "base_model_path": str(base_dir)}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "merged"
    _load_merge_module().merge_fsdp_checkpoint(str(actor_dir), str(output_dir))

    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16),
        adapter_dir,
    )
    actual = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    with torch.no_grad():
        expected_logits = expected(input_ids=input_ids).logits
        actual_logits = actual(input_ids=input_ids).logits
    torch.testing.assert_close(expected_logits, actual_logits, rtol=0.02, atol=0.002)
