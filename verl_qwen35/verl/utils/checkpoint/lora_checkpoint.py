"""Compact, exactly-resumable LoRA checkpoint helpers.

The regular FSDP checkpoint stores the complete model state even when only a
LoRA adapter is trainable.  A compact checkpoint instead reconstructs the
frozen base model from ``actor_rollout_ref.model.path`` and stores the changing
adapter alongside the optimizer and auxiliary training state.

The adapter must be selected before the model is wrapped by FSDP.  Therefore
``prepare_compact_lora_resume`` runs in ``TaskRunner`` before workers are
created; the normal worker initialization then uses PEFT's supported
``PeftModel.from_pretrained`` path.  The regular FSDP checkpoint manager remains
responsible for optimizer, scheduler, RNG, and dataloader restoration.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf, open_dict
from safetensors.torch import save_file

from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


COMPACT_LORA_FORMAT = "verl-compact-lora-v1"
COMPACT_LORA_METADATA = "compact_lora_checkpoint.json"
ADAPTER_DIRNAME = "lora_adapter"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
_EXACT_RESUME_CONTENTS = frozenset({"optimizer", "extra"})


class CompactLoraCheckpointError(RuntimeError):
    """Raised when a compact checkpoint cannot guarantee exact resumption."""


def _contents(config: Any, field: str) -> list[str]:
    value = OmegaConf.select(config, f"actor_rollout_ref.actor.checkpoint.{field}")
    if value is None:
        return ["model", "optimizer", "extra"]
    return list(value)


def compact_lora_checkpoint_enabled(config: Any) -> bool:
    """Return whether actor checkpoints intentionally omit the full model."""

    return "model" not in _contents(config, "save_contents")


def validate_compact_lora_config(config: Any) -> None:
    """Reject compact configurations that cannot produce an exact resume."""

    if not compact_lora_checkpoint_enabled(config):
        return

    save_contents = set(_contents(config, "save_contents"))
    load_contents = set(_contents(config, "load_contents"))
    if save_contents != _EXACT_RESUME_CONTENTS or load_contents != _EXACT_RESUME_CONTENTS:
        raise CompactLoraCheckpointError(
            "Compact LoRA checkpoints require save_contents and load_contents "
            "to both be exactly [optimizer, extra]; the adapter is stored "
            "separately and the frozen base model is reloaded from model.path."
        )

    lora_rank = OmegaConf.select(config, "actor_rollout_ref.model.lora_rank", default=0) or 0
    lora_path = OmegaConf.select(config, "actor_rollout_ref.model.lora_adapter_path")
    if lora_rank <= 0 and not lora_path:
        raise CompactLoraCheckpointError(
            "A checkpoint without full model weights is only supported when LoRA is enabled."
        )


def _normalise_model_identity(value: str) -> str:
    path = Path(os.path.expanduser(value))
    if path.exists():
        return str(path.resolve())
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_peft_config(peft_config: Any) -> dict[str, Any]:
    if is_dataclass(peft_config):
        value = asdict(peft_config)
    elif isinstance(peft_config, Mapping):
        value = dict(peft_config)
    elif hasattr(peft_config, "to_dict"):
        value = peft_config.to_dict()
    else:
        raise TypeError(f"Unsupported PEFT config type: {type(peft_config)!r}")

    # PEFT configs contain Enum values on some releases.  JSON round-tripping
    # through ``default=str`` would emit ``TaskType.CAUSAL_LM`` rather than the
    # accepted value, so normalise Enum-like objects explicitly.
    for key, item in list(value.items()):
        if hasattr(item, "value"):
            value[key] = item.value
        elif isinstance(item, set):
            value[key] = sorted(item)
    if isinstance(value.get("target_modules"), set):
        value["target_modules"] = sorted(value["target_modules"])
    return value


def save_lora_adapter_checkpoint(
    *,
    actor_path: str | os.PathLike[str],
    lora_params: Mapping[str, Any],
    peft_config: Any,
    base_model_path: str,
    save_contents: list[str],
    compact: bool,
) -> Path:
    """Atomically write adapter artifacts and, for compact mode, a marker.

    The metadata marker is committed last.  Resume refuses a compact checkpoint
    without this marker or with a mismatched adapter hash.
    """

    actor_dir = Path(actor_path)
    adapter_dir = actor_dir / ADAPTER_DIRNAME
    adapter_dir.mkdir(parents=True, exist_ok=True)

    weights_path = adapter_dir / ADAPTER_WEIGHTS
    config_path = adapter_dir / ADAPTER_CONFIG
    weights_tmp = adapter_dir / f".{ADAPTER_WEIGHTS}.{os.getpid()}.tmp"
    config_tmp = adapter_dir / f".{ADAPTER_CONFIG}.{os.getpid()}.tmp"

    try:
        save_file(dict(lora_params), str(weights_tmp))
        with config_tmp.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable_peft_config(peft_config), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(weights_tmp, weights_path)
        os.replace(config_tmp, config_path)
    finally:
        weights_tmp.unlink(missing_ok=True)
        config_tmp.unlink(missing_ok=True)

    if compact:
        state_paths = sorted(actor_dir.glob("optim_world_size_*_rank_*.pt")) + sorted(
            actor_dir.glob("extra_state_world_size_*_rank_*.pt")
        )
        if not any(path.name.startswith("optim_") for path in state_paths) or not any(
            path.name.startswith("extra_state_") for path in state_paths
        ):
            raise CompactLoraCheckpointError(
                f"Optimizer/extra state must be saved before the compact adapter marker: {actor_dir}"
            )
        metadata = {
            "format": COMPACT_LORA_FORMAT,
            "base_model_path": base_model_path,
            "save_contents": list(save_contents),
            "adapter_weights": f"{ADAPTER_DIRNAME}/{ADAPTER_WEIGHTS}",
            "adapter_config": f"{ADAPTER_DIRNAME}/{ADAPTER_CONFIG}",
            "adapter_size_bytes": weights_path.stat().st_size,
            "adapter_sha256": _sha256(weights_path),
            "adapter_config_size_bytes": config_path.stat().st_size,
            "adapter_config_sha256": _sha256(config_path),
            "state_files": {
                path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in state_paths
            },
        }
        marker_path = actor_dir / COMPACT_LORA_METADATA
        marker_tmp = actor_dir / f".{COMPACT_LORA_METADATA}.{os.getpid()}.tmp"
        try:
            with marker_tmp.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(marker_tmp, marker_path)
        finally:
            marker_tmp.unlink(missing_ok=True)

    return adapter_dir


def validate_compact_lora_checkpoint(actor_path: str | os.PathLike[str], base_model_path: str) -> Path:
    """Validate a completed compact actor checkpoint and return its adapter."""

    actor_dir = Path(actor_path)
    marker_path = actor_dir / COMPACT_LORA_METADATA
    try:
        metadata = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompactLoraCheckpointError(f"Compact LoRA marker is missing: {marker_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CompactLoraCheckpointError(f"Invalid compact LoRA marker: {marker_path}") from exc

    if metadata.get("format") != COMPACT_LORA_FORMAT:
        raise CompactLoraCheckpointError(
            f"Unsupported compact LoRA checkpoint format in {marker_path}: {metadata.get('format')!r}"
        )
    if _normalise_model_identity(str(metadata.get("base_model_path", ""))) != _normalise_model_identity(
        base_model_path
    ):
        raise CompactLoraCheckpointError(
            "Compact checkpoint base model does not match the configured model.path: "
            f"{metadata.get('base_model_path')!r} != {base_model_path!r}"
        )
    if set(metadata.get("save_contents", [])) != _EXACT_RESUME_CONTENTS:
        raise CompactLoraCheckpointError(f"Compact checkpoint has unsafe save_contents in {marker_path}")

    adapter_dir = actor_dir / ADAPTER_DIRNAME
    weights_path = adapter_dir / ADAPTER_WEIGHTS
    config_path = adapter_dir / ADAPTER_CONFIG
    if not weights_path.is_file() or not config_path.is_file():
        raise CompactLoraCheckpointError(f"Compact checkpoint adapter is incomplete: {adapter_dir}")
    if weights_path.stat().st_size != metadata.get("adapter_size_bytes"):
        raise CompactLoraCheckpointError(f"Compact checkpoint adapter size mismatch: {weights_path}")
    if _sha256(weights_path) != metadata.get("adapter_sha256"):
        raise CompactLoraCheckpointError(f"Compact checkpoint adapter hash mismatch: {weights_path}")
    if config_path.stat().st_size != metadata.get("adapter_config_size_bytes"):
        raise CompactLoraCheckpointError(f"Compact checkpoint adapter config size mismatch: {config_path}")
    if _sha256(config_path) != metadata.get("adapter_config_sha256"):
        raise CompactLoraCheckpointError(f"Compact checkpoint adapter config hash mismatch: {config_path}")

    fsdp_config_path = actor_dir / "fsdp_config.json"
    try:
        fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
        world_size = int(fsdp_config["world_size"])
    except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CompactLoraCheckpointError(f"Invalid FSDP metadata: {fsdp_config_path}") from exc
    for rank in range(world_size):
        for prefix in ("optim", "extra_state"):
            state_path = actor_dir / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            if not state_path.is_file():
                raise CompactLoraCheckpointError(f"Compact checkpoint state is missing: {state_path}")
            state_metadata = metadata.get("state_files", {}).get(state_path.name)
            if not state_metadata:
                raise CompactLoraCheckpointError(f"Compact checkpoint state metadata is missing: {state_path}")
            if state_path.stat().st_size != state_metadata.get("size_bytes"):
                raise CompactLoraCheckpointError(f"Compact checkpoint state size mismatch: {state_path}")
            if _sha256(state_path) != state_metadata.get("sha256"):
                raise CompactLoraCheckpointError(f"Compact checkpoint state hash mismatch: {state_path}")

    return adapter_dir


def _resume_checkpoint_path(config: Any) -> str | None:
    resume_mode = OmegaConf.select(config, "trainer.resume_mode", default="auto")
    if resume_mode == "disable":
        return None
    if resume_mode == "resume_path":
        value = OmegaConf.select(config, "trainer.resume_from_path")
        if not isinstance(value, str) or "global_step_" not in value:
            raise CompactLoraCheckpointError("trainer.resume_from_path must name a global_step_* directory")
        return str(Path(value).expanduser().resolve())
    if resume_mode != "auto":
        raise CompactLoraCheckpointError(f"Unsupported trainer.resume_mode: {resume_mode!r}")

    checkpoint_root = OmegaConf.select(config, "trainer.default_local_dir")
    if checkpoint_root is None:
        return None
    return find_latest_ckpt_path(str(Path(checkpoint_root).expanduser().resolve()))


def prepare_compact_lora_resume(config: Any) -> str | None:
    """Inject the checkpoint adapter before actor/ref FSDP construction.

    Returns the adapter path when resuming, otherwise ``None``.  Full-model
    checkpoint configurations are left untouched.
    """

    validate_compact_lora_config(config)
    if not compact_lora_checkpoint_enabled(config):
        return None

    global_step_path = _resume_checkpoint_path(config)
    if global_step_path is None:
        return None

    global_step_dir = Path(global_step_path)
    data_path = global_step_dir / "data.pt"
    if not data_path.is_file():
        raise CompactLoraCheckpointError(f"Dataloader state is missing: {data_path}")

    base_model_path = str(OmegaConf.select(config, "actor_rollout_ref.model.path"))
    adapter_dir = validate_compact_lora_checkpoint(global_step_dir / "actor", base_model_path)
    with open_dict(config.actor_rollout_ref.model):
        config.actor_rollout_ref.model.lora_adapter_path = str(adapter_dir)
    print(f"Compact LoRA resume: loading adapter before FSDP initialization from {adapter_dir}")
    return str(adapter_dir)
