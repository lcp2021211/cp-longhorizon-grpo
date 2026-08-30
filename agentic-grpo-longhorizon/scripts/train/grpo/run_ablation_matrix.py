#!/usr/bin/env python3
"""Run the 2^3 SALT/ProGPO/LATA ablation matrix sequentially.

This runner deliberately uses one Hydra training config and changes only the
three algorithm switches plus experiment-specific output names.  It does not
use Hydra multirun because all variants normally share one policy GPU and one
external tau2 user-simulator service.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TextIO

import yaml


MATRIX_FILENAME = "salt_progpo_lata_matrix.yaml"
TRAIN_CONFIG_NAME = "agentic_ablation_tau2"
COMPONENT_FIELDS = ("salt_enabled", "progpo_enabled", "lata_enabled")
MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA = "agentic-ablation-run/v2"
MODEL_HASH_CACHE_SCHEMA = "agentic-model-content-cache/v1"
MODEL_HASH_CACHE_FILENAME = "model_content_sha256_v1.json"
REQUIRED_RUNTIME_MODULES = (
    ("qwen_vl_utils", "qwen-vl-utils==0.0.14"),
)
PINNED_TAU2_COMMIT = "c3398666e6559e3a063da3fc04b5acf7f941464e"
STATIC_IDENTITY_FIELDS = (
    "schema",
    "experiment_id",
    "switches",
    "models",
    "training_config",
    "runtime",
)
FULL_IDENTITY_FIELDS = STATIC_IDENTITY_FIELDS + ("datasets",)

EXCLUDED_TREE_DIRECTORIES = frozenset(
    {
        ".git",
        ".cache",
        ".pytest_cache",
        "__pycache__",
        "cache",
        "caches",
        "temp",
        "tmp",
    }
)
EXCLUDED_TREE_FILENAMES = frozenset({".DS_Store"})
EXCLUDED_TREE_SUFFIXES = (
    ".incomplete",
    ".lock",
    ".partial",
    ".swp",
    ".swo",
    ".temp",
    ".tmp",
)
IMPLEMENTATION_PATHS = (
    ("progpo_progress", "project", "src/envs/progpo_progress.py"),
    ("tau2_adapter", "project", "src/envs/tau2_adapter.py"),
    ("tau2_context", "project", "src/envs/tau_bench_context.py"),
    ("tau2_interaction", "project", "src/envs/tau_bench_interaction.py"),
    ("tau2_tools", "project", "src/envs/tau_bench_tools.py"),
    (
        "salt_trace",
        "repository",
        "verl_qwen35/verl/experimental/agent_loop/salt_trace.py",
    ),
    (
        "tool_agent_loop",
        "repository",
        "verl_qwen35/verl/experimental/agent_loop/tool_agent_loop.py",
    ),
    ("core_algos", "repository", "verl_qwen35/verl/trainer/ppo/core_algos.py"),
    ("ray_trainer", "repository", "verl_qwen35/verl/trainer/ppo/ray_trainer.py"),
    ("main_ppo", "repository", "verl_qwen35/verl/trainer/main_ppo.py"),
    ("fsdp_workers", "repository", "verl_qwen35/verl/workers/fsdp_workers.py"),
    (
        "qwen35_transformers",
        "repository",
        "verl_qwen35/verl/models/transformers/qwen3_5.py",
    ),
    (
        "compact_lora_checkpoint",
        "repository",
        "verl_qwen35/verl/utils/checkpoint/lora_checkpoint.py",
    ),
    (
        "checkpoint_manager",
        "repository",
        "verl_qwen35/verl/utils/checkpoint/checkpoint_manager.py",
    ),
    (
        "fsdp_checkpoint_manager",
        "repository",
        "verl_qwen35/verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    ),
)
TAU2_ENVIRONMENT_KEYS = (
    ("TAU2_USER_MODEL", "user_model"),
    ("TAU2_USER_PROVIDER", "user_provider"),
    ("TAU2_USER_BASE_URL", "user_base_url"),
)
OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+)(?:,(.*))?\}$")


class RunnerError(RuntimeError):
    """Raised for a safe, actionable runner failure."""


class MatrixValidationError(ValueError):
    """Raised when the ablation matrix is not the complete 2^3 design."""


@dataclass(frozen=True)
class RunnerPaths:
    project_dir: Path
    repository_dir: Path
    tau2_checkout: Path
    pinned_tau2_commit: str
    matrix_path: Path
    train_config_path: Path
    config_dir: Path
    build_data_script: Path
    train_data_path: Path
    val_data_path: Path
    ablation_root: Path
    model_hash_cache_path: Path

    @classmethod
    def for_project(cls, project_dir: Path) -> "RunnerPaths":
        project_dir = project_dir.resolve()
        repository_dir = project_dir.parent
        return cls(
            project_dir=project_dir,
            repository_dir=repository_dir,
            tau2_checkout=repository_dir / "tau2-bench",
            pinned_tau2_commit=PINNED_TAU2_COMMIT,
            matrix_path=project_dir / "configs" / "ablation" / MATRIX_FILENAME,
            train_config_path=(
                project_dir / "configs" / "train" / "grpo" / f"{TRAIN_CONFIG_NAME}.yaml"
            ),
            config_dir=project_dir / "configs" / "train" / "grpo",
            build_data_script=(
                project_dir / "scripts" / "train" / "grpo" / "build_grpo_parquet.py"
            ),
            train_data_path=project_dir / "experiments" / "tau2" / "train.parquet",
            val_data_path=project_dir / "experiments" / "tau2" / "val.parquet",
            ablation_root=project_dir / "experiments" / "ablations",
            model_hash_cache_path=(
                project_dir
                / "experiments"
                / ".identity_cache"
                / MODEL_HASH_CACHE_FILENAME
            ),
        )


@dataclass(frozen=True)
class TrainingDefaults:
    total_steps: int
    total_epochs: int


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    spec: Mapping[str, Any]
    checkpoint_dir: Path
    action: str
    reason: str
    latest_step: int | None = None


def default_paths() -> RunnerPaths:
    # .../scripts/train/grpo/run_ablation_matrix.py -> project directory
    return RunnerPaths.for_project(Path(__file__).resolve().parents[3])


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise RunnerError(f"{label} not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RunnerError(f"Invalid YAML in {label} {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RunnerError(f"{label} must contain a YAML mapping: {path}")
    return loaded


def load_and_validate_matrix(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate the complete, non-duplicated 2^3 boolean matrix."""
    raw = _load_yaml_mapping(path, "Ablation matrix")
    experiments = raw.get("experiments")
    if not isinstance(experiments, dict):
        raise MatrixValidationError("Matrix must define an 'experiments' mapping")
    if len(experiments) != 8:
        raise MatrixValidationError(
            f"Matrix must contain exactly 8 experiments, found {len(experiments)}"
        )

    validated: dict[str, dict[str, Any]] = {}
    combinations: set[tuple[bool, bool, bool]] = set()
    for experiment_id, spec in experiments.items():
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise MatrixValidationError(
                "Every experiment ID must be a non-empty string"
            )
        if not isinstance(spec, dict):
            raise MatrixValidationError(
                f"Experiment {experiment_id!r} must contain a mapping"
            )

        values: list[bool] = []
        for field in COMPONENT_FIELDS:
            value = spec.get(field)
            if type(value) is not bool:
                raise MatrixValidationError(
                    f"Experiment {experiment_id!r} field {field!r} must be boolean"
                )
            values.append(value)
        combination = tuple(values)
        if combination in combinations:
            raise MatrixValidationError(
                f"Duplicate component combination in experiment {experiment_id!r}: "
                f"{combination}"
            )
        combinations.add(combination)
        validated[experiment_id] = dict(spec)

    expected = set(itertools.product((False, True), repeat=3))
    if combinations != expected:
        missing = sorted(expected - combinations)
        extra = sorted(combinations - expected)
        raise MatrixValidationError(
            f"Matrix is not the complete 2^3 design; missing={missing}, extra={extra}"
        )
    return validated


def load_training_defaults(path: Path) -> TrainingDefaults:
    raw = _load_yaml_mapping(path, "Training config")
    trainer = raw.get("trainer")
    if not isinstance(trainer, dict):
        raise RunnerError(f"Training config has no trainer mapping: {path}")
    steps = trainer.get("total_training_steps")
    epochs = trainer.get("total_epochs")
    if type(steps) is not int or steps <= 0:
        raise RunnerError("trainer.total_training_steps must be a positive integer")
    if type(epochs) is not int or epochs <= 0:
        raise RunnerError("trainer.total_epochs must be a positive integer")
    return TrainingDefaults(total_steps=steps, total_epochs=epochs)


def sha256_file(path: Path) -> str:
    """Hash a file without loading model- or parquet-sized inputs into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunnerError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _hash_file_cached(path: Path, cache: MutableMapping[str, str] | None = None) -> str:
    cache_key = str(path.resolve(strict=False))
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    result = sha256_file(path)
    if cache is not None:
        cache[cache_key] = result
    return result


def _tree_path_is_excluded(relative_path: Path) -> bool:
    directory_names = {part.lower() for part in relative_path.parts[:-1]}
    if directory_names & EXCLUDED_TREE_DIRECTORIES:
        return True
    filename = relative_path.name
    lowered = filename.lower()
    return (
        filename in EXCLUDED_TREE_FILENAMES
        or lowered.endswith(EXCLUDED_TREE_SUFFIXES)
        or lowered.endswith("~")
    )


def build_content_manifest(
    root: Path,
    *,
    include: Callable[[Path], bool] | None = None,
    file_hash_cache: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Hash a deterministic sorted view of regular file contents below ``root``."""
    if not root.exists():
        return {
            "state": "missing",
            "tree_sha256": None,
            "file_count": 0,
            "files": [],
        }

    candidates: list[tuple[Path, Path]] = []
    if root.is_file():
        candidates.append((Path(root.name), root))
    elif root.is_dir():
        for directory, directory_names, filenames in os.walk(root, followlinks=False):
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name.lower() not in EXCLUDED_TREE_DIRECTORIES
            )
            directory_path = Path(directory)
            for filename in sorted(filenames):
                path = directory_path / filename
                relative = path.relative_to(root)
                if _tree_path_is_excluded(relative):
                    continue
                # is_file follows file symlinks, which is necessary for normal
                # Hugging Face snapshot layouts whose weights are symlinked.
                if path.is_file():
                    candidates.append((relative, path))
    else:
        raise RunnerError(f"Content identity root is not a file/directory: {root}")

    entries: list[dict[str, Any]] = []
    for relative, path in sorted(candidates, key=lambda item: item[0].as_posix()):
        if include is not None and not include(relative):
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise RunnerError(f"Cannot stat identity input {path}: {exc}") from exc
        entries.append(
            {
                "path": relative.as_posix(),
                "size": size,
                "sha256": _hash_file_cached(path, file_hash_cache),
            }
        )

    canonical = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "state": "present",
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def _model_content_metadata(root: Path) -> dict[str, Any]:
    """Collect a fast, content-free fingerprint for a local model tree."""
    candidates: list[tuple[Path, Path]] = []
    if root.is_file():
        candidates.append((Path(root.name), root))
    elif root.is_dir():
        for directory, directory_names, filenames in os.walk(root, followlinks=False):
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name.lower() not in EXCLUDED_TREE_DIRECTORIES
            )
            directory_path = Path(directory)
            for filename in sorted(filenames):
                path = directory_path / filename
                relative = path.relative_to(root)
                if _tree_path_is_excluded(relative) or not path.is_file():
                    continue
                candidates.append((relative, path))
    else:
        raise RunnerError(f"Content identity root is not a file/directory: {root}")

    files: list[dict[str, Any]] = []
    for relative, path in sorted(candidates, key=lambda item: item[0].as_posix()):
        try:
            stat_result = path.stat()
            link_target = os.readlink(path) if path.is_symlink() else None
        except OSError as exc:
            raise RunnerError(
                f"Cannot stat model identity input {path}: {exc}"
            ) from exc
        files.append(
            {
                "path": relative.as_posix(),
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "ctime_ns": stat_result.st_ctime_ns,
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
                "link_target": link_target,
            }
        )
    return {"file_count": len(files), "files": files}


def _empty_model_hash_cache() -> dict[str, Any]:
    return {"schema": MODEL_HASH_CACHE_SCHEMA, "entries": {}}


def _load_model_hash_cache(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError:
        return _empty_model_hash_cache()
    except (OSError, json.JSONDecodeError):
        # A cache is only an optimization. Never trust or block on a corrupt one.
        return _empty_model_hash_cache()
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema") != MODEL_HASH_CACHE_SCHEMA
        or not isinstance(loaded.get("entries"), dict)
    ):
        return _empty_model_hash_cache()
    return loaded


def _validated_cached_model_identity(
    cache: Mapping[str, Any], effective_path: str, metadata: Mapping[str, Any]
) -> dict[str, Any] | None:
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(effective_path)
    if not isinstance(entry, dict) or entry.get("metadata") != metadata:
        return None
    identity = entry.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("effective_path") != effective_path
    ):
        return None
    content = identity.get("local_content")
    if not isinstance(content, dict) or not isinstance(content.get("files"), list):
        return None

    metadata_files = metadata.get("files")
    if not isinstance(metadata_files, list):
        return None
    expected_path_sizes = [
        {"path": item.get("path"), "size": item.get("size")}
        for item in metadata_files
        if isinstance(item, dict)
    ]
    cached_path_sizes: list[dict[str, Any]] = []
    for item in content["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            return None
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        cached_path_sizes.append({"path": item.get("path"), "size": item.get("size")})
    if cached_path_sizes != expected_path_sizes:
        return None

    canonical = json.dumps(
        content["files"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    tree_sha256 = hashlib.sha256(canonical).hexdigest()
    if (
        content.get("state") != "present"
        or content.get("file_count") != len(content["files"])
        or content.get("tree_sha256") != tree_sha256
        or identity.get("local_content_sha256") != tree_sha256
    ):
        return None
    return identity


def _store_model_hash_cache_entry(
    cache_path: Path,
    effective_path: str,
    metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    cache = _load_model_hash_cache(cache_path)
    entries = cache["entries"]
    entries[effective_path] = {"metadata": metadata, "identity": identity}
    atomic_write_manifest(cache_path, cache)


def normalize_model_reference(value: str, project_dir: Path) -> str:
    """Return the path/identifier that is effectively used from project cwd."""
    expanded = Path(value).expanduser()
    candidate = expanded if expanded.is_absolute() else project_dir / expanded
    # Preserve non-local Hugging Face identifiers such as ``Qwen/model``.  A
    # local relative path is canonicalized once it exists or clearly uses a
    # project-local prefix.
    local_prefix = value.startswith((".", "experiments/", "models/"))
    if expanded.is_absolute() or candidate.exists() or local_prefix:
        return str(candidate.resolve(strict=False))
    return value


def _model_identity(
    reference: str,
    project_dir: Path,
    *,
    file_hash_cache: MutableMapping[str, str] | None = None,
    persistent_cache_path: Path | None = None,
    persist_cache: bool = True,
) -> dict[str, Any]:
    effective = normalize_model_reference(reference, project_dir)
    path = Path(effective)
    if not path.is_absolute():
        return {
            "kind": "huggingface_identifier",
            "identifier": effective,
            "effective_path": effective,
            "local_content": None,
            "local_content_sha256": None,
        }
    if not path.exists():
        raise RunnerError(
            f"Local model path does not exist, so its content cannot be identified: "
            f"{path}"
        )
    if not (path.is_dir() or path.is_file()):
        raise RunnerError(f"Local model path is not a file/directory: {path}")

    metadata_before = _model_content_metadata(path)
    if metadata_before["file_count"] == 0:
        raise RunnerError(
            f"Local model path contains no identity-bearing files: {path}"
        )
    cache_path = persistent_cache_path or (
        project_dir / "experiments" / ".identity_cache" / MODEL_HASH_CACHE_FILENAME
    )
    cached_identity = _validated_cached_model_identity(
        _load_model_hash_cache(cache_path), effective, metadata_before
    )
    if cached_identity is not None:
        return cached_identity

    content = build_content_manifest(path, file_hash_cache=file_hash_cache)
    metadata_after = _model_content_metadata(path)
    if metadata_after != metadata_before:
        raise RunnerError(
            f"Local model changed while its content was being hashed: {path}. "
            "Retry after the model files are stable."
        )
    identity = {
        "kind": "local_directory" if path.is_dir() else "local_file",
        "effective_path": effective,
        "local_content": content,
        "local_content_sha256": content["tree_sha256"],
    }
    if persist_cache:
        _store_model_hash_cache_entry(cache_path, effective, metadata_after, identity)
    return identity


def build_models_identity(
    actor_model: str,
    reference_model: str,
    *,
    project_dir: Path,
    model_identity_cache: MutableMapping[str, dict[str, Any]] | None = None,
    persistent_cache_path: Path | None = None,
    persist_cache: bool = True,
) -> dict[str, Any]:
    """Build actor/ref identity, hashing an identical local model only once."""
    identities = model_identity_cache if model_identity_cache is not None else {}
    file_hash_cache: dict[str, str] = {}

    def identify(reference: str) -> dict[str, Any]:
        effective = normalize_model_reference(reference, project_dir)
        if effective not in identities:
            identities[effective] = _model_identity(
                effective,
                project_dir,
                file_hash_cache=file_hash_cache,
                persistent_cache_path=persistent_cache_path,
                persist_cache=persist_cache,
            )
        return identities[effective]

    return {
        "actor": identify(actor_model),
        "reference": identify(reference_model),
    }


def resolve_model_references(
    training_config: Mapping[str, Any],
    *,
    project_dir: Path,
    model_override: str | None,
) -> tuple[str, str]:
    if model_override is not None:
        effective = normalize_model_reference(model_override, project_dir)
        return effective, effective
    try:
        actor = training_config["actor_rollout_ref"]["model"]["path"]
    except (KeyError, TypeError) as exc:
        raise RunnerError(
            "Training config must define actor_rollout_ref.model.path"
        ) from exc
    if not isinstance(actor, str):
        raise RunnerError("Actor/reference model path must be a string")
    effective = normalize_model_reference(actor, project_dir)
    # Modern veRL shares actor_rollout_ref.model between actor and reference.
    return effective, effective


def _nested_config_value(
    config: Mapping[str, Any], keys: Sequence[str], *, label: str
) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise RunnerError(f"Training config must define {'.'.join(keys)} ({label})")
        value = value[key]
    return value


def _project_config_path(value: Any, paths: RunnerPaths, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RunnerError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    path = candidate if candidate.is_absolute() else paths.project_dir / candidate
    path = path.resolve(strict=False)
    if not path.is_file():
        raise RunnerError(f"{label} not found: {path}")
    return path


def _file_identity(
    path: Path, *, file_hash_cache: MutableMapping[str, str] | None = None
) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RunnerError(f"Cannot stat identity input {path}: {exc}") from exc
    return {
        "effective_path": str(path.resolve(strict=False)),
        "size": size,
        "sha256": _hash_file_cached(path, file_hash_cache),
    }


def _resolve_interaction_value(raw_value: Any, environment_key: str) -> str:
    if environment_key in os.environ:
        return os.environ[environment_key]
    if not isinstance(raw_value, str):
        raise RunnerError(
            f"Interaction config value for {environment_key} must be a string"
        )
    match = OC_ENV_PATTERN.fullmatch(raw_value)
    if match is None:
        return raw_value
    configured_key, default = match.groups()
    if configured_key != environment_key:
        raise RunnerError(
            f"Interaction config maps {environment_key} through unexpected "
            f"environment key {configured_key}"
        )
    if default is None:
        raise RunnerError(
            f"{environment_key} is unset and the interaction config has no default"
        )
    return default


def _effective_user_simulator_environment(interaction_path: Path) -> dict[str, str]:
    interaction_config = _load_yaml_mapping(interaction_path, "Interaction config")
    interactions = interaction_config.get("interaction")
    if not isinstance(interactions, list) or not interactions:
        raise RunnerError(
            f"Interaction config must define a non-empty interaction list: "
            f"{interaction_path}"
        )
    selected: Mapping[str, Any] | None = None
    for interaction in interactions:
        if not isinstance(interaction, Mapping):
            continue
        candidate = interaction.get("config")
        if isinstance(candidate, Mapping) and all(
            field in candidate for _, field in TAU2_ENVIRONMENT_KEYS
        ):
            selected = candidate
            break
    if selected is None:
        fields = ", ".join(field for _, field in TAU2_ENVIRONMENT_KEYS)
        raise RunnerError(
            f"Interaction config {interaction_path} has no entry defining {fields}"
        )
    # Deliberately omit API credentials.  Only non-secret routing/model values
    # that can change rollout behavior belong in the manifest.
    return {
        environment_key: _resolve_interaction_value(selected[field], environment_key)
        for environment_key, field in TAU2_ENVIRONMENT_KEYS
    }


def _git_capture(
    checkout: Path, arguments: Sequence[str]
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RunnerError(
            f"Cannot inspect tau2 git checkout {checkout}: {exc}"
        ) from exc


def _tau2_import_identity(checkout: Path) -> dict[str, Any]:
    """Identify the tau2 package that this runner's Python would import."""
    expected_root = (checkout / "src" / "tau2").resolve(strict=False)
    try:
        spec = importlib.util.find_spec("tau2")
    except (ImportError, AttributeError, ValueError) as exc:
        return {
            "state": "lookup_error",
            "origin": None,
            "search_locations": [],
            "expected_root": str(expected_root),
            "matches_checkout": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if spec is None:
        return {
            "state": "missing",
            "origin": None,
            "search_locations": [],
            "expected_root": str(expected_root),
            "matches_checkout": False,
            "error": None,
        }

    locations = [
        str(Path(location).resolve(strict=False))
        for location in (spec.submodule_search_locations or [])
    ]
    candidates = [Path(location) for location in locations]
    origin: str | None = None
    if spec.origin not in (None, "built-in", "frozen"):
        origin_path = Path(spec.origin).resolve(strict=False)
        origin = str(origin_path)
        candidates.append(origin_path)

    matches_checkout = any(
        candidate == expected_root or expected_root in candidate.parents
        for candidate in candidates
    )
    return {
        "state": "present",
        "origin": origin,
        "search_locations": locations,
        "expected_root": str(expected_root),
        "matches_checkout": matches_checkout,
        "error": None,
    }


def build_tau2_identity(
    checkout: Path,
    *,
    expected_commit: str,
    required: bool,
    file_hash_cache: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    checkout = checkout.resolve(strict=False)
    if not checkout.exists():
        if required:
            raise RunnerError(
                f"Required pinned tau2 checkout is missing: {checkout}. "
                "Run setup.sh before training."
            )
        return {
            "checkout_path": str(checkout),
            "state": "missing",
            "expected_git_head": expected_commit,
            "git_head": None,
            "dirty_tracked": None,
            "tracked_diff_sha256": None,
            "tracked_diff_size": None,
            "airline_domain_content": {
                "state": "missing",
                "tree_sha256": None,
                "file_count": 0,
                "files": [],
            },
            "python_import": _tau2_import_identity(checkout),
        }
    if not checkout.is_dir():
        if required:
            raise RunnerError(f"tau2 checkout path is not a directory: {checkout}")
        return {
            "checkout_path": str(checkout),
            "state": "not_directory",
            "expected_git_head": expected_commit,
            "git_head": None,
            "dirty_tracked": None,
            "tracked_diff_sha256": None,
            "tracked_diff_size": None,
            "airline_domain_content": {
                "state": "missing",
                "tree_sha256": None,
                "file_count": 0,
                "files": [],
            },
            "python_import": _tau2_import_identity(checkout),
        }

    head_process = _git_capture(checkout, ["rev-parse", "--verify", "HEAD"])
    if head_process.returncode != 0:
        state = "not_git_checkout"
        git_head = None
        dirty_tracked: bool | None = None
        diff_digest = None
        diff_size = None
    else:
        state = "git_checkout"
        git_head = head_process.stdout.decode("utf-8", errors="replace").strip()
        diff_process = _git_capture(
            checkout,
            ["diff", "--no-ext-diff", "--binary", "HEAD", "--"],
        )
        if diff_process.returncode != 0:
            raise RunnerError(
                f"Cannot inspect tracked tau2 changes in {checkout}: "
                f"{diff_process.stderr.decode('utf-8', errors='replace').strip()}"
            )
        diff = diff_process.stdout
        dirty_tracked = bool(diff)
        diff_digest = hashlib.sha256(diff).hexdigest()
        diff_size = len(diff)

    airline_content = build_content_manifest(
        checkout,
        include=lambda relative: any(
            "airline" in component.lower() for component in relative.parts
        ),
        file_hash_cache=file_hash_cache,
    )
    import_identity = _tau2_import_identity(checkout)
    identity = {
        "checkout_path": str(checkout),
        "state": state,
        "expected_git_head": expected_commit,
        "git_head": git_head,
        "dirty_tracked": dirty_tracked,
        "tracked_diff_sha256": diff_digest,
        "tracked_diff_size": diff_size,
        "airline_domain_content": airline_content,
        "python_import": import_identity,
    }
    if required:
        problems: list[str] = []
        if state != "git_checkout":
            problems.append("path is not a Git checkout")
        if git_head != expected_commit:
            problems.append(
                f"HEAD is {git_head or 'unknown'}, expected {expected_commit}"
            )
        if dirty_tracked:
            problems.append("tracked tau2 files are modified")
        if airline_content["file_count"] <= 0:
            problems.append("airline domain content is empty")
        if not import_identity["matches_checkout"]:
            problems.append(
                "current Python does not import tau2 from the pinned checkout "
                f"(origin={import_identity['origin']!r})"
            )
        if problems:
            raise RunnerError(
                f"Pinned tau2 identity check failed for {checkout}: "
                + "; ".join(problems)
            )
    return identity


def build_runtime_identity(
    paths: RunnerPaths,
    training_config: Mapping[str, Any],
    *,
    require_tau2: bool,
) -> dict[str, Any]:
    file_hash_cache: dict[str, str] = {}
    data_tool_path = _project_config_path(
        _nested_config_value(
            training_config,
            ("data", "tool_config_path"),
            label="dataset tool config",
        ),
        paths,
        label="data.tool_config_path",
    )
    rollout_tool_path = _project_config_path(
        _nested_config_value(
            training_config,
            (
                "actor_rollout_ref",
                "rollout",
                "multi_turn",
                "tool_config_path",
            ),
            label="rollout tool config",
        ),
        paths,
        label="actor_rollout_ref.rollout.multi_turn.tool_config_path",
    )
    interaction_path = _project_config_path(
        _nested_config_value(
            training_config,
            (
                "actor_rollout_ref",
                "rollout",
                "multi_turn",
                "interaction_config_path",
            ),
            label="interaction config",
        ),
        paths,
        label="actor_rollout_ref.rollout.multi_turn.interaction_config_path",
    )

    implementation: dict[str, dict[str, Any]] = {}
    for label, base, relative in IMPLEMENTATION_PATHS:
        root = paths.project_dir if base == "project" else paths.repository_dir
        path = (root / relative).resolve(strict=False)
        if not path.is_file():
            raise RunnerError(f"Key implementation file not found: {path}")
        implementation[label] = _file_identity(path, file_hash_cache=file_hash_cache)

    return {
        "tool_configs": {
            "data": _file_identity(data_tool_path, file_hash_cache=file_hash_cache),
            "rollout": _file_identity(
                rollout_tool_path, file_hash_cache=file_hash_cache
            ),
        },
        "interaction_config": _file_identity(
            interaction_path, file_hash_cache=file_hash_cache
        ),
        "user_simulator_environment": _effective_user_simulator_environment(
            interaction_path
        ),
        "implementation": implementation,
        "tau2": build_tau2_identity(
            paths.tau2_checkout,
            expected_commit=paths.pinned_tau2_commit,
            required=require_tau2,
            file_hash_cache=file_hash_cache,
        ),
    }


def build_run_identity(
    paths: RunnerPaths,
    experiment_id: str,
    spec: Mapping[str, Any],
    *,
    actor_model: str,
    reference_model: str,
    include_datasets: bool,
    training_config: Mapping[str, Any] | None = None,
    model_identity_cache: MutableMapping[str, dict[str, Any]] | None = None,
    persist_model_cache: bool = True,
) -> dict[str, Any]:
    """Build the immutable identity used to guard checkpoint resumption.

    The target step is intentionally absent: a matching run may safely extend
    its step target with ``--steps``.
    """
    current_training_config = (
        dict(training_config)
        if training_config is not None
        else _load_yaml_mapping(paths.train_config_path, "Training config")
    )
    identity: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": experiment_id,
        "switches": {field: spec[field] for field in COMPONENT_FIELDS},
        "models": build_models_identity(
            actor_model,
            reference_model,
            project_dir=paths.project_dir,
            model_identity_cache=model_identity_cache,
            persistent_cache_path=paths.model_hash_cache_path,
            persist_cache=persist_model_cache,
        ),
        "training_config": {
            "effective_path": str(paths.train_config_path.resolve(strict=False)),
            "sha256": sha256_file(paths.train_config_path),
        },
        "runtime": build_runtime_identity(
            paths,
            current_training_config,
            require_tau2=include_datasets,
        ),
    }
    if include_datasets:
        missing = [
            str(path)
            for path in (paths.train_data_path, paths.val_data_path)
            if not path.is_file()
        ]
        if missing:
            raise RunnerError(
                "Cannot build full run identity; parquet file(s) missing: "
                + ", ".join(missing)
            )
        identity["datasets"] = {
            "train": {"sha256": sha256_file(paths.train_data_path)},
            "val": {"sha256": sha256_file(paths.val_data_path)},
        }
    return identity


def build_current_plan_identity(
    paths: RunnerPaths,
    plan: ExperimentPlan,
    *,
    model_override: str | None,
    include_datasets: bool,
    model_identity_cache: MutableMapping[str, dict[str, Any]] | None = None,
    persist_model_cache: bool = True,
) -> tuple[dict[str, Any], str, str]:
    """Reload config and construct identity from the state that exists now."""
    training_config = _load_yaml_mapping(paths.train_config_path, "Training config")
    actor_model, reference_model = resolve_model_references(
        training_config,
        project_dir=paths.project_dir,
        model_override=model_override,
    )
    return (
        build_run_identity(
            paths,
            plan.experiment_id,
            plan.spec,
            actor_model=actor_model,
            reference_model=reference_model,
            include_datasets=include_datasets,
            training_config=training_config,
            model_identity_cache=model_identity_cache,
            persist_model_cache=persist_model_cache,
        ),
        actor_model,
        reference_model,
    )


def _identity_view(identity: Mapping[str, Any], *, static_only: bool) -> dict[str, Any]:
    fields = STATIC_IDENTITY_FIELDS if static_only else FULL_IDENTITY_FIELDS
    return {field: identity.get(field) for field in fields}


def _identity_differences(
    existing: Any, requested: Any, *, prefix: str = ""
) -> list[str]:
    if isinstance(existing, dict) and isinstance(requested, dict):
        differences: list[str] = []
        for key in sorted(set(existing) | set(requested)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in existing:
                differences.append(f"{path}: missing in manifest")
            elif key not in requested:
                differences.append(f"{path}: unexpected in manifest")
            else:
                differences.extend(
                    _identity_differences(existing[key], requested[key], prefix=path)
                )
        return differences
    if existing != requested:
        return [f"{prefix}: manifest={existing!r}, requested={requested!r}"]
    return []


def load_run_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise RunnerError(f"Run manifest not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Invalid run manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RunnerError(f"Run manifest must contain a JSON object: {path}")
    return manifest


def atomic_write_manifest(path: Path, identity: Mapping[str, Any]) -> None:
    """Durably replace a manifest without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _run_dir_has_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        raise RunnerError(f"Experiment output path is not a directory: {run_dir}")
    for entry in run_dir.iterdir():
        if entry.name == MANIFEST_FILENAME:
            continue
        if (
            entry.name == "checkpoints"
            and entry.is_dir()
            and not _directory_has_entries(entry)
        ):
            continue
        return True
    return False


def validate_or_create_manifest(
    plan: ExperimentPlan,
    identity: Mapping[str, Any],
    *,
    run_dir: Path,
    dry_run: bool,
    static_only: bool,
    require_existing: bool = False,
) -> str:
    """Validate an existing identity or atomically create it before training."""
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        existing = load_run_manifest(manifest_path)
        differences = _identity_differences(
            _identity_view(existing, static_only=static_only),
            _identity_view(identity, static_only=static_only),
        )
        if differences:
            detail = "; ".join(differences[:8])
            if len(differences) > 8:
                detail += f"; ... ({len(differences) - 8} more)"
            raise RunnerError(
                f"{plan.experiment_id}: run identity mismatch in {manifest_path}: "
                f"{detail}. Move the old experiment directory aside before starting "
                "a different run."
            )
        scope = "static" if static_only else "full"
        return f"validated {scope} run identity"

    if require_existing:
        raise RunnerError(
            f"{plan.experiment_id}: {MANIFEST_FILENAME} disappeared before launch: "
            f"{manifest_path}. Refusing to start an identity-unverified process."
        )
    if plan.latest_step is not None or _run_dir_has_artifacts(run_dir):
        raise RunnerError(
            f"{plan.experiment_id}: existing run/checkpoint has no {MANIFEST_FILENAME}: "
            f"{run_dir}. Refusing an unverifiable resume; move it aside or restore "
            "the original manifest."
        )
    if dry_run:
        return "would atomically create run manifest after full identity validation"
    if static_only:
        raise RunnerError(
            "Cannot create a run manifest before dataset hashes are available"
        )
    atomic_write_manifest(manifest_path, identity)
    return f"created {manifest_path}"


def parse_experiment_selection(
    requested: Sequence[str], matrix: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Resolve ``all`` or a space/comma-separated subset in matrix order."""
    tokens: list[str] = []
    for item in requested:
        tokens.extend(token.strip() for token in item.split(",") if token.strip())
    if not tokens:
        raise RunnerError("--experiments requires at least one experiment ID or 'all'")
    if "all" in tokens:
        if len(tokens) != 1:
            raise RunnerError("'all' cannot be combined with explicit experiment IDs")
        return list(matrix)

    unknown = [experiment_id for experiment_id in tokens if experiment_id not in matrix]
    if unknown:
        raise RunnerError(
            f"Unknown experiment(s): {', '.join(unknown)}. "
            f"Use --list to see valid IDs."
        )
    if len(set(tokens)) != len(tokens):
        raise RunnerError("--experiments contains duplicate experiment IDs")
    return tokens


def _directory_has_entries(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        raise RunnerError(f"Checkpoint path exists but is not a directory: {path}")
    try:
        return next(path.iterdir(), None) is not None
    except OSError as exc:
        raise RunnerError(f"Cannot inspect checkpoint directory {path}: {exc}") from exc


def latest_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Read veRL's atomic checkpoint tracker and validate its target folder."""
    if not checkpoint_dir.exists():
        return None
    if not checkpoint_dir.is_dir():
        raise RunnerError(
            f"Checkpoint path exists but is not a directory: {checkpoint_dir}"
        )
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        if _directory_has_entries(checkpoint_dir):
            raise RunnerError(
                f"Checkpoint directory is non-empty but has no tracker file: "
                f"{checkpoint_dir}. Move it aside or repair the checkpoint before resuming."
            )
        return None
    try:
        value = tracker.read_text(encoding="utf-8").strip()
        step = int(value)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"Invalid checkpoint tracker {tracker}") from exc
    if step < 0:
        raise RunnerError(f"Checkpoint step cannot be negative in {tracker}")
    step_dir = checkpoint_dir / f"global_step_{step}"
    if not step_dir.is_dir():
        raise RunnerError(
            f"Checkpoint tracker points to a missing directory: {step_dir}"
        )
    return step


def plan_experiment(
    experiment_id: str,
    spec: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    target_steps: int,
    resume_mode: str,
) -> ExperimentPlan:
    if resume_mode == "disable":
        if _directory_has_entries(checkpoint_dir):
            raise RunnerError(
                f"{experiment_id}: --resume-mode disable refuses the non-empty "
                f"checkpoint directory {checkpoint_dir}. Use auto, or move the old "
                "directory aside before starting a fresh run."
            )
        return ExperimentPlan(
            experiment_id,
            spec,
            checkpoint_dir,
            "run",
            "fresh run",
        )
    if resume_mode != "auto":
        raise RunnerError(f"Unsupported resume mode: {resume_mode}")

    latest_step = latest_checkpoint_step(checkpoint_dir)
    if latest_step is not None and latest_step >= target_steps:
        return ExperimentPlan(
            experiment_id,
            spec,
            checkpoint_dir,
            "skip",
            f"checkpoint step {latest_step} already reached target {target_steps}",
            latest_step,
        )
    reason = (
        "start from scratch"
        if latest_step is None
        else f"resume from step {latest_step}"
    )
    return ExperimentPlan(
        experiment_id,
        spec,
        checkpoint_dir,
        "run",
        reason,
        latest_step,
    )


def build_data_command(paths: RunnerPaths, python_executable: str) -> list[str]:
    return [
        python_executable,
        str(paths.build_data_script),
        "--train-task-split",
        "train",
        "--val-task-split",
        "test",
        "--output-train",
        str(paths.train_data_path),
        "--output-val",
        str(paths.val_data_path),
    ]


def build_training_command(
    paths: RunnerPaths,
    plan: ExperimentPlan,
    *,
    python_executable: str,
    target_steps: int,
    total_epochs: int,
    resume_mode: str,
    model_path: str | None,
) -> list[str]:
    bool_value = lambda value: "true" if value else "false"
    run_dir = paths.ablation_root / plan.experiment_id
    # An explicit step cap always wins.  Keeping at least one epoch per desired
    # step prevents a larger --steps value from being cut short by total_epochs.
    effective_epochs = max(total_epochs, target_steps)
    command = [
        python_executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={paths.config_dir}",
        f"--config-name={TRAIN_CONFIG_NAME}",
        ("algorithm.ablation.salt_enabled=" + bool_value(plan.spec["salt_enabled"])),
        (
            "algorithm.ablation.progpo_enabled="
            + bool_value(plan.spec["progpo_enabled"])
        ),
        ("algorithm.ablation.lata_enabled=" + bool_value(plan.spec["lata_enabled"])),
        f"trainer.total_training_steps={target_steps}",
        f"trainer.total_epochs={effective_epochs}",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.default_local_dir={plan.checkpoint_dir}",
        f"trainer.experiment_name={plan.experiment_id}",
        f"hydra.run.dir={run_dir / 'hydra'}/${{now:%Y%m%d_%H%M%S}}",
        "hydra.job.chdir=false",
    ]
    if model_path is not None:
        command.append(f"actor_rollout_ref.model.path={model_path}")
    return command


def format_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_streaming_command(
    command: Sequence[str], *, cwd: Path, log_path: Path, output: TextIO = sys.stdout
) -> int:
    """Tee combined stdout/stderr to console and a file, preserving exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        header = f"\n$ {format_command(command)}\n"
        output.write(header)
        output.flush()
        log_handle.write(header)
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            message = f"Failed to start command: {exc}\n"
            output.write(message)
            output.flush()
            log_handle.write(message)
            return 127

        assert process.stdout is not None
        try:
            with process.stdout:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    log_handle.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def run_simple_command(command: Sequence[str], *, cwd: Path) -> int:
    try:
        return subprocess.run(list(command), cwd=cwd, check=False).returncode
    except OSError as exc:
        raise RunnerError(
            f"Failed to start command: {format_command(command)}: {exc}"
        ) from exc


def validate_runtime_dependencies() -> None:
    """Fail before model loading when an agent-loop runtime module is absent."""
    missing = [
        requirement
        for module, requirement in REQUIRED_RUNTIME_MODULES
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        requirements = " ".join(missing)
        raise RunnerError(
            "Missing training runtime dependencies: "
            f"{requirements}. Install them in the active environment with "
            f"`python -m pip install {requirements}`."
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SALT/ProGPO/LATA 2^3 ablation matrix sequentially"
    )
    parser.add_argument(
        "--list", action="store_true", help="list matrix experiments and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact plan and commands without creating or changing files",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        metavar="ID",
        help="space/comma-separated experiment IDs, or all (default: all)",
    )
    parser.add_argument(
        "--model-path",
        help="override both the trainable policy and reference-model paths",
    )
    parser.add_argument(
        "--steps",
        type=_positive_int,
        help="training-step target (default: value in the common Hydra config)",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("auto", "disable"),
        default="auto",
        help="auto resumes/skips completed runs; disable requires an empty checkpoint dir",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--skip-data",
        action="store_true",
        help="never build parquet files; fail if either file is missing",
    )
    data_group.add_argument(
        "--rebuild-data",
        action="store_true",
        help="rebuild the shared tau2 parquet files once before selected runs",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with later variants after a planning or training failure",
    )
    return parser


def print_matrix(matrix: Mapping[str, Mapping[str, Any]], output: TextIO) -> None:
    output.write("ID                         S  P  L  Description\n")
    for experiment_id, spec in matrix.items():
        flags = ["1" if spec[field] else "0" for field in COMPONENT_FIELDS]
        description = str(spec.get("description", ""))
        output.write(
            f"{experiment_id:<26} {flags[0]}  {flags[1]}  {flags[2]}  {description}\n"
        )


def _prepare_shared_data(
    paths: RunnerPaths,
    *,
    python_executable: str,
    dry_run: bool,
    skip_data: bool,
    rebuild_data: bool,
    output: TextIO,
) -> bool:
    """Prepare shared parquet files and report whether hashes are available."""
    files_exist = paths.train_data_path.is_file() and paths.val_data_path.is_file()
    if skip_data:
        if not files_exist:
            if dry_run:
                output.write(
                    "[dry-run][data] --skip-data would fail because parquet files "
                    "are missing; full identity validation cannot run.\n"
                )
                return False
            raise RunnerError(
                "--skip-data was requested, but shared parquet files are missing: "
                f"{paths.train_data_path}, {paths.val_data_path}"
            )
        output.write("[data] Using existing parquet files (--skip-data).\n")
        return True
    if files_exist and not rebuild_data:
        output.write("[data] Reusing existing tau2 parquet files.\n")
        return True

    command = build_data_command(paths, python_executable)
    if dry_run:
        output.write(f"[dry-run][data] {format_command(command)}\n")
        output.write(
            "[dry-run][identity] Dataset hashes will be validated after the "
            "planned build; no manifest will be written now.\n"
        )
        return False
    output.write("[data] Building the shared tau2 parquet files once.\n")
    output.flush()
    return_code = run_simple_command(command, cwd=paths.project_dir)
    if return_code != 0:
        raise RunnerError(f"Dataset builder exited with status {return_code}")
    if not paths.train_data_path.is_file() or not paths.val_data_path.is_file():
        raise RunnerError(
            "Dataset builder succeeded but did not create both parquet files"
        )
    return True


def execute(
    args: argparse.Namespace,
    *,
    paths: RunnerPaths,
    python_executable: str,
    output: TextIO,
    error: TextIO,
) -> int:
    matrix = load_and_validate_matrix(paths.matrix_path)
    if args.list:
        print_matrix(matrix, output)
        return 0

    validate_runtime_dependencies()

    defaults = load_training_defaults(paths.train_config_path)
    target_steps = args.steps or defaults.total_steps
    selected = parse_experiment_selection(args.experiments, matrix)

    plans: list[ExperimentPlan] = []
    failures = 0
    for experiment_id in selected:
        checkpoint_dir = paths.ablation_root / experiment_id / "checkpoints"
        try:
            plan = plan_experiment(
                experiment_id,
                matrix[experiment_id],
                checkpoint_dir=checkpoint_dir,
                target_steps=target_steps,
                resume_mode=args.resume_mode,
            )
        except RunnerError as exc:
            failures += 1
            error.write(f"[error] {exc}\n")
            if not args.continue_on_error:
                return 1
            continue
        plans.append(plan)

    if not plans:
        return 1 if failures else 0

    try:
        datasets_available = _prepare_shared_data(
            paths,
            python_executable=python_executable,
            dry_run=args.dry_run,
            skip_data=args.skip_data,
            rebuild_data=args.rebuild_data,
            output=output,
        )
    except RunnerError as exc:
        error.write(f"[error] {exc}\n")
        return 1

    validated_plans: list[ExperimentPlan] = []
    planned_actor_models: dict[str, str] = {}
    shared_model_identity_cache: dict[str, dict[str, Any]] = {}
    for plan in plans:
        try:
            identity, actor_model, _ = build_current_plan_identity(
                paths,
                plan,
                model_override=args.model_path,
                include_datasets=datasets_available,
                model_identity_cache=shared_model_identity_cache,
                persist_model_cache=not args.dry_run,
            )
            identity_status = validate_or_create_manifest(
                plan,
                identity,
                run_dir=paths.ablation_root / plan.experiment_id,
                dry_run=args.dry_run,
                static_only=not datasets_available,
            )
        except RunnerError as exc:
            failures += 1
            error.write(f"[error] {exc}\n")
            if not args.continue_on_error:
                return 1
            continue
        validated_plans.append(plan)
        planned_actor_models[plan.experiment_id] = actor_model
        prefix = "dry-run" if args.dry_run else "plan"
        output.write(
            f"[{prefix}] {plan.experiment_id}: {plan.action} ({plan.reason}); "
            f"identity={identity_status}\n"
        )
        if not datasets_available:
            output.write(
                f"[{prefix}][identity] {plan.experiment_id}: static identity status "
                f"recorded above; train/val hashes will be checked after data "
                "preparation.\n"
            )

    runnable = [plan for plan in validated_plans if plan.action == "run"]
    for plan in runnable:
        command = build_training_command(
            paths,
            plan,
            python_executable=python_executable,
            target_steps=target_steps,
            total_epochs=defaults.total_epochs,
            resume_mode=args.resume_mode,
            model_path=(
                planned_actor_models[plan.experiment_id]
                if args.model_path is not None
                else None
            ),
        )
        if args.dry_run:
            output.write(f"[dry-run][train] {format_command(command)}\n")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = (
            paths.ablation_root
            / plan.experiment_id
            / "logs"
            / f"training_{timestamp}.log"
        )
        output.write(f"[train] {plan.experiment_id}: {plan.reason}; log={log_path}\n")
        output.flush()
        try:
            # Rebuild identity at the last possible point before Popen. Local
            # models use the persistent content hash only when their complete
            # metadata fingerprint is unchanged, so this is a fast pre-launch
            # check without rereading weight bytes. Other, small identity inputs
            # are re-hashed to close the plan/launch drift window.
            launch_identity, _, _ = build_current_plan_identity(
                paths,
                plan,
                model_override=args.model_path,
                include_datasets=True,
                model_identity_cache=None,
            )
            launch_status = validate_or_create_manifest(
                plan,
                launch_identity,
                run_dir=paths.ablation_root / plan.experiment_id,
                dry_run=False,
                static_only=False,
                require_existing=True,
            )
            output.write(
                f"[identity] {plan.experiment_id}: pre-launch {launch_status}\n"
            )
            output.flush()
        except RunnerError as exc:
            failures += 1
            error.write(f"[error] {exc}\n")
            if not args.continue_on_error:
                return 1
            continue
        return_code = run_streaming_command(
            command, cwd=paths.project_dir, log_path=log_path, output=output
        )
        if return_code != 0:
            failures += 1
            error.write(
                f"[error] {plan.experiment_id} exited with status {return_code}; "
                f"see {log_path}\n"
            )
            if not args.continue_on_error:
                return return_code
        else:
            output.write(f"[done] {plan.experiment_id}\n")

    return 1 if failures else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    paths: RunnerPaths | None = None,
    python_executable: str | None = None,
    output: TextIO = sys.stdout,
    error: TextIO = sys.stderr,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return execute(
            args,
            paths=paths or default_paths(),
            python_executable=python_executable or sys.executable,
            output=output,
            error=error,
        )
    except (RunnerError, MatrixValidationError) as exc:
        error.write(f"[error] {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
