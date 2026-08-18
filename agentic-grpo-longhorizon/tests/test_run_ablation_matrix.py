from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_DIR / "scripts" / "train" / "grpo" / "run_ablation_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_ablation_matrix", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


VALID_MATRIX = """
experiments:
  e000: {salt_enabled: false, progpo_enabled: false, lata_enabled: false}
  e001: {salt_enabled: false, progpo_enabled: false, lata_enabled: true}
  e010: {salt_enabled: false, progpo_enabled: true, lata_enabled: false}
  e011: {salt_enabled: false, progpo_enabled: true, lata_enabled: true}
  e100: {salt_enabled: true, progpo_enabled: false, lata_enabled: false}
  e101: {salt_enabled: true, progpo_enabled: false, lata_enabled: true}
  e110: {salt_enabled: true, progpo_enabled: true, lata_enabled: false}
  e111: {salt_enabled: true, progpo_enabled: true, lata_enabled: true}
"""


def make_paths(root: Path, matrix_text: str = VALID_MATRIX) -> runner.RunnerPaths:
    project = root / "project"
    matrix = project / "configs" / "ablation" / runner.MATRIX_FILENAME
    config = project / "configs" / "train" / "grpo" / f"{runner.TRAIN_CONFIG_NAME}.yaml"
    build_script = project / "scripts" / "train" / "grpo" / "build_grpo_parquet.py"
    matrix.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    build_script.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_text(matrix_text, encoding="utf-8")
    config.write_text(
        """data:
  tool_config_path: configs/tool_config/tools.yaml
actor_rollout_ref:
  model:
    path: experiments/default_model
  rollout:
    multi_turn:
      tool_config_path: configs/tool_config/tools.yaml
      interaction_config_path: configs/interaction_config/interaction.yaml
  ref:
    model:
      path: experiments/default_model
trainer:
  total_training_steps: 300
  total_epochs: 50
""",
        encoding="utf-8",
    )
    build_script.write_text(
        "raise SystemExit('must not run in dry-run')\n", encoding="utf-8"
    )
    tool_config = project / "configs" / "tool_config" / "tools.yaml"
    interaction_config = project / "configs" / "interaction_config" / "interaction.yaml"
    tool_config.parent.mkdir(parents=True)
    interaction_config.parent.mkdir(parents=True)
    tool_config.write_text("tools: []\n", encoding="utf-8")
    interaction_config.write_text(
        """interaction:
  - name: tau2
    config:
      user_model: ${oc.env:TAU2_USER_MODEL,fixture-user-model}
      user_provider: ${oc.env:TAU2_USER_PROVIDER,openai}
      user_base_url: ${oc.env:TAU2_USER_BASE_URL,http://localhost:8001/v1}
""",
        encoding="utf-8",
    )

    for _, base, relative in runner.IMPLEMENTATION_PATHS:
        base_path = project if base == "project" else root
        implementation = base_path / relative
        implementation.parent.mkdir(parents=True, exist_ok=True)
        implementation.write_text(
            f"# fixture implementation: {relative}\n", encoding="utf-8"
        )

    default_model = project / "experiments" / "default_model"
    default_model.mkdir(parents=True)
    (default_model / "config.json").write_text("{}\n", encoding="utf-8")
    (default_model / "model.safetensors").write_bytes(b"default-weight-v1")
    (default_model / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    paths = runner.RunnerPaths.for_project(project)
    airline_data = (
        paths.tau2_checkout / "src" / "tau2" / "domains" / "airline" / "tasks.json"
    )
    airline_data.parent.mkdir(parents=True)
    airline_data.write_text('{"tasks": ["v1"]}\n', encoding="utf-8")
    (paths.tau2_checkout / "src" / "tau2" / "__init__.py").write_text(
        '__version__ = "fixture"\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(paths.tau2_checkout)], check=True)
    subprocess.run(["git", "-C", str(paths.tau2_checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(paths.tau2_checkout),
            "-c",
            "user.name=Runner Test",
            "-c",
            "user.email=runner@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(paths.tau2_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return replace(paths, pinned_tau2_commit=head)


def make_identity(
    paths: runner.RunnerPaths,
    *,
    experiment_id: str = "e111",
    model_name: str = "model-a",
    create_data: bool = True,
    include_datasets: bool = True,
) -> tuple[dict, dict]:
    matrix = runner.load_and_validate_matrix(paths.matrix_path)
    if create_data:
        paths.train_data_path.parent.mkdir(parents=True, exist_ok=True)
        paths.train_data_path.write_bytes(b"train-data-v1")
        paths.val_data_path.write_bytes(b"val-data-v1")
    model_dir = paths.project_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    tokenizer_path = model_dir / "tokenizer.json"
    if not config_path.exists():
        config_path.write_text('{"architectures": ["TestModel"]}\n', encoding="utf-8")
    if not weight_path.exists():
        weight_path.write_bytes(b"fixture-weight-v1")
    if not tokenizer_path.exists():
        tokenizer_path.write_text('{"version": 1}\n', encoding="utf-8")
    identity = runner.build_run_identity(
        paths,
        experiment_id,
        matrix[experiment_id],
        actor_model=str(model_dir),
        reference_model=str(model_dir),
        include_datasets=include_datasets,
    )
    return identity, matrix[experiment_id]


def rebuild_identity(
    paths: runner.RunnerPaths,
    original: dict,
    spec: dict,
    *,
    include_datasets: bool = True,
) -> dict:
    return runner.build_run_identity(
        paths,
        original["experiment_id"],
        spec,
        actor_model=original["models"]["actor"]["effective_path"],
        reference_model=original["models"]["reference"]["effective_path"],
        include_datasets=include_datasets,
    )


def init_fake_tau2_checkout(paths: runner.RunnerPaths) -> Path:
    checkout = paths.tau2_checkout
    airline_data = checkout / "src" / "tau2" / "domains" / "airline" / "tasks.json"
    if not airline_data.is_file():
        raise AssertionError(f"fake tau2 checkout is incomplete: {checkout}")
    return airline_data


def fake_tau2_import_identity(checkout: Path) -> dict:
    expected_root = (checkout / "src" / "tau2").resolve(strict=False)
    return {
        "state": "present",
        "origin": str(expected_root / "__init__.py"),
        "search_locations": [str(expected_root)],
        "expected_root": str(expected_root),
        "matches_checkout": True,
        "error": None,
    }


class Tau2IdentityTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tau2_import_patch = mock.patch.object(
            runner,
            "_tau2_import_identity",
            side_effect=fake_tau2_import_identity,
        )
        self.tau2_import_patch.start()

    def tearDown(self):
        self.tau2_import_patch.stop()
        super().tearDown()


class MatrixValidationTests(unittest.TestCase):
    def test_runner_and_setup_pin_the_same_tau2_commit(self):
        setup_text = (PROJECT_DIR.parent / "setup.sh").read_text(encoding="utf-8")
        self.assertIn(
            f'TAU2_COMMIT="{runner.PINNED_TAU2_COMMIT}"',
            setup_text,
        )

    def test_repository_matrix_is_complete_factorial(self):
        matrix = runner.load_and_validate_matrix(
            PROJECT_DIR / "configs" / "ablation" / runner.MATRIX_FILENAME
        )
        combinations = {
            tuple(spec[field] for field in runner.COMPONENT_FIELDS)
            for spec in matrix.values()
        }
        self.assertEqual(len(matrix), 8)
        self.assertEqual(combinations, set(itertools.product((False, True), repeat=3)))

    def test_duplicate_combination_is_rejected(self):
        duplicate = VALID_MATRIX.replace(
            "e111: {salt_enabled: true, progpo_enabled: true, lata_enabled: true}",
            "e111: {salt_enabled: true, progpo_enabled: true, lata_enabled: false}",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir), duplicate)
            with self.assertRaises(runner.MatrixValidationError):
                runner.load_and_validate_matrix(paths.matrix_path)

    def test_non_boolean_switch_is_rejected(self):
        invalid = VALID_MATRIX.replace("salt_enabled: false", "salt_enabled: 0", 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir), invalid)
            with self.assertRaises(runner.MatrixValidationError):
                runner.load_and_validate_matrix(paths.matrix_path)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = {
            "a": {},
            "b": {},
            "c": {},
        }

    def test_all_preserves_matrix_order(self):
        self.assertEqual(
            runner.parse_experiment_selection(["all"], self.matrix),
            ["a", "b", "c"],
        )

    def test_space_and_comma_selections_are_supported(self):
        self.assertEqual(
            runner.parse_experiment_selection(["a,b", "c"], self.matrix),
            ["a", "b", "c"],
        )

    def test_unknown_and_duplicate_ids_are_rejected(self):
        with self.assertRaises(runner.RunnerError):
            runner.parse_experiment_selection(["missing"], self.matrix)
        with self.assertRaises(runner.RunnerError):
            runner.parse_experiment_selection(["a", "a"], self.matrix)


class CheckpointPlanningTests(unittest.TestCase):
    def test_auto_skips_a_completed_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            (checkpoint_dir / "global_step_300").mkdir(parents=True)
            (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(
                "300", encoding="utf-8"
            )
            plan = runner.plan_experiment(
                "full",
                {},
                checkpoint_dir=checkpoint_dir,
                target_steps=300,
                resume_mode="auto",
            )
            self.assertEqual(plan.action, "skip")
            self.assertEqual(plan.latest_step, 300)

    def test_auto_resumes_an_incomplete_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            (checkpoint_dir / "global_step_50").mkdir(parents=True)
            (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(
                "50", encoding="utf-8"
            )
            plan = runner.plan_experiment(
                "full",
                {},
                checkpoint_dir=checkpoint_dir,
                target_steps=300,
                resume_mode="auto",
            )
            self.assertEqual(plan.action, "run")
            self.assertEqual(plan.latest_step, 50)

    def test_disable_refuses_nonempty_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "old-file").write_text("x", encoding="utf-8")
            with self.assertRaises(runner.RunnerError):
                runner.plan_experiment(
                    "full",
                    {},
                    checkpoint_dir=checkpoint_dir,
                    target_steps=300,
                    resume_mode="disable",
                )

    def test_auto_rejects_a_tracker_with_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(
                "50", encoding="utf-8"
            )
            with self.assertRaises(runner.RunnerError):
                runner.latest_checkpoint_step(checkpoint_dir)


class ManifestIdentityTests(Tau2IdentityTestCase):
    def test_same_identity_resumes_and_step_target_can_extend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            identity, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, identity)
            checkpoint_dir = run_dir / "checkpoints"
            (checkpoint_dir / "global_step_50").mkdir(parents=True)
            (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(
                "50", encoding="utf-8"
            )
            plan = runner.plan_experiment(
                "e111",
                spec,
                checkpoint_dir=checkpoint_dir,
                target_steps=600,
                resume_mode="auto",
            )
            status = runner.validate_or_create_manifest(
                plan,
                identity,
                run_dir=run_dir,
                dry_run=False,
                static_only=False,
            )
            self.assertEqual(plan.action, "run")
            self.assertIn("validated full", status)
            self.assertNotIn("target", identity)
            self.assertEqual(identity["schema"], runner.MANIFEST_SCHEMA)
            self.assertEqual(identity["experiment_id"], "e111")
            self.assertEqual(set(identity["switches"]), set(runner.COMPONENT_FIELDS))
            self.assertEqual(len(identity["training_config"]["sha256"]), 64)
            self.assertEqual(
                len(identity["models"]["actor"]["local_content_sha256"]), 64
            )
            self.assertGreaterEqual(
                identity["models"]["actor"]["local_content"]["file_count"], 3
            )
            self.assertEqual(len(identity["datasets"]["train"]["sha256"]), 64)
            self.assertEqual(len(identity["datasets"]["val"]["sha256"]), 64)
            self.assertIn("interaction_config", identity["runtime"])
            self.assertIn("tau2", identity["runtime"])

    def test_same_local_actor_and_reference_are_hashed_only_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            model_dir = paths.project_dir / "experiments" / "default_model"
            original_hash = runner.sha256_file
            model_hashes: list[Path] = []

            def counting_hash(path: Path) -> str:
                if model_dir in path.parents:
                    model_hashes.append(path)
                return original_hash(path)

            with mock.patch.object(runner, "sha256_file", side_effect=counting_hash):
                identity = runner.build_models_identity(
                    str(model_dir),
                    str(model_dir),
                    project_dir=paths.project_dir,
                )
            self.assertEqual(len(model_hashes), 3)
            self.assertIs(identity["actor"], identity["reference"])

    def test_huggingface_identifier_explicitly_has_no_local_content_hash(self):
        identity = runner.build_models_identity(
            "Qwen/example-remote-model",
            "Qwen/example-remote-model",
            project_dir=PROJECT_DIR,
        )["actor"]
        self.assertEqual(identity["kind"], "huggingface_identifier")
        self.assertEqual(identity["identifier"], "Qwen/example-remote-model")
        self.assertIsNone(identity["local_content"])
        self.assertIsNone(identity["local_content_sha256"])

    def test_model_path_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            original, spec = make_identity(paths, model_name="model-a")
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            requested, _ = make_identity(paths, model_name="model-b")
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(runner.RunnerError, "effective_path"):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_common_config_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            original, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            with paths.train_config_path.open("a", encoding="utf-8") as handle:
                handle.write("# identity-changing edit\n")
            requested, _ = make_identity(paths)
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(runner.RunnerError, "training_config.sha256"):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_dataset_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            original, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            paths.train_data_path.write_bytes(b"different-train-data")
            requested, _ = make_identity(paths, create_data=False)
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(runner.RunnerError, "datasets.train.sha256"):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_same_path_weight_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            original, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            model_dir = Path(original["models"]["actor"]["effective_path"])
            (model_dir / "model.safetensors").write_bytes(b"fixture-weight-v2")
            requested = rebuild_identity(paths, original, spec)
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(runner.RunnerError, "local_content"):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_interaction_config_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            original, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            interaction = (
                paths.project_dir
                / "configs"
                / "interaction_config"
                / "interaction.yaml"
            )
            interaction.write_text(
                interaction.read_text(encoding="utf-8") + "# changed\n",
                encoding="utf-8",
            )
            requested = rebuild_identity(paths, original, spec)
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(
                runner.RunnerError, "runtime.interaction_config"
            ):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_user_simulator_environment_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            with mock.patch.dict(
                os.environ,
                {
                    "TAU2_USER_MODEL": "simulator-a",
                    "OPENAI_API_KEY": "must-not-enter-manifest",
                },
                clear=False,
            ):
                original, spec = make_identity(paths)
            self.assertNotIn("OPENAI_API_KEY", str(original))
            self.assertNotIn("must-not-enter-manifest", str(original))
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            with mock.patch.dict(
                os.environ, {"TAU2_USER_MODEL": "simulator-b"}, clear=False
            ):
                requested = rebuild_identity(paths, original, spec)
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(runner.RunnerError, "TAU2_USER_MODEL"):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_tau2_tracked_airline_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            airline_data = init_fake_tau2_checkout(paths)
            airline_data.write_text('{"tasks": ["v2"]}\n', encoding="utf-8")
            with self.assertRaisesRegex(runner.RunnerError, "tracked tau2"):
                make_identity(paths)

    def test_full_identity_rejects_missing_tau2_checkout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            missing_paths = replace(
                paths, tau2_checkout=paths.repository_dir / "missing-tau2"
            )
            with self.assertRaisesRegex(runner.RunnerError, "checkout is missing"):
                make_identity(missing_paths)

            static_identity, _ = make_identity(missing_paths, include_datasets=False)
            self.assertEqual(static_identity["runtime"]["tau2"]["state"], "missing")

    def test_full_identity_rejects_wrong_tau2_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            wrong_commit_paths = replace(paths, pinned_tau2_commit="0" * 40)
            with self.assertRaisesRegex(runner.RunnerError, "expected 0000"):
                make_identity(wrong_commit_paths)

    def test_full_identity_rejects_shadowed_tau2_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            expected_root = str(paths.tau2_checkout / "src" / "tau2")
            shadowed = {
                "state": "present",
                "origin": "/global/site-packages/tau2/__init__.py",
                "search_locations": ["/global/site-packages/tau2"],
                "expected_root": expected_root,
                "matches_checkout": False,
                "error": None,
            }
            with mock.patch.object(
                runner, "_tau2_import_identity", return_value=shadowed
            ), self.assertRaisesRegex(runner.RunnerError, "does not import tau2"):
                make_identity(paths)

    def test_key_implementation_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            original, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(run_dir / runner.MANIFEST_FILENAME, original)
            implementation = paths.project_dir / "src" / "envs" / "progpo_progress.py"
            implementation.write_text("# changed implementation\n", encoding="utf-8")
            requested = rebuild_identity(paths, original, spec)
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "run", "fresh"
            )
            with self.assertRaisesRegex(
                runner.RunnerError, "runtime.implementation.progpo_progress"
            ):
                runner.validate_or_create_manifest(
                    plan,
                    requested,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )

    def test_static_identity_can_be_checked_without_parquet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            full_identity, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            runner.atomic_write_manifest(
                run_dir / runner.MANIFEST_FILENAME, full_identity
            )
            paths.train_data_path.unlink()
            paths.val_data_path.unlink()
            static_identity, _ = make_identity(
                paths, create_data=False, include_datasets=False
            )
            plan = runner.ExperimentPlan(
                "e111", spec, run_dir / "checkpoints", "skip", "complete", 300
            )
            status = runner.validate_or_create_manifest(
                plan,
                static_identity,
                run_dir=run_dir,
                dry_run=True,
                static_only=True,
            )
            self.assertIn("validated static", status)

    def test_existing_checkpoint_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            identity, spec = make_identity(paths)
            run_dir = paths.ablation_root / "e111"
            checkpoint_dir = run_dir / "checkpoints"
            (checkpoint_dir / "global_step_50").mkdir(parents=True)
            plan = runner.ExperimentPlan(
                "e111", spec, checkpoint_dir, "run", "resume", 50
            )
            with self.assertRaisesRegex(runner.RunnerError, "no run_manifest.json"):
                runner.validate_or_create_manifest(
                    plan,
                    identity,
                    run_dir=run_dir,
                    dry_run=False,
                    static_only=False,
                )


class CommandTests(Tau2IdentityTestCase):
    def test_training_command_contains_only_requested_switches_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            spec = {
                "salt_enabled": True,
                "progpo_enabled": False,
                "lata_enabled": True,
            }
            checkpoint_dir = paths.ablation_root / "e101" / "checkpoints"
            plan = runner.ExperimentPlan(
                "e101", spec, checkpoint_dir, "run", "fresh run"
            )
            command = runner.build_training_command(
                paths,
                plan,
                python_executable="python-test",
                target_steps=12,
                total_epochs=5,
                resume_mode="auto",
                model_path="/models/policy",
            )
            self.assertIn("algorithm.ablation.salt_enabled=true", command)
            self.assertIn("algorithm.ablation.progpo_enabled=false", command)
            self.assertIn("algorithm.ablation.lata_enabled=true", command)
            self.assertIn("trainer.total_training_steps=12", command)
            self.assertIn("trainer.total_epochs=12", command)
            self.assertIn(f"trainer.default_local_dir={checkpoint_dir}", command)
            self.assertIn("actor_rollout_ref.model.path=/models/policy", command)
            self.assertIn("actor_rollout_ref.ref.model.path=/models/policy", command)
            self.assertIn("hydra.job.chdir=false", command)

    def test_dry_run_has_no_filesystem_side_effects_and_builds_data_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            output = io.StringIO()
            error = io.StringIO()
            return_code = runner.main(
                [
                    "--dry-run",
                    "--experiments",
                    "e000",
                    "e111",
                    "--steps",
                    "1",
                ],
                paths=paths,
                python_executable="python-test",
                output=output,
                error=error,
            )
            text = output.getvalue()
            self.assertEqual(return_code, 0, error.getvalue())
            self.assertEqual(text.count("build_grpo_parquet.py"), 1)
            self.assertEqual(text.count("-m verl.trainer.main_ppo"), 2)
            self.assertFalse(paths.train_data_path.exists())
            self.assertFalse(paths.val_data_path.exists())
            self.assertFalse(paths.ablation_root.exists())
            self.assertFalse(
                (paths.ablation_root / "e000" / runner.MANIFEST_FILENAME).exists()
            )

    def test_streaming_command_tees_output_and_returns_child_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            log_path = temp / "logs" / "child.log"
            output = io.StringIO()
            return_code = runner.run_streaming_command(
                [
                    sys.executable,
                    "-c",
                    "print('tee-marker'); raise SystemExit(7)",
                ],
                cwd=temp,
                log_path=log_path,
                output=output,
            )
            self.assertEqual(return_code, 7)
            self.assertIn("tee-marker", output.getvalue())
            self.assertIn("tee-marker", log_path.read_text(encoding="utf-8"))

    def test_shared_data_is_built_only_once_then_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            counter_path = paths.build_data_script.with_suffix(".count")
            paths.build_data_script.write_text(
                """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--train-task-split')
parser.add_argument('--val-task-split')
parser.add_argument('--output-train')
parser.add_argument('--output-val')
args = parser.parse_args()
counter = Path(__file__).with_suffix('.count')
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
for output in (args.output_train, args.output_val):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('parquet-placeholder')
""",
                encoding="utf-8",
            )
            output = io.StringIO()
            for _ in range(2):
                runner._prepare_shared_data(
                    paths,
                    python_executable=sys.executable,
                    dry_run=False,
                    skip_data=False,
                    rebuild_data=False,
                    output=output,
                )
            self.assertEqual(counter_path.read_text(encoding="utf-8"), "1")
            self.assertIn("Reusing existing", output.getvalue())

    def test_identity_drift_after_planning_is_rejected_before_subprocess(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            paths.train_data_path.parent.mkdir(parents=True)
            paths.train_data_path.write_text("data", encoding="utf-8")
            paths.val_data_path.write_text("data", encoding="utf-8")
            model_weight = (
                paths.project_dir
                / "experiments"
                / "default_model"
                / "model.safetensors"
            )
            original_builder = runner.build_training_command

            def build_then_drift(*args, **kwargs):
                command = original_builder(*args, **kwargs)
                model_weight.write_bytes(b"changed-after-plan")
                return command

            output = io.StringIO()
            error = io.StringIO()
            with mock.patch.object(
                runner, "build_training_command", side_effect=build_then_drift
            ), mock.patch.object(runner, "run_streaming_command") as stream:
                return_code = runner.main(
                    ["--experiments", "e111", "--steps", "1"],
                    paths=paths,
                    python_executable="python-test",
                    output=output,
                    error=error,
                )
            self.assertEqual(return_code, 1)
            stream.assert_not_called()
            self.assertIn("run identity mismatch", error.getvalue())
            self.assertIn("local_content", error.getvalue())

    def test_continue_on_error_runs_later_variants_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = make_paths(Path(temp_dir))
            paths.train_data_path.parent.mkdir(parents=True)
            paths.train_data_path.write_text("data", encoding="utf-8")
            paths.val_data_path.write_text("data", encoding="utf-8")
            output = io.StringIO()
            error = io.StringIO()
            with mock.patch.object(
                runner, "run_streaming_command", side_effect=(9, 0)
            ) as stream:
                return_code = runner.main(
                    [
                        "--experiments",
                        "e000",
                        "e111",
                        "--continue-on-error",
                        "--steps",
                        "1",
                    ],
                    paths=paths,
                    python_executable="python-test",
                    output=output,
                    error=error,
                )
            self.assertEqual(return_code, 1)
            self.assertEqual(stream.call_count, 2)
            self.assertIn("e000 exited with status 9", error.getvalue())
            self.assertIn("[done] e111", output.getvalue())
            for experiment_id in ("e000", "e111"):
                manifest = (
                    paths.ablation_root / experiment_id / runner.MANIFEST_FILENAME
                )
                self.assertTrue(manifest.is_file())
                self.assertEqual(
                    runner.load_run_manifest(manifest)["experiment_id"],
                    experiment_id,
                )


if __name__ == "__main__":
    unittest.main()
