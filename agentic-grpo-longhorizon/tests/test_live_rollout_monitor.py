from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
MONITOR_PATH = (
    PROJECT_DIR / "scripts" / "train" / "grpo" / "log_to_tensorboard.py"
)
RUNNER_PATH = (
    PROJECT_DIR / "scripts" / "train" / "grpo" / "run_ablation_matrix.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


monitor = load_module("log_to_tensorboard", MONITOR_PATH)
runner = load_module("run_ablation_matrix_live_monitor", RUNNER_PATH)


class TrainingConfigTests(unittest.TestCase):
    def test_policy_is_non_thinking_and_tool_responses_allow_2048_characters(self):
        config_path = (
            PROJECT_DIR
            / "configs"
            / "train"
            / "grpo"
            / "agentic_ablation_tau2.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertIs(
            config["data"]["apply_chat_template_kwargs"]["enable_thinking"],
            False,
        )
        multi_turn = config["actor_rollout_ref"]["rollout"]["multi_turn"]
        self.assertEqual(multi_turn["format"], "qwen3_coder")
        self.assertEqual(multi_turn["max_tool_response_length"], 2048)

        defaults = runner.load_training_defaults(config_path)
        self.assertEqual(defaults.rollouts_per_step, 32)

    def test_monitor_command_uses_the_same_log_and_tensorboard_directory(self):
        paths = runner.RunnerPaths.for_project(PROJECT_DIR)
        log_path = PROJECT_DIR / "experiments" / "example" / "training.log"
        tensorboard_dir = PROJECT_DIR / "experiments" / "example" / "tensorboard"
        command = runner.build_rollout_monitor_command(
            paths,
            python_executable="python-test",
            log_path=log_path,
            tensorboard_dir=tensorboard_dir,
            rollouts_per_step=32,
        )

        self.assertEqual(command[0], "python-test")
        self.assertEqual(Path(command[1]), MONITOR_PATH)
        self.assertEqual(command[command.index("--log-file") + 1], str(log_path))
        self.assertEqual(
            command[command.index("--tensorboard-dir") + 1],
            str(tensorboard_dir),
        )
        self.assertEqual(
            command[command.index("--rollouts-per-step") + 1],
            "32",
        )


class RolloutMetricsTests(unittest.TestCase):
    def test_metrics_are_written_immediately_and_update_for_each_reward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = monitor.SummaryWriter(
                log_dir=temp_dir,
                max_queue=1,
                flush_secs=1,
                filename_suffix=".test_rollout_monitor",
            )
            metrics = monitor.RolloutMetrics(writer, rollouts_per_step=4)
            metrics.consume_line(
                "\x1b[36m(AgentLoopWorker) Evaluation result: reward=1.0\x1b[0m"
            )
            metrics.consume_line(
                "(AgentLoopWorker) Evaluation result: reward=0.0 "
                "[repeated 2x across cluster]"
            )
            writer.close()

            accumulator = EventAccumulator(temp_dir)
            accumulator.Reload()
            tags = set(accumulator.Tags()["scalars"])
            self.assertIn("rollout_live/completed", tags)
            self.assertIn("rollout_live/progress_percent", tags)
            self.assertIn("rollout_live/reward_running_mean", tags)
            self.assertIn("rollout_live/success_rate", tags)

            completed = accumulator.Scalars("rollout_live/completed")
            self.assertEqual([event.value for event in completed], [0.0, 1.0, 3.0])
            means = accumulator.Scalars("rollout_live/reward_running_mean")
            self.assertAlmostEqual(means[-1].value, 1.0 / 3.0, places=6)
            progress = accumulator.Scalars("rollout_live/progress_percent")
            self.assertEqual(progress[-1].value, 75.0)

    def test_finished_training_step_resets_live_counter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = monitor.SummaryWriter(log_dir=temp_dir, max_queue=1)
            metrics = monitor.RolloutMetrics(writer, rollouts_per_step=4)
            metrics.record_reward(1.0)
            metrics.consume_line("training/global_step:1")
            writer.close()

            accumulator = EventAccumulator(temp_dir)
            accumulator.Reload()
            completed = accumulator.Scalars("rollout_live/completed")
            self.assertEqual(completed[-1].value, 0.0)
            training_step = accumulator.Scalars("rollout_live/training_step")
            self.assertEqual(training_step[-1].value, 2.0)


if __name__ == "__main__":
    unittest.main()
