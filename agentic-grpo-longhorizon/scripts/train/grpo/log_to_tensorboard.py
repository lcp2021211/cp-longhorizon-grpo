#!/usr/bin/env python3
"""Tail one training log and expose live rollout progress in TensorBoard.

The monitor is intentionally read-only with respect to training: it only opens
the runner's text log for reading and writes its own TensorBoard event stream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any

from torch.utils.tensorboard import SummaryWriter


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
REWARD_PATTERN = re.compile(rf"Evaluation result:\s*reward=({FLOAT_PATTERN})")
GLOBAL_STEP_PATTERN = re.compile(r"training/global_step['\"]?\s*:\s*(\d+)")
RESUME_STEP_PATTERN = re.compile(r"Setting global step to\s+(\d+)")
REPEATED_PATTERN = re.compile(r"\[repeated\s+(\d+)x\s+across cluster\]")
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class RolloutMetrics:
    writer: Any
    rollouts_per_step: int
    training_step: int = 1
    completed: int = 0
    reward_sum: float = 0.0
    success_count: int = 0
    reward_min: float | None = None
    reward_max: float | None = None

    def __post_init__(self) -> None:
        self._write_progress()

    @property
    def event_step(self) -> int:
        return (self.training_step - 1) * self.rollouts_per_step + self.completed

    def _write_progress(self) -> None:
        step = self.event_step
        self.writer.add_scalar("rollout_live/completed", self.completed, step)
        self.writer.add_scalar(
            "rollout_live/total", self.rollouts_per_step, step
        )
        self.writer.add_scalar(
            "rollout_live/progress_percent",
            100.0 * self.completed / self.rollouts_per_step,
            step,
        )
        self.writer.add_scalar("rollout_live/training_step", self.training_step, step)
        self.writer.flush()

    def record_reward(self, reward: float, count: int = 1) -> None:
        remaining = self.rollouts_per_step - self.completed
        accepted = min(max(count, 0), max(remaining, 0))
        if accepted == 0:
            return

        self.completed += accepted
        self.reward_sum += reward * accepted
        if reward > 0:
            self.success_count += accepted
        self.reward_min = reward if self.reward_min is None else min(self.reward_min, reward)
        self.reward_max = reward if self.reward_max is None else max(self.reward_max, reward)

        step = self.event_step
        self._write_progress()
        self.writer.add_scalar("rollout_live/reward_latest", reward, step)
        self.writer.add_scalar(
            "rollout_live/reward_running_mean",
            self.reward_sum / self.completed,
            step,
        )
        self.writer.add_scalar("rollout_live/reward_min", self.reward_min, step)
        self.writer.add_scalar("rollout_live/reward_max", self.reward_max, step)
        self.writer.add_scalar(
            "rollout_live/success_rate",
            self.success_count / self.completed,
            step,
        )
        self.writer.flush()

    def start_after_global_step(self, finished_step: int) -> None:
        next_step = finished_step + 1
        if next_step <= self.training_step:
            return
        self.training_step = next_step
        self.completed = 0
        self.reward_sum = 0.0
        self.success_count = 0
        self.reward_min = None
        self.reward_max = None
        self._write_progress()

    def heartbeat(self, elapsed_seconds: float) -> None:
        self.writer.add_scalar(
            "rollout_live/elapsed_minutes",
            elapsed_seconds / 60.0,
            self.event_step,
        )
        self.writer.flush()

    def consume_line(self, line: str) -> None:
        clean = ANSI_PATTERN.sub("", line)
        reward_match = REWARD_PATTERN.search(clean)
        if reward_match is not None:
            repeated_match = REPEATED_PATTERN.search(clean)
            count = int(repeated_match.group(1)) if repeated_match else 1
            self.record_reward(float(reward_match.group(1)), count)

        step_match = GLOBAL_STEP_PATTERN.search(clean)
        if step_match is not None:
            self.start_after_global_step(int(step_match.group(1)))
            return

        resume_match = RESUME_STEP_PATTERN.search(clean)
        if resume_match is not None:
            self.start_after_global_step(int(resume_match.group(1)))


def tail_log(
    log_file: Path,
    tensorboard_dir: Path,
    *,
    rollouts_per_step: int,
    poll_interval: float,
    heartbeat_seconds: float,
) -> None:
    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while not log_file.exists() and not stopped:
        time.sleep(poll_interval)
    if stopped:
        return

    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(tensorboard_dir),
        max_queue=1,
        flush_secs=1,
        filename_suffix=".rollout_monitor",
    )
    metrics = RolloutMetrics(writer, rollouts_per_step)
    started_at = time.monotonic()
    last_heartbeat = started_at

    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as handle:
            while not stopped:
                line = handle.readline()
                if line:
                    metrics.consume_line(line)
                    continue
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_seconds:
                    metrics.heartbeat(now - started_at)
                    last_heartbeat = now
                time.sleep(poll_interval)

            # The training process is already stopped when the runner sends
            # SIGTERM, so drain text that reached the log just before shutdown.
            for line in handle:
                metrics.consume_line(line)
    finally:
        metrics.heartbeat(time.monotonic() - started_at)
        writer.close()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a veRL training log and write live rollout metrics"
    )
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--rollouts-per-step", type=positive_int, required=True)
    parser.add_argument("--poll-interval", type=positive_float, default=0.5)
    parser.add_argument("--heartbeat-seconds", type=positive_float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        f"Monitoring {args.log_file} -> {args.tensorboard_dir} "
        f"({args.rollouts_per_step} rollouts/step)",
        flush=True,
    )
    tail_log(
        args.log_file,
        args.tensorboard_dir,
        rollouts_per_step=args.rollouts_per_step,
        poll_interval=args.poll_interval,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
