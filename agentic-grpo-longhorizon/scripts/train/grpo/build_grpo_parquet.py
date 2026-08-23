"""
Build GRPO training/validation parquet files.

Usage:
    python scripts/train/grpo/build_grpo_parquet.py \
        --train-task-split train --val-task-split test \
        --output-train experiments/tau2/train.parquet \
        --output-val experiments/tau2/val.parquet

Design:
- Each row = one official tau2 task ID; rollout.n expands it at runtime.
- prompt contains date grounding; tau2 policy + first user message are injected
  from Interaction.get_initial_messages after Gym reset.
- extra_info: index, task_id, split, interaction_kwargs
- No traj_uid column (veRL repeat mechanism makes it non-unique)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT = (
    "# Current Date Context\n"
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, "
    "always assume they refer to 2024. "
    "All flight searches and reservations should use 2024 dates unless explicitly stated otherwise."
)

INTERACTION_NAME = "tau_bench_airline"
# Only used by the explicit legacy --seen-task-ids path.
NUM_LEGACY_AIRLINE_TASKS = 50


def build_rows(task_ids: list[int | str], split: str) -> list[dict]:
    rows = []
    for idx, tid in enumerate(task_ids):
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
            "extra_info": {
                "index": idx,
                "task_id": tid,
                "split": split,
                "interaction_kwargs": {
                    "name": INTERACTION_NAME,
                    "task_id": tid,
                },
            },
            "data_source": INTERACTION_NAME,
            "reward_model": {"ground_truth": ""},
            "ability": INTERACTION_NAME,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build GRPO parquet datasets")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--seen-task-ids", type=str, help="Comma-separated seen task IDs")
    group.add_argument("--seen-task-ids-from", type=str, help="Path to metadata.json with seen_task_ids")
    parser.add_argument("--output-train", default="experiments/vanilla/train.parquet")
    parser.add_argument("--output-val", default="experiments/vanilla/val.parquet")
    parser.add_argument("--num-total-tasks", type=int, default=NUM_LEGACY_AIRLINE_TASKS)
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--train-task-split", default="train")
    parser.add_argument("--val-task-split", default="test")
    args = parser.parse_args()

    if args.seen_task_ids or args.seen_task_ids_from:
        if args.seen_task_ids:
            seen_ids = [int(x.strip()) for x in args.seen_task_ids.split(",")]
        else:
            meta_path = Path(args.seen_task_ids_from)
            with open(meta_path) as f:
                meta = json.load(f)
            if "seen_task_ids" in meta:
                seen_ids = meta["seen_task_ids"]
            elif "covered_task_ids" in meta:
                seen_ids = meta["covered_task_ids"]
            else:
                raise ValueError(
                    f"No seen_task_ids or covered_task_ids in {meta_path}"
                )

        all_ids = list(range(args.num_total_tasks))
        unseen_ids = [t for t in all_ids if t not in seen_ids]
        seen_set = set(seen_ids)
        train_rows = build_rows(seen_ids, split="seen")
        val_rows = []
        for tid in all_ids:
            split_tag = "seen" if tid in seen_set else "unseen"
            val_rows.extend(build_rows([tid], split=split_tag))
    else:
        from src.envs.tau2_adapter import get_tau2_task_ids

        train_ids = get_tau2_task_ids(args.domain, args.train_task_split)
        val_ids = get_tau2_task_ids(args.domain, args.val_task_split)
        seen_ids = train_ids
        unseen_ids = val_ids
        train_rows = build_rows(train_ids, split=args.train_task_split)
        val_rows = build_rows(val_ids, split=args.val_task_split)

    train_path = Path(args.output_train)
    val_path = Path(args.output_val)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Train: {len(train_rows)} rows -> {train_path}")
    print(f"Val:   {len(val_rows)} rows -> {val_path}")
    print(f"Train task IDs ({len(seen_ids)}): {seen_ids[:10]}{'...' if len(seen_ids) > 10 else ''}")
    print(f"Val task IDs ({len(unseen_ids)}): {unseen_ids[:10]}{'...' if len(unseen_ids) > 10 else ''}")


if __name__ == "__main__":
    main()
