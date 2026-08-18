"""Evaluate an exported policy with the current tau2 Gym adapter.

Without ``--split-file`` this reports the official tau2 split as one overall
set, which is the supported path for the new ablations. ``--split-file`` keeps
the historical 50-task covered/uncovered report available for legacy results;
that split must not be used for the current official train/test experiment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os

import yaml
import numpy as np

from src.envs.tau_bench_wrapper import TauBenchWrapper
from src.models.vllm_policy import VLLMPolicy
from src.evaluation.pass_k_eval import run_eval

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pass_at_k(n: int, c: int, k: int) -> float:
    """HumanEval-style unbiased estimator"""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate_subset(per_task_results: list[dict], task_ids_subset: set[int]) -> dict:
    """
    从 EvalReport.per_task_results 里挑出指定 task_id 的子集，重新算聚合指标
    """
    subset = [r for r in per_task_results if r["task_id"] in task_ids_subset]
    if not subset:
        return {
            "n_tasks": 0, "pass_at_1": 0.0, "pass_hat_1": 0.0,
            "pass_hat_4": 0.0, "pass_hat_8": 0.0,
            "avg_turns": 0.0, "avg_tool_calls": 0.0, "error_rate": 0.0,
            "avg_reasoning_tokens_per_turn": 0.0,
            "max_reasoning_tokens_single_turn": 0,
        }

    pass_1_list, pass_4_list, pass_8_list, pass_at_1_list = [], [], [], []
    all_turns, all_tool_calls, all_errors = [], [], []
    all_reasoning_tokens = []  # 每个 assistant turn 的 token 数

    for r in subset:
        n = r["total_samples"]
        c = r["success_count"]
        pass_1_list.append(pass_at_k(n, c, 1))
        pass_at_1_list.append(1.0 if c > 0 else 0.0)
        if n >= 4:
            pass_4_list.append(pass_at_k(n, c, 4))
        if n >= 8:
            pass_8_list.append(pass_at_k(n, c, 8))
        for tr in r["trajectories"]:
            all_turns.append(tr["num_turns"])
            all_tool_calls.append(tr["num_tool_calls"])
            all_errors.append(1.0 if tr.get("error") else 0.0)

            # 收集每轮 assistant content 的 token 数
            per_turn_tokens = tr.get("per_turn_assistant_content_tokens", [])
            all_reasoning_tokens.extend(per_turn_tokens)

    return {
        "n_tasks": len(subset),
        "pass_at_1": float(np.mean(pass_at_1_list)),
        "pass_hat_1": float(np.mean(pass_1_list)),
        "pass_hat_4": float(np.mean(pass_4_list)) if pass_4_list else 0.0,
        "pass_hat_8": float(np.mean(pass_8_list)) if pass_8_list else 0.0,
        "avg_turns": float(np.mean(all_turns)) if all_turns else 0.0,
        "avg_tool_calls": float(np.mean(all_tool_calls)) if all_tool_calls else 0.0,
        "error_rate": float(np.mean(all_errors)) if all_errors else 0.0,
        "avg_reasoning_tokens_per_turn": float(np.mean(all_reasoning_tokens)) if all_reasoning_tokens else 0.0,
        "max_reasoning_tokens_single_turn": int(max(all_reasoning_tokens)) if all_reasoning_tokens else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split-file", type=str, default=None,
                        help="split.json 路径（选项 D 必填，否则只报 overall）")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Match the training interaction config: the same local or hosted user
    # simulator can be selected without generating a second YAML file.
    for config_key, env_name in (
        ("user_model", "TAU2_USER_MODEL"),
        ("user_provider", "TAU2_USER_PROVIDER"),
        ("user_base_url", "TAU2_USER_BASE_URL"),
    ):
        if value := os.environ.get(env_name):
            cfg["env"][config_key] = value

    if args.tiny:
        cfg["eval"]["num_tasks"] = 4
        cfg["eval"]["num_samples_per_task"] = 2
        cfg["output"]["dir"] = cfg["output"]["dir"] + "_tiny"

    # 读 split
    seen_ids: set[int] = set()
    unseen_ids: set[int] = set()
    if args.split_file:
        with open(args.split_file) as f:
            split = json.load(f)
        seen_ids = set(split["seen_task_ids"])
        unseen_ids = set(split["unseen_task_ids"])
        print(f"读取 split: {len(seen_ids)} seen + {len(unseen_ids)} unseen")

    wrapper = TauBenchWrapper(
        env_name=cfg["env"]["name"],
        user_strategy=cfg["env"]["user_strategy"],
        user_model=cfg["env"]["user_model"],
        user_provider=cfg["env"]["user_provider"],
        user_base_url=cfg["env"].get("user_base_url"),
        task_split=cfg["env"]["task_split"],
    )

    shared_policy = VLLMPolicy(**cfg["policy"])

    def policy_factory():
        return shared_policy

    report = run_eval(
        wrapper=wrapper,
        policy_factory=policy_factory,
        num_tasks=cfg["eval"]["num_tasks"],
        num_samples_per_task=cfg["eval"]["num_samples_per_task"],
        max_turns=cfg["eval"]["max_turns"],
        num_workers=cfg["eval"]["num_workers"],
        output_dir=cfg["output"]["dir"],
    )

    print(f"\nSFT 评测原始报告: {cfg['output']['dir']}/eval_report.json")

    # ----------  分组聚合 ----------
    if args.split_file and seen_ids:
        # 从采集结果知道哪些 task 有成功 trajectory（读 summary.json 或 task 文件）
        collect_dir = Path(args.split_file).parent
        covered_ids: set[int] = set()
        for tf in sorted(collect_dir.glob("task_*.jsonl")):
            if "contaminated" in tf.name:
                continue
            task_id = int(tf.stem.split("_")[1])
            if tf.stat().st_size > 0:  # 非空文件 = 有成功 trajectory
                covered_ids.add(task_id)

        covered_seen = seen_ids & covered_ids
        uncovered_seen = seen_ids - covered_ids

        covered_metrics = aggregate_subset(report.per_task_results, covered_seen)
        uncovered_seen_metrics = aggregate_subset(report.per_task_results, uncovered_seen)
        unseen_metrics = aggregate_subset(report.per_task_results, unseen_ids)
        overall_metrics = {
            "n_tasks": report.num_tasks,
            "pass_at_1": report.pass_at_1,
            "pass_hat_1": report.pass_hat_1,
            "pass_hat_4": report.pass_hat_4,
            "pass_hat_8": report.pass_hat_8,
            "avg_turns": report.avg_turns,
            "avg_tool_calls": report.avg_tool_calls,
            "error_rate": report.error_rate,
        }

        # 写 split_eval_report.json
        split_report = {
            "config": cfg,
            "split_file": args.split_file,
            "covered_seen": covered_metrics,
            "uncovered_seen": uncovered_seen_metrics,
            "unseen": unseen_metrics,
            "overall": overall_metrics,
        }
        out_path = Path(cfg["output"]["dir"]) / "split_eval_report.json"
        with open(out_path, "w") as f:
            json.dump(split_report, f, indent=2, ensure_ascii=False)

        # 简化版输出: 四组 + 关键判据
        BASELINE_OVERALL_P1 = 0.160     # 数字（72B-user）
        BASELINE_OVERALL_PA1 = 0.340
        BASELINE_OVERALL_TURNS = 12.29

        print()
        print("=" * 60)
        print("  SFT 评测 (covered-seen / uncovered-seen / unseen / overall)")
        print("=" * 60)
        print(f"  covered-seen   ({covered_metrics['n_tasks']:>2}): "
              f"pass^1={covered_metrics['pass_hat_1']:.3f}  "
              f"pass@1={covered_metrics['pass_at_1']:.3f}  "
              f"avg_turns={covered_metrics['avg_turns']:.2f}")
        print(f"  uncovered-seen ({uncovered_seen_metrics['n_tasks']:>2}): "
              f"pass^1={uncovered_seen_metrics['pass_hat_1']:.3f}  "
              f"pass@1={uncovered_seen_metrics['pass_at_1']:.3f}  "
              f"avg_turns={uncovered_seen_metrics['avg_turns']:.2f}")
        print(f"  unseen         ({unseen_metrics['n_tasks']:>2}): "
              f"pass^1={unseen_metrics['pass_hat_1']:.3f}  "
              f"pass@1={unseen_metrics['pass_at_1']:.3f}  "
              f"avg_turns={unseen_metrics['avg_turns']:.2f}")
        print(f"  overall        ({overall_metrics['n_tasks']:>2}): "
              f"pass^1={overall_metrics['pass_hat_1']:.3f}  "
              f"pass@1={overall_metrics['pass_at_1']:.3f}  "
              f"avg_turns={overall_metrics['avg_turns']:.2f}")
        print(f"  Baseline        ({50:>2}): "
              f"pass^1={BASELINE_OVERALL_P1:.3f}  "
              f"pass@1={BASELINE_OVERALL_PA1:.3f}  "
              f"avg_turns={BASELINE_OVERALL_TURNS:.2f}")
        print("=" * 60)

        # 关键判据
        delta_overall = overall_metrics["pass_hat_1"] - BASELINE_OVERALL_P1
        covered_minus_unseen = covered_metrics["pass_hat_1"] - unseen_metrics["pass_hat_1"]

        print(f"\n  Δ overall pass^1 vs baseline: {delta_overall:+.3f}  "
              f"{'✅ 达标 (>=+0.05)' if delta_overall >= 0.05 else '⚠️ 未达标'}")
        print(f"  covered-seen − unseen gap:   {covered_minus_unseen:+.3f}  "
              f"({'过拟合明显' if covered_minus_unseen > 0.10 else 'gap 小，泛化好' if covered_minus_unseen < 0.02 else 'gap 中等'})")
        print(f"\n  分组报告: {out_path}")
    else:
        # Current tau2 mode: do not compare against the incompatible legacy
        # 50-task baseline. The JSON field names remain backward compatible.
        print("\n=== Current tau2 overall ===")
        print(f"  success rate (pass_hat_1): {report.pass_hat_1:.3f}")
        print(
            f"  task solved at least once among "
            f"{report.num_samples_per_task} samples (legacy pass_at_1): "
            f"{report.pass_at_1:.3f}"
        )
        print(f"  avg_turns: {report.avg_turns:.2f}")


if __name__ == "__main__":
    main()
