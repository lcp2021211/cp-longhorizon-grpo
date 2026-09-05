"""Generate veRL tool_config YAML from the current tau2 domain registry.

Usage:
    python scripts/train/grpo/gen_tool_config.py \
        --env airline \
        --output configs/tool_config/tau_bench_airline_tools.yaml

Includes tau2 Gym's ``done`` tool so tool-side termination is represented in
the same schema used during rollout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_tool_schemas(env_name: str = "airline", task_index: int = 0) -> list[dict]:
    del task_index  # schemas are domain-level in tau2
    from src.envs.tau2_adapter import get_tau2_tool_schemas

    return get_tau2_tool_schemas(env_name, include_done=True)


def build_tool_config(env_name: str, schemas: list[dict]) -> dict:
    del env_name  # class names are currently shared by the supported domain
    tools = []
    for schema in schemas:
        schema = deepcopy(schema)
        func = schema.get("function", schema)
        parameters = func.setdefault(
            "parameters", {"type": "object", "properties": {}}
        )
        # veRL serializes this field explicitly. JSON Schema itself allows it
        # to be absent when a function has no required arguments.
        parameters.setdefault("required", [])
        name = func["name"]
        cls_name = f"src.envs.tau_bench_tools.TauBench_{name}_Tool"
        tools.append({
            "class_name": cls_name,
            "config": {"type": "native"},
            "tool_schema": schema,
        })
    return {"tools": tools}


def main():
    parser = argparse.ArgumentParser(description="Generate veRL tool config from tau2-bench")
    parser.add_argument("--env", default="airline", help="tau2 domain name")
    parser.add_argument("--task-id", type=int, default=0, help="deprecated; kept for CLI compatibility")
    parser.add_argument("--output", default="configs/tool_config/tau_bench_airline_tools.yaml")
    args = parser.parse_args()

    schemas = get_tool_schemas(args.env, args.task_id)
    print(f"Extracted {len(schemas)} tool schemas from {args.env} env")

    config = build_tool_config(args.env, schemas)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Written to {output_path}")
    for t in config["tools"]:
        print(f"  - {t['class_name']}")


if __name__ == "__main__":
    main()
