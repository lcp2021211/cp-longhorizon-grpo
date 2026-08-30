import importlib.util
from pathlib import Path

import yaml

from verl.tools.schemas import OpenAIFunctionToolSchema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "scripts/train/grpo/gen_tool_config.py"
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "project_gen_tool_config", GENERATOR_PATH
)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
GENERATOR_MODULE = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(GENERATOR_MODULE)
build_tool_config = GENERATOR_MODULE.build_tool_config


def _no_argument_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "done",
            "description": "No-argument test tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_verl_and_generator_accept_schema_without_required():
    schema = _no_argument_schema()
    parsed = OpenAIFunctionToolSchema.model_validate(schema)
    config = build_tool_config("airline", [schema])
    parameters = config["tools"][0]["tool_schema"]["function"]["parameters"]
    assert parsed.function.parameters.required == []
    assert "required" not in schema["function"]["parameters"]
    assert parameters["required"] == []


def test_checked_in_tau2_tool_config_has_required_for_every_tool():
    path = PROJECT_ROOT / "configs/tool_config/tau_bench_airline_tools.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [
        tool["tool_schema"]["function"]["name"]
        for tool in config["tools"]
        if "required" not in tool["tool_schema"]["function"]["parameters"]
    ]
    assert missing == []
