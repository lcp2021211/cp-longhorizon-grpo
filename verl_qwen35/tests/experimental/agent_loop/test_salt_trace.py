from types import SimpleNamespace

from verl.experimental.agent_loop.salt_trace import (
    build_text_action_key,
    build_tool_action_key,
    build_tool_observation_key,
)


def test_tool_action_key_canonicalizes_json_argument_order():
    left = SimpleNamespace(name="lookup", arguments='{"b": 2, "a": 1}')
    right = SimpleNamespace(name="lookup", arguments='{"a":1,"b":2}')

    assert build_tool_action_key([left]) == build_tool_action_key([right])


def test_text_and_tool_observation_keys_normalize_surface_whitespace():
    assert build_text_action_key("please   continue\nnow") == build_text_action_key(
        "please continue now"
    )
    left = SimpleNamespace(text='{"b": 2, "a": 1}', image=None, video=None)
    right = SimpleNamespace(text='{"a":1,"b":2}', image=None, video=None)
    assert build_tool_observation_key([left]) == build_tool_observation_key([right])


def test_plain_text_matching_does_not_parse_json_literals():
    assert build_text_action_key('"continue"') != build_text_action_key("continue")


def test_tool_argument_preserves_json_string_type():
    string_value = SimpleNamespace(name="lookup", arguments='{"id":"123"}')
    numeric_value = SimpleNamespace(name="lookup", arguments='{"id":123}')
    assert build_tool_action_key([string_value]) != build_tool_action_key([numeric_value])
