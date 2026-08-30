from verl.experimental.agent_loop.tool_agent_loop import (
    merge_leading_system_messages,
)


def test_merges_dataset_context_and_tau2_policy_into_one_system_message():
    messages = [
        {"role": "system", "content": "current date"},
        {"role": "system", "content": "domain policy"},
        {"role": "user", "content": "hello"},
    ]

    merged = merge_leading_system_messages(messages)

    assert merged == [
        {"role": "system", "content": "current date\n\ndomain policy"},
        {"role": "user", "content": "hello"},
    ]
    assert messages[0]["content"] == "current date"


def test_does_not_reorder_a_late_invalid_system_message():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "invalid late policy"},
    ]

    assert merge_leading_system_messages(messages) == messages
