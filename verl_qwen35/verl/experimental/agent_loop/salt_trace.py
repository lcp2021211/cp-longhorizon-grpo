"""Stable trajectory keys used by SALT advantage refinement.

SALT only needs equality checks between transitions sampled on different
rollout workers.  Keep the representation JSON-safe and deterministic: Python's
built-in ``hash`` must not be used because its seed differs across processes.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str):
        return _normalize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _normalize_text(str(value))


def _canonical_json_or_text(value: str) -> dict[str, Any]:
    normalized = _normalize_text(value)
    try:
        decoded = json.loads(normalized)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"format": "text", "value": normalized}
    return {"format": "json", "value": _canonical_value(decoded)}


def canonical_key(value: Any) -> str:
    """Return a cross-process stable canonical string for a JSON-like value."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_root_key(messages: list[dict[str, Any]]) -> str:
    """Fingerprint the actual prompt seen before the first assistant action."""
    payload = canonical_key(messages).encode("utf-8")
    return "prompt:" + hashlib.blake2b(payload, digest_size=20).hexdigest()


def build_tool_action_key(tool_calls: Iterable[Any]) -> str:
    calls = []
    for call in tool_calls:
        raw_arguments = getattr(call, "arguments", "")
        try:
            arguments = {
                "format": "json",
                "value": _canonical_value(json.loads(raw_arguments)),
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = {
                "format": "raw",
                "value": _normalize_text(str(raw_arguments)),
            }
        calls.append(
            {
                "name": _normalize_text(str(getattr(call, "name", ""))),
                "arguments": arguments,
            }
        )
    return canonical_key({"type": "tool", "calls": calls})


def build_text_action_key(text: str) -> str:
    return canonical_key({"type": "respond", "content": text})


def build_tool_observation_key(tool_responses: Iterable[Any]) -> str:
    observations = []
    for response in tool_responses:
        observations.append(
            {
                "text": _canonical_json_or_text(
                    getattr(response, "text", "") or ""
                ),
                "has_image": getattr(response, "image", None) is not None,
                "has_video": getattr(response, "video", None) is not None,
            }
        )
    return canonical_key({"type": "tool", "observations": observations})


def build_text_observation_key(text: str) -> str:
    return canonical_key({"type": "user", "content": text or ""})
