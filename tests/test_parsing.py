"""Behavioural tests for harness_kernel.parsing.

These exercise the paths that matter in production: recovering structured
tool_calls from messy natural-language output, repairing malformed JSON, and
detecting a stuck loop. Run with `python -m pytest tests -q`.
"""
import json

from harness_kernel import parsing as P
from harness_kernel.config import TOOL_CALL_START, TOOL_CALL_END


def _wrap(payload: str) -> str:
    """Wrap a payload in the injected protocol boundary markers."""
    return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"


# ===================== boundary markers =====================

def test_find_tc_start_locates_marker():
    text = f"thinking... {TOOL_CALL_START}{{}}"
    pos, length = P.find_tc_start(text)
    assert pos == len("thinking... ")
    assert length == len(TOOL_CALL_START)


def test_find_tc_start_returns_sentinel_when_absent():
    assert P.find_tc_start("no markers here") == (-1, 0)


def test_contains_tc_marker():
    assert P.contains_tc_marker(_wrap('{"tool_calls": []}')) is True
    assert P.contains_tc_marker("plain answer") is False


# ===================== JSON repair =====================

def test_safe_json_obj_parses_valid_object():
    assert P._safe_json_obj('{"a": 1}') == {"a": 1}


def test_safe_json_obj_survives_trailing_garbage():
    assert P._safe_json_obj('{"a": 1} trailing noise') == {"a": 1}


def test_fix_json_repairs_windows_backslash_paths():
    # A model emitting a raw Windows path produces invalid JSON escapes;
    # the repair layer must make it parseable rather than dropping the call.
    raw = '{"path": "C:\\Users\\test\\file.txt"}'
    obj = P._safe_json_obj(P._fix_json(raw))
    assert isinstance(obj, dict)


def test_parse_tool_json_recovers_call_from_envelope():
    out = []
    P._parse_tool_json(
        '{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]}', out
    )
    assert len(out) == 1
    assert out[0]["function"]["name"] == "read_file"
    assert json.loads(out[0]["function"]["arguments"]) == {"path": "a.txt"}


def test_parse_tool_json_ignores_non_object_and_nameless_calls():
    out = []
    P._parse_tool_json('[1, 2, 3]', out)
    P._parse_tool_json('{"tool_calls": [{"arguments": {}}]}', out)
    assert out == []


# ===================== multi-format extraction =====================

def test_parse_primary_marker_path():
    calls = P.parse_xml_tool_calls(_wrap('{"tool_calls": [{"name": "search", "arguments": {"q": "x"}}]}'))
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"


def test_fallback_to_xml_tool_tag():
    # The deprecated <tool> shape must still be recovered when the primary
    # markers never appear.
    text = '<tool>{"tool_calls": [{"name": "search", "arguments": {"q": "x"}}]}</tool>'
    calls = P.parse_xml_tool_calls(text)
    assert len(calls) >= 1
    assert any(c["function"]["name"] == "search" for c in calls)


def test_fallback_to_json_code_block():
    text = 'Here you go:\n```json\n{"tool_calls": [{"name": "ping", "arguments": {}}]}\n```\n'
    calls = P.parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ping"


def test_truncated_stream_recovered_without_end_marker():
    # A stream cut mid-flight loses the closing marker; the parser must still
    # salvage the call rather than silently dropping it.
    text = TOOL_CALL_START + '{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]}'
    calls = P.parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"


def test_parse_xml_tool_calls_ignores_plain_text():
    assert P.parse_xml_tool_calls("just prose") == []


# ===================== deduplication =====================

def test_deduplicate_removes_identical_calls():
    call = P._make_tool_call("read_file", {"path": "a.txt"})
    assert len(P.deduplicate_tool_calls([call, dict(call)])) == 1


def test_deduplicate_keeps_distinct_arguments():
    a = P._make_tool_call("read_file", {"path": "a.txt"})
    b = P._make_tool_call("read_file", {"path": "b.txt"})
    assert len(P.deduplicate_tool_calls([a, b])) == 2


# ===================== loop detection =====================

def _assistant_turn(call):
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def _tool_result():
    return {"role": "tool", "content": "ok"}


def test_no_block_when_no_candidate_calls():
    blocked, reason = P.should_block_tool_calls([], [])
    assert blocked is False
    assert reason == "no_calls"


def test_block_on_repeated_non_progressing_tail():
    # The same call repeated with identical output and no new arguments is the
    # signature of a model spinning in place.
    call = P._make_tool_call("read_file", {"path": "a.txt"})
    messages = []
    for _ in range(6):
        messages.append(_assistant_turn(dict(call)))
        messages.append(_tool_result())

    blocked, reason = P.should_block_tool_calls(messages, [dict(call)])
    assert blocked is True
    assert "no_progress_repeat" in reason


def test_no_block_when_arguments_advance():
    messages = []
    for i in range(6):
        messages.append(_assistant_turn(P._make_tool_call("read_file", {"path": f"{i}.txt"})))
        messages.append(_tool_result())

    blocked, _ = P.should_block_tool_calls(
        messages, [P._make_tool_call("read_file", {"path": "99.txt"})]
    )
    assert blocked is False


def test_hard_round_limit_is_enforced():
    from harness_kernel.config import MAX_TOOL_ROUNDS_HARD
    call = P._make_tool_call("read_file", {"path": "a.txt"})
    messages = []
    for i in range(MAX_TOOL_ROUNDS_HARD + 1):
        messages.append(_assistant_turn(P._make_tool_call("read_file", {"path": f"{i}.txt"})))
        messages.append(_tool_result())

    blocked, reason = P.should_block_tool_calls(messages, [dict(call)])
    assert blocked is True
    assert reason.startswith("hard_round_limit")


# ===================== reasoning / final split =====================

def test_split_reasoning_and_final_separates_quoted_reasoning():
    text = "> I should check the file first.\n> Then summarise.\n\nThe file has 3 lines."
    reasoning, final = P.split_reasoning_and_final(text)
    assert "check the file" in reasoning
    assert "3 lines" in final
    assert "check the file" not in final


def test_split_handles_empty_input():
    assert P.split_reasoning_and_final("") == ("", "")


def test_incremental_splitter_emits_on_feed_and_flush():
    s = P.IncrementalReasoningSplitter()
    out = s.feed("> reasoning line\n\nfinal text")
    assert isinstance(out, dict)
    out2 = s.flush()
    assert isinstance(out2, dict)
