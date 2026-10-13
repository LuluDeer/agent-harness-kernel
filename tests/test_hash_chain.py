"""Behavioural tests for harness_kernel.hash_chain.

Only the pure-logic surface is covered here: the progressive context-pruning
hash and the similarity metric. These are the parts that encode the actual
matching strategy, and they run without MySQL or Redis. The DB-backed tiers
(exact / partial / branch / fuzzy lookup) are integration-level and require a
live schema, so they are exercised separately.

Instantiation passes autoconnect=False, which also asserts the no-side-effect
contract: constructing the manager must not reach for the network.
"""
import pytest

pymysql = pytest.importorskip("pymysql", reason="hash_chain needs pymysql")
pytest.importorskip("redis", reason="hash_chain needs redis")

from harness_kernel.hash_chain import HashChainManager, MatchResult  # noqa: E402


@pytest.fixture()
def mgr():
    # autoconnect=False: no network access during tests.
    return HashChainManager(autoconnect=False)


def _msgs(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n)]


# ===================== construction contract =====================

def test_construction_is_side_effect_free(mgr):
    assert mgr.mysql_pool is None
    assert mgr.redis_client is None


# ===================== progressive pruning hash =====================

def test_slice_index_one_hashes_full_context(mgr):
    msgs = _msgs(5)
    assert mgr._compute_message_hash(msgs, 1) == mgr._compute_message_hash(msgs, 1)


def test_deeper_slices_differ_from_full_hash(mgr):
    # Each truncation level must produce a distinct hash, otherwise the
    # hash chain could not distinguish rounds.
    msgs = _msgs(6)
    hashes = {mgr._compute_message_hash(msgs, i) for i in range(1, 7)}
    assert len(hashes) == 6


def test_slice_beyond_length_returns_empty(mgr):
    msgs = _msgs(3)
    assert mgr._compute_message_hash(msgs, 4) == ""
    assert mgr._compute_message_hash(msgs, 99) == ""


def test_empty_message_list_returns_empty(mgr):
    assert mgr._compute_message_hash([], 1) == ""


def test_hash_is_sha256_hex(mgr):
    h = mgr._compute_message_hash(_msgs(2), 1)
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex


def test_hash_is_stable_across_equivalent_input(mgr):
    assert (mgr._compute_message_hash(_msgs(4), 2)
            == mgr._compute_message_hash(_msgs(4), 2))


def test_appending_a_message_changes_full_hash(mgr):
    base = _msgs(4)
    extended = base + [{"role": "user", "content": "tail"}]
    assert mgr._compute_message_hash(base, 1) != mgr._compute_message_hash(extended, 1)


def test_prefix_property_enables_match(mgr):
    """A later round's truncation must reproduce an earlier round's full hash.

    This is the property the whole matcher depends on: round N+1's [:-1]
    slice equals round N's full context, so the chain can be walked backwards.
    """
    round_n = _msgs(4)
    round_n1 = round_n + [{"role": "user", "content": "next"}, {"role": "assistant", "content": "ok"}]

    full_n = mgr._compute_message_hash(round_n, 1)
    truncated_n1 = mgr._compute_message_hash(round_n1, 3)  # drops last 2 -> == round_n
    assert full_n == truncated_n1


def test_tool_calls_participate_in_hash(mgr):
    # tool_calls must affect the digest, or two different tool rounds would
    # collapse to the same session identity.
    plain = [{"role": "assistant", "content": "x"}]
    with_tc = [{"role": "assistant", "content": "x",
                "tool_calls": [{"id": "1", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}]}]
    assert mgr._compute_message_hash(plain, 1) != mgr._compute_message_hash(with_tc, 1)


def test_list_content_is_flattened_to_text(mgr):
    msg = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    flat = [{"role": "user", "content": "hello"}]
    assert mgr._compute_message_hash(msg, 1) == mgr._compute_message_hash(flat, 1)


# ===================== similarity metric =====================

def test_identical_hashes_are_maximally_similar(mgr):
    h = "a" * 64
    assert mgr._calculate_hash_similarity(h, h) == 1.0


def test_empty_input_scores_zero(mgr):
    assert mgr._calculate_hash_similarity("", "abc") == 0.0
    assert mgr._calculate_hash_similarity("abc", "") == 0.0


def test_similarity_is_bounded(mgr):
    s = mgr._calculate_hash_similarity("a" * 64, "b" * 64)
    assert 0.0 <= s <= 1.0


def test_partial_overlap_scores_between_bounds(mgr):
    s = mgr._calculate_hash_similarity("a" * 32 + "b" * 32, "a" * 32 + "c" * 32)
    assert 0.0 < s < 1.0


# ===================== result contract =====================

def test_match_result_carries_degradation_tier():
    r = MatchResult(chatbot_id="cb1", match_type="partial", round_diff=2,
                    branch_id="main", confidence=0.8, matched_slice_index=3)
    assert r.match_type == "partial"
    assert r.confidence == 0.8


def test_all_four_tiers_are_representable():
    tiers = ["exact", "partial", "branch", "fuzzy"]
    for t in tiers:
        r = MatchResult(chatbot_id="cb", match_type=t, round_diff=0,
                        branch_id="main", confidence=1.0, matched_slice_index=1)
        assert r.match_type == t
