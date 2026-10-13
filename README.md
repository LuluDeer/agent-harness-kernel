# agent-harness-kernel

Runtime primitives for driving an LLM that has **no native function calling and
no server-side context management**.

Two independent modules:

| Module | Lines | Deps | Problem it solves |
|---|---|---|---|
| `harness_kernel.parsing` | 913 | **stdlib only** | Rebuild structured `tool_calls` from a model that only emits prose |
| `harness_kernel.hash_chain` | 653 | `pymysql`, `redis` | Reattach a stateless request to its session via progressive context hashing |

```bash
pip install -e .          # parsing half needs nothing else
pip install -e ".[chain]" # adds pymysql + redis for the matcher
pytest -q                 # 39 tests, no external services required
```

---

## The problem

Some inference endpoints are conversational but **stateless in the wrong way**:
you get a chat surface, but no `tools` parameter, no `tool_calls` in the
response, and no context handle you control. The endpoint identifies a
conversation by an opaque session id, and it manages the history itself.

That breaks every standard agent loop. The framework expects:

```
messages[] -> model -> tool_calls[] -> execute -> append results -> repeat
```

What actually comes back is a paragraph of prose describing what the model
*wants* to do. There is nothing to dispatch.

`agent-harness-kernel` is the layer that closes that gap. It does not call any
vendor API and it holds no vendor-specific knowledge — it is the parsing and
session-identity half of the runtime.

---

## Part 1 — `parsing`: rebuilding tool calling

### Protocol injection

OpenAI's `tools` schema is serialized into the prompt as text, wrapped in
boundary markers chosen to be effectively absent from natural model output —
U+241E / U+241F (Record / Unit Separator):

```
␞{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]}␟
```

Control characters beat prose delimiters (` ```json `, `<tool>`) because they
cannot be produced by accident and cannot be confused with markdown the model
is legitimately writing.

### Reverse parsing

`parse_xml_tool_calls()` runs a **primary path plus two fallback tiers**:

1. **Primary** — marker-delimited JSON. Handles multiple calls per turn, and
   recovers a **truncated tail** (stream cut before the closing marker) instead
   of discarding the turn.
2. **Fallback** — ` ```json ` fenced blocks.
3. **Fallback** — `<tool>...</tool>` XML, for legacy output shapes.

Because models drift across a long conversation, both the marker set
(`TOOL_CALL_START_ALTS`) and legacy formats (`ALLOW_LEGACY_TOOL_CALL_FORMATS`)
are opt-in configurable rather than hardcoded.

### JSON repair

Real model output is invalid JSON more often than you expect — unescaped
Windows paths, stray inner quotes, missing braces, single backslashes. Rather
than dropping the call, `_fix_json()` applies layered repair:

| Repair | Target failure |
|---|---|
| `_fix_windows_path_escapes` | `"C:\Users\x"` — invalid `\U` escape |
| `_fix_unescaped_inner_quotes` | quotes inside a string value |
| `_fix_missing_mid_braces` | truncated object literal |
| `_escape_single_backslashes` | lone `\` from a file path |

A dropped tool call is a silent agent failure. Repairing is almost always the
better trade.

### Hard gates and loop breaking

Unbounded agent loops are the default failure mode. Three independent brakes:

```python
MAX_TOOL_ROUNDS_HARD   = 500   # runaway ceiling
MAX_TOOL_CALLS_PER_TURN = 8    # per-turn fan-out cap
deduplicate_tool_calls()       # collapse identical (name, arguments)
```

`should_block_tool_calls()` is the interesting one. It detects a model
**spinning in place** — repeating the same call with no new information:

```python
same_tail    = _count_same_tail_rounds(messages, candidate_calls)
has_progress = _has_progress_vs_last(messages, candidate_calls)
if same_tail >= 4 and not has_progress:
    return True, f"no_progress_repeat:{same_tail}"
```

Note the `and`: repetition alone is legitimate (polling, retry). Repetition
**without progress** is the actual bug. Distinguishing the two is the whole
point, and it is what keeps the guard from firing on healthy work.

### Streaming reasoning/final split

`IncrementalReasoningSplitter` separates a model's `>`-prefixed reasoning from
its final answer incrementally, so it works on a stream rather than only after
the full response arrives.

---

## Part 2 — `hash_chain`: matching a request to its session

### The constraint

The upstream will not let you pass context. It gives you an opaque session id
on creation and expects you to reattach by... existing. But your caller is
stateless HTTP, and may hit a different worker process. You must recover the
session id **from the request body alone**.

### Context-pruning hash

Hash the message list, then hash it again with the last message dropped, then
the last two, and so on — storing the whole chain:

```
slice_index=1  ->  messages[:]      ->  round N
slice_index=2  ->  messages[:-1]    ->  round N-1
slice_index=3  ->  messages[:-2]    ->  round N-2
```

This yields a useful invariant, and the test suite asserts it directly:

```python
round_n = _msgs(4)
round_n1 = round_n + [tail...]
assert (mgr._compute_message_hash(round_n,   1)      # full context
        == mgr._compute_message_hash(round_n1, 3))   # truncated to match
```

**A later round's truncated slice reproduces an earlier round's full hash.**
That is what makes backward traversal possible.

`tool_calls` are folded into the digest — otherwise two distinct tool rounds
would collapse into the same session identity.

### Four-tier degraded matching

Exact matching fails constantly in practice: models drop a prefix, a system
message gets injected, output format drifts as context grows. So matching
degrades instead of failing:

| Tier | Strategy | Confidence |
|---|---|---|
| `exact` | Full-context hash hit | 1.0 |
| `partial` | Walk the chain to an earlier matching round | high, scaled by `round_diff` |
| `branch` | Resolve a forked conversation via `create_branch()` | inherited |
| `fuzzy` | Character-level similarity | `< HASH_CHAIN_TOLERANCE` |

```python
result = find_matching_chatbot(messages, client_id, model_id, client_ip)
if result is None:
    chatbot_id = create_new_session()   # only when every tier misses
```

The tiers are tried in order and the first hit wins. `MatchResult` records
which tier fired, so degradation is observable rather than silent.

Redis caches the hot slices (first 10, 6h TTL); MySQL holds the durable chain.
**Redis is optional** — a failed connection degrades to DB-only, it does not
break matching.

---

## Design notes

**No import-time side effects.** `HashChainManager(autoconnect=False)` is the
default and touches no network. The original design connected at import, which
made the module unimportable in any environment without live credentials.

**Zero-dep parsing half.** `parsing.py` imports only `re`, `json`, `uuid`,
`typing`. You can adopt it standalone without pulling in a database driver.

**Fail-soft by default.** `store_hash_chain` returns `False` rather than
raising; `cleanup_expired_sessions` logs and continues. Session-matching
infrastructure should degrade the agent, not kill the request path.

---

## Testing

```bash
pytest -q     # 39 passed
```

The suite covers marker location, all three extraction tiers, truncated-stream
recovery, four JSON-repair paths, deduplication, both loop-detection branches,
the pruning-hash prefix invariant, the similarity metric, and the
no-side-effect construction contract. MySQL/Redis are not required — the
DB-backed tiers are integration-level.

---

## Limitations

- **Session state is in memory.** Single-worker deployment is required for the
  in-process stores; the hash chain is what makes multi-worker *possible*, not
  automatic.
- **`pymysql`/`redis` clients are synchronous.** Under high concurrency these
  block the event loop; migrate to async clients or a pool before scaling.
- **The fuzzy tier is character-level**, not semantic. It is a last resort and
  its confidence value should be treated as a weak signal.
- **`MAX_TOOL_ROUNDS_HARD = 500` is deliberately high.** It is a runaway guard,
  not a product limit — the progress detector is what normally stops loops.

## License

MIT
