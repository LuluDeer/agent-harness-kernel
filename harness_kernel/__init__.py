"""harness_kernel — Agent runtime primitives for tool-calling-less upstreams.

Two independent subsystems:

``harness_kernel.parsing``
    Rebuilds structured ``tool_calls`` from a model that was never trained to
    emit them. Covers protocol injection artifacts, multi-shape output,
    malformed-JSON repair, and non-progress loop detection.

``harness_kernel.hash_chain``
    Reattaches a stateless request to its upstream session by hashing
    progressively truncated context slices, with four-tier degraded matching
    (exact / partial / branch / fuzzy).

Neither module carries any vendor-specific knowledge. Both are plain Python
with no import-time side effects beyond logger construction.
"""
from . import config, parsing  # noqa: F401

__all__ = ["config", "parsing", "hash_chain"]
__version__ = "0.1.0"


def __getattr__(name):
    # hash_chain imports pymysql/redis; keep it lazy so the parsing half of the
    # kernel is usable with zero third-party dependencies installed.
    if name == "hash_chain":
        from . import hash_chain as _hc
        return _hc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
