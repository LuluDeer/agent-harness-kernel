"""Runtime configuration for the harness kernel.

Everything here is deliberately dependency-free and environment-driven so the
kernel can be embedded in any host process. Values are read once at import
time, matching the original deployment semantics (a service restart is the
config reload boundary).
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def _int_env(name: str, default: int, minimum: int | None = None,
             maximum: int | None = None) -> int:
    """Read an int from the environment, falling back on bad/out-of-range input."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


# ===================== Tool-call protocol markers =====================
# Primary boundary markers. U+241E (Record Separator) and U+241F (Unit
# Separator) are chosen because they are effectively absent from natural
# model output, which keeps the injected protocol unambiguous.
TOOL_CALL_START = "\u241e"  # ␞
TOOL_CALL_END = "\u241f"    # ␟


def _parse_marker_list(env_name: str) -> list:
    """Comma-separated extra markers; empty by default (pure opt-in)."""
    raw = os.environ.get(env_name, "")
    return [m for m in raw.split(",") if m]


def _merge_markers(primary: str, alts: list) -> list:
    """Merge primary + alternates, de-duplicated, order preserved."""
    seen = set()
    out = []
    for m in [primary] + list(alts):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


# Models drift to other symbols over long conversations; alternates let the
# parser keep up without a code change.
TOOL_CALL_START_ALTS = _parse_marker_list("TOOL_CALL_START_ALTS")
TOOL_CALL_END_ALTS = _parse_marker_list("TOOL_CALL_END_ALTS")
TOOL_CALL_START_MARKERS = _merge_markers(TOOL_CALL_START, TOOL_CALL_START_ALTS)
TOOL_CALL_END_MARKERS = _merge_markers(TOOL_CALL_END, TOOL_CALL_END_ALTS)

# ===================== Orchestration gates =====================
# Hard ceiling on tool rounds. This is a runaway guard, not a UX limit: a
# value this high only ever fires on a genuinely stuck loop.
MAX_TOOL_ROUNDS_HARD = _int_env("MAX_TOOL_ROUNDS_HARD", 500, 1)
# Upper bound on calls accepted from a single assistant turn.
MAX_TOOL_CALLS_PER_TURN = _int_env("MAX_TOOL_CALLS_PER_TURN", 8, 1)
# Opt-in support for legacy/looser output shapes.
ALLOW_LEGACY_TOOL_CALL_FORMATS = os.environ.get(
    "ALLOW_LEGACY_TOOL_CALL_FORMATS", "false"
).strip().lower() in {"1", "true", "yes", "on"}


# ===================== Logging =====================
class _Color:
    RESET = "\033[0m"
    GREY = "\033[90m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"


class _ColorFormatter(logging.Formatter):
    """Level-coloured formatter, used on the console only."""

    _MAP = {
        logging.DEBUG: _Color.GREY,
        logging.INFO: _Color.CYAN,
        logging.WARNING: _Color.YELLOW,
        logging.ERROR: _Color.RED,
        logging.CRITICAL: _Color.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._MAP.get(record.levelno, _Color.GREY)
        record.module = record.name
        return (
            f"{color}{self.formatTime(record, '%H:%M:%S')}{_Color.RESET} "
            f"{color}[{record.levelname:<5}]{_Color.RESET} "
            f"{self._format_msg(record)}"
        )

    @staticmethod
    def _format_msg(record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.exc_info:
            import traceback
            msg += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return msg


def _build_logger() -> logging.Logger:
    """Console (coloured only on a TTY) + optional rotating file handler.

    Non-TTY targets (systemd, supervisor, redirected output) disable ANSI so
    escape codes do not end up rendered as literal text in log files.
    FORCE_COLOR=1 / NO_COLOR=1 override the TTY detection.
    """
    lg = logging.getLogger("harness_kernel")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    lg.propagate = False

    import sys
    no_color = os.environ.get("NO_COLOR", "").lower() in {"1", "true", "yes", "on"}
    force_color = os.environ.get("FORCE_COLOR", "").lower() in {"1", "true", "yes", "on"}
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    use_color = force_color or (is_tty and not no_color)

    ch = logging.StreamHandler()
    if use_color:
        ch.setFormatter(_ColorFormatter())
    else:
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    lg.addHandler(ch)

    # Optional file sink. Absent by default so an embedded kernel never writes
    # unexpected files into the host's working directory.
    log_dir = os.environ.get("HARNESS_LOG_DIR", "").strip()
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            fh = RotatingFileHandler(
                os.path.join(log_dir, "harness.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            lg.addHandler(fh)
        except OSError:
            pass
    return lg


log = _build_logger()


def get_logger(name: str = "") -> logging.Logger:
    """Return a namespaced child logger that funnels into the shared sink."""
    return log.getChild(name) if name else log
