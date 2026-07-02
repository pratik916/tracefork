"""Optional observability: structlog JSON logging + OTel self-instrumentation.

Both `structlog` and `opentelemetry-api`/`opentelemetry-sdk` are OPTIONAL
dependencies (the `observability` extra — see `pyproject.toml`). The offline,
$0 core (record, replay, fork, blame, validate) behaves identically whether or
not either package is installed: `import tracefork` and the full test suite
must never require them.

Two independent pieces:

* **structlog JSON pipeline** — `configure_structlog_json()` (opt-in call)
  switches structlog to render JSON lines; `get_logger()` always returns a
  usable logger (structlog if installed, else stdlib `logging`), so a caller
  never has to branch on which is active.
* **OTel self-instrumentation** — `traced_span()` / `@instrument(...)` wrap
  `record`/`replay`/`fork`/`blame`'s entry points with an OTel span *only*
  when both (a) `opentelemetry-api`/`-sdk` are installed and (b) instrumentation
  is explicitly enabled (`enable_otel_instrumentation()` or
  `TRACEFORK_OTEL_ENABLED=1`). Merely installing the package changes nothing —
  the same "nothing loads unless explicitly allowlisted" posture as
  `plugins.py`'s entry-point loader. When disabled (the default), `traced_span`
  yields a no-op stand-in, so every wrapped call site is byte-for-byte
  unchanged from before this module existed.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

#: Env var truthy values ("1", "true", "yes", "on", case-insensitive) enable
#: OTel self-instrumentation without an explicit `enable_otel_instrumentation()`
#: call. Unset (the default) means disabled.
OTEL_ENABLED_ENV = "TRACEFORK_OTEL_ENABLED"

STRUCTLOG_IMPORT_HINT = (
    "structlog JSON logging needs the optional 'observability' extra: "
    "pip install 'tracefork[observability]'"
)
OTEL_IMPORT_HINT = (
    "OTel self-instrumentation needs the optional 'observability' extra: "
    "pip install 'tracefork[observability]'"
)

_TRUTHY = {"1", "true", "yes", "on"}

# `None` (the default) defers to `OTEL_ENABLED_ENV`; an explicit call to
# `enable_otel_instrumentation()` always wins over the env var.
_otel_enabled_override: bool | None = None


def structlog_available() -> bool:
    """Whether the optional ``structlog`` package is importable."""
    try:
        import structlog  # noqa: F401
    except ImportError:
        return False
    return True


def require_structlog() -> None:
    """Raise a helpful ``ImportError`` if ``structlog`` is missing."""
    if not structlog_available():
        raise ImportError(STRUCTLOG_IMPORT_HINT)


def otel_available() -> bool:
    """Whether the optional ``opentelemetry`` API package is importable."""
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError:
        return False
    return True


def require_otel() -> None:
    """Raise a helpful ``ImportError`` if ``opentelemetry`` is missing."""
    if not otel_available():
        raise ImportError(OTEL_IMPORT_HINT)


def enable_otel_instrumentation(enabled: bool = True) -> None:
    """Explicit opt-in/opt-out for OTel self-instrumentation.

    Independent of ``TRACEFORK_OTEL_ENABLED`` — an explicit call always wins
    over the environment variable. Pass ``enabled=False`` to force it off
    again (mainly for tests that don't want to depend on process-wide state).
    """
    global _otel_enabled_override
    _otel_enabled_override = enabled


def reset_otel_instrumentation_override() -> None:
    """Clear the explicit override, reverting to ``TRACEFORK_OTEL_ENABLED``."""
    global _otel_enabled_override
    _otel_enabled_override = None


def _otel_instrumentation_enabled() -> bool:
    if _otel_enabled_override is not None:
        return _otel_enabled_override
    return os.environ.get(OTEL_ENABLED_ENV, "").strip().lower() in _TRUTHY


def configure_structlog_json(level: int = logging.INFO) -> None:
    """Configure ``structlog`` to render one JSON object per log line to stdout.

    Raises ``ImportError`` (with an install hint) if ``structlog`` isn't
    installed. Never called implicitly — a caller opts in explicitly, so
    default logging behavior (whatever the host application already does)
    is untouched unless this is called.
    """
    require_structlog()
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """A ``structlog`` logger if installed (JSON output once
    ``configure_structlog_json()`` has run), else a plain stdlib
    ``logging.Logger`` — callers never need to branch on which is active."""
    if structlog_available():
        import structlog

        return structlog.get_logger(name)
    return logging.getLogger(name)


class _NoopSpan:
    """Stand-in yielded by `traced_span` when instrumentation is disabled or
    the optional SDK isn't installed — same shape as an OTel span's common
    surface, but every call is a no-op."""

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Yield an OTel span when self-instrumentation is enabled *and* the
    optional SDK is installed; otherwise yield a no-op stand-in.

    Never raises for a missing install or a disabled opt-in — wrapping a call
    site with this is always safe to add without changing behavior for a
    caller who hasn't opted in (the default).
    """
    if _otel_instrumentation_enabled() and otel_available():
        from opentelemetry import trace

        tracer = trace.get_tracer("tracefork")
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield span
        return
    yield _NoopSpan()


_F = TypeVar("_F", bound=Callable[..., Any])


def instrument(span_name: str) -> Callable[[_F], _F]:
    """Decorator form of `traced_span`, for wrapping an existing function or
    method with a span covering its whole call — with zero changes to its
    body. Used to self-instrument `BlameEngine.rank`, `ForkEngine.fork`, and
    `ReplayVerifier.verify` (see those modules); a no-op wrapper when
    instrumentation is disabled or the optional SDK isn't installed.
    """

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with traced_span(span_name):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
