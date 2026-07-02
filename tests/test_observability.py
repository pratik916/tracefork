"""Observability tests — structlog JSON pipeline + opt-in OTel self-instrumentation.

Both `structlog` and `opentelemetry-*` are optional (the `observability`
extra); these tests exercise the availability guard both ways (mirroring
`test_mcp_client.py`'s pattern for the `mcp` extra) and prove `traced_span`/
`@instrument` never change behavior when the opt-in is off (the default) —
byte-for-byte identical to before this module existed, no install required.
"""

import logging
import os

import pytest

from tracefork.observability import (
    OTEL_ENABLED_ENV,
    _NoopSpan,
    enable_otel_instrumentation,
    get_logger,
    instrument,
    otel_available,
    require_otel,
    require_structlog,
    reset_otel_instrumentation_override,
    structlog_available,
    traced_span,
)


@pytest.fixture(autouse=True)
def _reset_instrumentation_state():
    """Every test starts from the disabled-by-default state."""
    reset_otel_instrumentation_override()
    os.environ.pop(OTEL_ENABLED_ENV, None)
    yield
    reset_otel_instrumentation_override()
    os.environ.pop(OTEL_ENABLED_ENV, None)


# ── availability guards ──────────────────────────────────────────────────────


def test_require_otel_matches_availability():
    if otel_available():
        require_otel()  # no raise when installed
    else:
        with pytest.raises(ImportError, match="observability"):
            require_otel()


def test_require_structlog_matches_availability():
    if structlog_available():
        require_structlog()  # no raise when installed
    else:
        with pytest.raises(ImportError, match="observability"):
            require_structlog()


def test_get_logger_always_returns_something_usable():
    """Whether or not structlog is installed, get_logger must never raise, and
    the returned object must support an `.info(...)` call."""
    logger = get_logger("tracefork.test")
    if structlog_available():
        logger.info("hello", key="value")
    else:
        assert isinstance(logger, logging.Logger)
        logger.info("hello")


# ── traced_span: no-op by default ───────────────────────────────────────────


def test_traced_span_is_noop_when_disabled():
    with traced_span("tracefork.test.span", foo="bar") as span:
        assert isinstance(span, _NoopSpan)
        span.set_attribute("k", "v")  # must not raise


def test_traced_span_is_noop_when_enabled_but_otel_not_installed():
    """Enabling instrumentation without the SDK installed must still be safe —
    "opt-in" never means "crashes if you forgot to install the extra."""
    enable_otel_instrumentation(True)
    if otel_available():
        pytest.skip("opentelemetry is installed in this environment")
    with traced_span("tracefork.test.span") as span:
        assert isinstance(span, _NoopSpan)


def test_env_var_enables_instrumentation_flag_without_explicit_call():
    os.environ[OTEL_ENABLED_ENV] = "1"
    if otel_available():
        pytest.skip("opentelemetry is installed in this environment")
    # Still safe/no-op without the SDK, whether enabled via env var or call.
    with traced_span("tracefork.test.span") as span:
        assert isinstance(span, _NoopSpan)


def test_explicit_override_wins_over_env_var():
    os.environ[OTEL_ENABLED_ENV] = "1"
    enable_otel_instrumentation(False)
    with traced_span("tracefork.test.span") as span:
        assert isinstance(span, _NoopSpan)


# ── @instrument decorator ────────────────────────────────────────────────────


def test_instrument_decorator_passes_through_return_value_and_identity():
    @instrument("tracefork.test.fn")
    def add(a: int, b: int) -> int:
        return a + b

    assert add.__name__ == "add"  # functools.wraps preserved
    assert add(2, 3) == 5


def test_instrument_decorator_propagates_exceptions():
    @instrument("tracefork.test.fn")
    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        boom()
