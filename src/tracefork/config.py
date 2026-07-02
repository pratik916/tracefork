"""``TraceforkConfig`` — typed operator knobs with ``TRACEFORK_*`` env overrides.

``TraceforkConfig()`` reproduces every current default exactly (zero behavior
change): no redaction, ``store.db``, a $5 blame budget, ``RecordMode.ONCE``,
plain-text CLI output. ``TraceforkConfig.from_env()`` layers ``TRACEFORK_*``
environment variables on top for operators who want to tune these without
touching code.

Nothing in tracefork calls ``from_env()`` implicitly except the handful of
CLI option defaults documented at their call sites (``cli.py``'s ``--store``
and ``--budget`` defaults) and ``Recorder``/``AsyncRecorder``'s optional
``config=`` parameter (only consulted when the caller passes one, and only to
fill in a redactor the caller didn't already supply explicitly) — every other
call site keeps constructing its own explicit values exactly as it did before
this module existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from .record_mode import RecordMode
from .redact import Redactor, safe_defaults, with_content_redaction

#: Default environment-variable prefix consulted by ``TraceforkConfig.from_env()``.
ENV_PREFIX = "TRACEFORK_"


class RedactionPolicy(StrEnum):
    """Which ``Redactor`` (if any) ``TraceforkConfig.build_redactor()`` builds.

    * ``NONE`` (default) — no ``Redactor``; byte-identical to tracefork's
      pre-redaction behavior (``Recorder(..., redactor=None)``).
    * ``METADATA`` — ``redact.safe_defaults()``: auth headers + known secret
      env values scrubbed; message content untouched, tape stays
      bit-exact-replayable.
    * ``CONTENT`` — ``redact.with_content_redaction()``: metadata redaction
      plus message CONTENT scrubbed (forensic-only; see ``redact.py``'s
      module docstring).
    """

    NONE = "none"
    METADATA = "metadata"
    CONTENT = "content"


class LogFormat(StrEnum):
    """CLI/log output shape. Only ``TEXT`` (today's ``typer.echo`` output) is
    wired up in this release; ``JSON`` is declared for future structured-log
    consumers and is not read anywhere yet."""

    TEXT = "text"
    JSON = "json"


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TraceforkConfig:
    """Operator-tunable knobs. Every field's default equals tracefork's
    current hardcoded behavior; only ``from_env()`` (or explicit construction
    with non-default arguments) changes anything.
    """

    db_path: str = "store.db"
    budget_usd: float = 5.0
    redaction_policy: RedactionPolicy = RedactionPolicy.NONE
    capture_message_content: bool | None = None
    record_mode: RecordMode = RecordMode.ONCE
    log_format: LogFormat = LogFormat.TEXT

    @classmethod
    def from_env(cls, *, prefix: str = ENV_PREFIX) -> TraceforkConfig:
        """Layer ``TRACEFORK_*`` environment variables over the defaults above.

        An unset (or empty) variable falls back to the hardcoded default, so
        ``from_env()`` in a clean environment equals ``TraceforkConfig()``
        exactly — this is what keeps existing behavior unchanged for anyone
        who hasn't set any of these variables.
        """
        defaults = cls()

        def _raw(name: str) -> str | None:
            val = os.environ.get(f"{prefix}{name}")
            return val if val else None

        capture_raw = _raw("CAPTURE_MESSAGE_CONTENT")
        budget_raw = _raw("BUDGET_USD")
        default_capture = defaults.capture_message_content
        return cls(
            db_path=_raw("DB_PATH") or defaults.db_path,
            budget_usd=float(budget_raw) if budget_raw is not None else defaults.budget_usd,
            redaction_policy=RedactionPolicy(_raw("REDACTION_POLICY") or defaults.redaction_policy),
            capture_message_content=(
                _truthy(capture_raw) if capture_raw is not None else default_capture
            ),
            record_mode=RecordMode(_raw("RECORD_MODE") or defaults.record_mode),
            log_format=LogFormat(_raw("LOG_FORMAT") or defaults.log_format),
        )

    def build_redactor(self) -> Redactor | None:
        """Build the ``Redactor`` (if any) implied by ``redaction_policy``.

        ``NONE`` -> ``None`` (today's default: ``Recorder(..., redactor=None)``,
        byte-identical). ``METADATA`` -> ``safe_defaults()``. ``CONTENT`` ->
        ``with_content_redaction(capture_message_content=...)``.
        """
        if self.redaction_policy is RedactionPolicy.NONE:
            return None
        if self.redaction_policy is RedactionPolicy.METADATA:
            return safe_defaults()
        return with_content_redaction(capture_message_content=self.capture_message_content)
