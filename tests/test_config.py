"""`TraceforkConfig` tests: hardcoded defaults reproduce tracefork's current
behavior exactly, `TRACEFORK_*` env vars override them one field at a time,
and `build_redactor()` maps `RedactionPolicy` onto `redact.py`'s existing
primitives without introducing a new redaction code path.

Offline, zero API keys.
"""

from __future__ import annotations

from tracefork.config import ENV_PREFIX, LogFormat, RedactionPolicy, TraceforkConfig
from tracefork.record_mode import RecordMode
from tracefork.redact import Redactor

_ENV_VARS = (
    "DB_PATH",
    "BUDGET_USD",
    "REDACTION_POLICY",
    "CAPTURE_MESSAGE_CONTENT",
    "RECORD_MODE",
    "LOG_FORMAT",
)


def _clear_env(monkeypatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(f"{ENV_PREFIX}{name}", raising=False)


# ── defaults ══════════════════════════════════════════════════════════════


def test_defaults_reproduce_todays_behavior():
    cfg = TraceforkConfig()
    assert cfg.db_path == "store.db"
    assert cfg.budget_usd == 5.0
    assert cfg.redaction_policy is RedactionPolicy.NONE
    assert cfg.capture_message_content is None
    assert cfg.record_mode is RecordMode.ONCE
    assert cfg.log_format is LogFormat.TEXT
    assert cfg.build_redactor() is None


def test_config_is_frozen():
    cfg = TraceforkConfig()
    try:
        cfg.db_path = "other.db"  # type: ignore[misc]
        raised = False
    except Exception:  # noqa: BLE001 - dataclasses raise FrozenInstanceError
        raised = True
    assert raised


# ── from_env ══════════════════════════════════════════════════════════════


def test_from_env_with_no_env_vars_equals_defaults(monkeypatch):
    _clear_env(monkeypatch)
    assert TraceforkConfig.from_env() == TraceforkConfig()


def test_from_env_overrides_db_path(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_DB_PATH", "/tmp/custom.db")
    cfg = TraceforkConfig.from_env()
    assert cfg.db_path == "/tmp/custom.db"
    # everything else stays at default
    assert cfg.budget_usd == 5.0


def test_from_env_overrides_budget_usd(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_BUDGET_USD", "12.5")
    assert TraceforkConfig.from_env().budget_usd == 12.5


def test_from_env_overrides_redaction_policy(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_REDACTION_POLICY", "metadata")
    assert TraceforkConfig.from_env().redaction_policy is RedactionPolicy.METADATA


def test_from_env_overrides_record_mode(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_RECORD_MODE", "none")
    assert TraceforkConfig.from_env().record_mode is RecordMode.NONE


def test_from_env_overrides_log_format(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_LOG_FORMAT", "json")
    assert TraceforkConfig.from_env().log_format is LogFormat.JSON


def test_from_env_overrides_capture_message_content_true(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_CAPTURE_MESSAGE_CONTENT", "true")
    assert TraceforkConfig.from_env().capture_message_content is True


def test_from_env_overrides_capture_message_content_false(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRACEFORK_CAPTURE_MESSAGE_CONTENT", "false")
    assert TraceforkConfig.from_env().capture_message_content is False


def test_from_env_respects_custom_prefix(monkeypatch):
    monkeypatch.setenv("MYAPP_DB_PATH", "/custom/prefixed.db")
    cfg = TraceforkConfig.from_env(prefix="MYAPP_")
    assert cfg.db_path == "/custom/prefixed.db"


# ── build_redactor() ══════════════════════════════════════════════════════


def test_build_redactor_none_policy_returns_none():
    assert TraceforkConfig(redaction_policy=RedactionPolicy.NONE).build_redactor() is None


def test_build_redactor_metadata_policy_is_replay_safe_and_untouched_content():
    redactor = TraceforkConfig(redaction_policy=RedactionPolicy.METADATA).build_redactor()
    assert isinstance(redactor, Redactor)
    assert redactor.content_redacted is False


def test_build_redactor_content_policy_marks_content_redacted():
    cfg = TraceforkConfig(redaction_policy=RedactionPolicy.CONTENT, capture_message_content=False)
    redactor = cfg.build_redactor()
    assert isinstance(redactor, Redactor)
    assert redactor.content_redacted is True


def test_build_redactor_content_policy_capture_true_disables_redaction():
    """Mirrors `with_content_redaction`'s own contract: an explicit
    `capture_message_content=True` keeps full content and does not mark the
    tape forensic-only, even under the `CONTENT` policy."""
    cfg = TraceforkConfig(redaction_policy=RedactionPolicy.CONTENT, capture_message_content=True)
    redactor = cfg.build_redactor()
    assert isinstance(redactor, Redactor)
    assert redactor.content_redacted is False
