"""tracefork — time-travel debugger for AI agents."""

from .boundary_guard import BoundaryGuard, BoundaryViolationError
from .config import RedactionPolicy, TraceforkConfig
from .mcp_client import RecordingMCPSession, mcp_available, require_mcp
from .record_mode import RecordMode
from .recorder import AsyncRecorder, Recorder
from .redact import Redactor, safe_defaults, with_content_redaction
from .tape import Tape
from .tools import (
    NativeToolSeam,
    ToolForkTransport,
    ToolTransport,
    make_result_frame,
    make_tool_call_frame,
)

__all__ = [
    "Recorder",
    "AsyncRecorder",
    "Tape",
    "Redactor",
    "safe_defaults",
    "with_content_redaction",
    "TraceforkConfig",
    "RedactionPolicy",
    "RecordMode",
    "BoundaryGuard",
    "BoundaryViolationError",
    "ToolTransport",
    "ToolForkTransport",
    "NativeToolSeam",
    "make_tool_call_frame",
    "make_result_frame",
    "RecordingMCPSession",
    "mcp_available",
    "require_mcp",
]
