"""tracefork — time-travel debugger for AI agents."""

from importlib import metadata as _metadata

from .adapters import (
    ADK_IMPORT_HINT,
    AdkAdapter,
    AutoGenAdapter,
    BaseFrameworkAdapter,
    BindResult,
    CrewAIAdapter,
    FrameworkAdapter,
    LangChainAdapter,
    OpenAIAgentsAdapter,
    Step,
    StepDAG,
    TapeBackedCheckpointStore,
    TraceforkAdkCore,
    TraceforkCallbackCore,
    TraceforkCrewEventCore,
    TraceforkInterventionCore,
    TraceforkTracingCore,
    adk_available,
    autogen_available,
    bind_default_client,
    crewai_available,
    get_framework_adapter,
    langchain_available,
    langgraph_available,
    make_callback_handler,
    make_event_listener,
    make_intervention_handler,
    make_plugin,
    make_tape_backed_checkpointer,
    make_tracing_processor,
    openai_agents_available,
    register_framework_adapter,
    registered_framework_adapters,
    require_adk,
    require_autogen,
    require_crewai,
    require_langchain,
    require_langgraph,
    require_openai_agents,
)

# ── core product API (1.0 SemVer surface) ───────────────────────────────────
# Every symbol below is what the *product* actually is — record/replay/fork/
# blame — as opposed to the framework-adapter/observability/redaction helpers
# above. Pre-1.0 none of this was top-level reachable (`Tape` was the lone
# exception); freezing an `__all__` that hid the product behind its own
# integrations was the gap this section closes. See `docs/plugin-api.md` and
# `docs/stability.md` for what SemVer covers here.
from .basis import RecordBasis, current_basis
from .blame import (
    BlameEngine,
    BlameReport,
    BudgetExceededError,
    BudgetGovernor,
    FlipRateResult,
    Oracle,
    ShapleyReport,
    StringMatchOracle,
)
from .boundary_guard import (
    BoundaryGuard,
    BoundaryViolationError,
    ConfinementSpec,
    ConfinementViolationError,
)
from .certificate import CertificateStrength, ReplayCertificate
from .checkpoint import CheckpointWriter, recover_checkpoint
from .config import RedactionPolicy, TraceforkConfig
from .fork import Branch, BranchSpec, ForkEngine
from .interop import (
    blame_report_from_json,
    build_openinference_dataset,
    build_otel_trace,
    ingest_openinference_dataset,
    ingest_otel_trace,
)
from .matcher import IdentityMatcher, RequestMatcher
from .mcp_client import RecordingMCPSession, mcp_available, require_mcp
from .nondet import DivergenceError, NondetSource, RecordingNondet, ReplayNondet
from .observability import (
    enable_otel_instrumentation,
    get_logger,
    otel_available,
    require_otel,
    require_structlog,
    structlog_available,
)
from .plugins import (
    ADAPTER_GROUP,
    MATCHER_GROUP,
    ORACLE_GROUP,
    PROVIDER_GROUP,
    SERIALIZER_GROUP,
    Registry,
)
from .providers import ProviderAdapter
from .record_mode import RecordMode
from .recorder import AsyncRecorder, Recorder
from .redact import Redactor, safe_defaults, with_content_redaction
from .replay import ReplayVerifier, VerificationResult
from .report import generate_report
from .session_chaos import session_chaos_release_orders, session_sibling_chaos_order
from .store import ForkPointDriftError, TapeConflictError, TapeStore
from .tape import Tape, TapeSerializer
from .tools import (
    NativeToolSeam,
    ToolForkTransport,
    ToolTransport,
    make_result_frame,
    make_tool_call_frame,
)
from .tournament import TournamentEngine
from .transport import AsyncTraceforkTransport, TraceforkTransport, chaos_release_order

try:
    __version__ = _metadata.version("tracefork")
except _metadata.PackageNotFoundError:  # pragma: no cover - not-installed edge case
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
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
    "CheckpointWriter",
    "recover_checkpoint",
    "ToolTransport",
    "ToolForkTransport",
    "NativeToolSeam",
    "make_tool_call_frame",
    "make_result_frame",
    "chaos_release_order",
    "session_chaos_release_orders",
    "session_sibling_chaos_order",
    "RecordingMCPSession",
    "mcp_available",
    "require_mcp",
    "build_otel_trace",
    "build_openinference_dataset",
    "ingest_otel_trace",
    "ingest_openinference_dataset",
    "blame_report_from_json",
    "otel_available",
    "require_otel",
    "structlog_available",
    "require_structlog",
    "enable_otel_instrumentation",
    "get_logger",
    "Step",
    "StepDAG",
    "BindResult",
    "FrameworkAdapter",
    "BaseFrameworkAdapter",
    "LangChainAdapter",
    "TraceforkCallbackCore",
    "TapeBackedCheckpointStore",
    "get_framework_adapter",
    "register_framework_adapter",
    "registered_framework_adapters",
    "make_callback_handler",
    "make_tape_backed_checkpointer",
    "langchain_available",
    "require_langchain",
    "langgraph_available",
    "require_langgraph",
    "OpenAIAgentsAdapter",
    "TraceforkTracingCore",
    "bind_default_client",
    "make_tracing_processor",
    "openai_agents_available",
    "require_openai_agents",
    "CrewAIAdapter",
    "TraceforkCrewEventCore",
    "crewai_available",
    "make_event_listener",
    "require_crewai",
    "AutoGenAdapter",
    "TraceforkInterventionCore",
    "autogen_available",
    "make_intervention_handler",
    "require_autogen",
    "ADK_IMPORT_HINT",
    "AdkAdapter",
    "TraceforkAdkCore",
    "adk_available",
    "make_plugin",
    "require_adk",
    # ── core product API ─────────────────────────────────────────────────
    "RecordBasis",
    "current_basis",
    "BlameEngine",
    "BlameReport",
    "BudgetExceededError",
    "BudgetGovernor",
    "FlipRateResult",
    "Oracle",
    "ShapleyReport",
    "StringMatchOracle",
    "ConfinementSpec",
    "ConfinementViolationError",
    "CertificateStrength",
    "ReplayCertificate",
    "Branch",
    "BranchSpec",
    "ForkEngine",
    "IdentityMatcher",
    "RequestMatcher",
    "DivergenceError",
    "NondetSource",
    "RecordingNondet",
    "ReplayNondet",
    "Registry",
    "PROVIDER_GROUP",
    "ORACLE_GROUP",
    "SERIALIZER_GROUP",
    "MATCHER_GROUP",
    "ADAPTER_GROUP",
    "ProviderAdapter",
    "ReplayVerifier",
    "VerificationResult",
    "generate_report",
    "TapeConflictError",
    "ForkPointDriftError",
    "TapeStore",
    "TapeSerializer",
    "TournamentEngine",
    "AsyncTraceforkTransport",
    "TraceforkTransport",
]
