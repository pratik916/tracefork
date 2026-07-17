"""Framework adapters: bind a framework's LLM client to tracefork's transport
seam and overlay a neutral step-DAG.

The byte capture stays at the httpx transport (``transport.py``); a framework's
callbacks/tracing are an observer-only annotation layer feeding ``StepDAG`` (see
``base.py``). Importing this package registers the built-in LangChain/LangGraph
(``"langchain"``), OpenAI Agents SDK (``"openai_agents"``), CrewAI
(``"crewai"``), AutoGen (``"autogen"``), Google ADK (``"adk"``), and Shepherd
(``"shepherd"``) adapters — every framework import they make is guarded, so
this import never requires any of those packages to be installed. Shepherd is
the one exception with nothing to guard: it is an unpublished codebase,
not a published package (see ``adapters/shepherd.py``'s module docstring).
"""

from __future__ import annotations

from . import adk as _adk  # noqa: F401  (side effect: registers "adk")
from . import autogen as _autogen  # noqa: F401  (side effect: registers "autogen")
from . import crewai as _crewai  # noqa: F401  (side effect: registers "crewai")
from . import langchain as _langchain  # noqa: F401  (side effect: registers "langchain")
from . import (
    openai_agents as _openai_agents,  # noqa: F401  (side effect: registers "openai_agents")
)
from . import shepherd as _shepherd  # noqa: F401  (side effect: registers "shepherd")
from .adk import (
    ADK_IMPORT_HINT,
    AdkAdapter,
    TraceforkAdkCore,
    adk_available,
    make_plugin,
    require_adk,
)
from .autogen import (
    AUTOGEN_IMPORT_HINT,
    AutoGenAdapter,
    TraceforkInterventionCore,
    autogen_available,
    make_intervention_handler,
    require_autogen,
)
from .base import (
    BaseFrameworkAdapter,
    BindResult,
    FrameworkAdapter,
    Step,
    StepDAG,
    UuidPatch,
    build_http_clients,
    get_framework_adapter,
    load_adapter_entry_points,
    register_framework_adapter,
    registered_framework_adapters,
)
from .crewai import (
    CREWAI_IMPORT_HINT,
    CrewAIAdapter,
    TraceforkCrewEventCore,
    crewai_available,
    make_event_listener,
    require_crewai,
)
from .langchain import (
    CheckpointRecord,
    LangChainAdapter,
    TapeBackedCheckpointStore,
    TraceforkCallbackCore,
    langchain_available,
    langgraph_available,
    make_callback_handler,
    make_tape_backed_checkpointer,
    require_langchain,
    require_langgraph,
)
from .openai_agents import (
    OPENAI_AGENTS_IMPORT_HINT,
    OpenAIAgentsAdapter,
    TraceforkTracingCore,
    bind_default_client,
    make_tracing_processor,
    openai_agents_available,
    require_openai_agents,
)
from .shepherd import ShepherdAdapter, TraceforkShepherdCore

__all__ = [
    # base seam
    "Step",
    "StepDAG",
    "BindResult",
    "FrameworkAdapter",
    "BaseFrameworkAdapter",
    "UuidPatch",
    "build_http_clients",
    "register_framework_adapter",
    "get_framework_adapter",
    "registered_framework_adapters",
    "load_adapter_entry_points",
    # langchain / langgraph
    "LangChainAdapter",
    "TraceforkCallbackCore",
    "TapeBackedCheckpointStore",
    "CheckpointRecord",
    "make_callback_handler",
    "make_tape_backed_checkpointer",
    "langchain_available",
    "require_langchain",
    "langgraph_available",
    "require_langgraph",
    # openai agents sdk
    "OPENAI_AGENTS_IMPORT_HINT",
    "OpenAIAgentsAdapter",
    "TraceforkTracingCore",
    "bind_default_client",
    "make_tracing_processor",
    "openai_agents_available",
    "require_openai_agents",
    # crewai
    "CREWAI_IMPORT_HINT",
    "CrewAIAdapter",
    "TraceforkCrewEventCore",
    "crewai_available",
    "make_event_listener",
    "require_crewai",
    # autogen
    "AUTOGEN_IMPORT_HINT",
    "AutoGenAdapter",
    "TraceforkInterventionCore",
    "autogen_available",
    "make_intervention_handler",
    "require_autogen",
    # google adk
    "ADK_IMPORT_HINT",
    "AdkAdapter",
    "TraceforkAdkCore",
    "adk_available",
    "make_plugin",
    "require_adk",
    # shepherd (openai-path only, synthetic-double-validated - see module docstring)
    "ShepherdAdapter",
    "TraceforkShepherdCore",
]
