"""Framework adapters: bind a framework's LLM client to tracefork's transport
seam and overlay a neutral step-DAG.

The byte capture stays at the httpx transport (``transport.py``); a framework's
callbacks/tracing are an observer-only annotation layer feeding ``StepDAG`` (see
``base.py``). Importing this package registers the built-in LangChain/LangGraph
adapter under ``"langchain"`` — its framework imports are guarded, so this import
never requires ``langchain``/``langgraph`` to be installed.
"""

from __future__ import annotations

from . import langchain as _langchain  # noqa: F401  (side effect: registers "langchain")
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
]
