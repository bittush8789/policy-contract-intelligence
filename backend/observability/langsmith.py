"""LangSmith Observability for Enterprise RAG Assistant.

Configures LangSmith tracing by propagating the required environment variables
before any LangChain components are initialized. Once active, every LangChain
call (LLM, retrieval, prompt, chain) is automatically traced to the configured
LangSmith project — no code-level instrumentation required.

Usage
-----
Call `setup_langsmith()` once at application startup (done in main.py lifespan).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def setup_langsmith(
    api_key: str = "",
    project: str = "enterprise-rag-assistant",
    endpoint: str = "https://api.smith.langchain.com",
    enabled: bool = False,
) -> bool:
    """Configure LangSmith tracing environment variables.

    LangChain reads these vars at import-time for each call, so they must be
    set before any chain, LLM, or retriever is invoked.

    Args:
        api_key: LangSmith API key (from smith.langchain.com).
        project:  Project name used to group traces in the LangSmith UI.
        endpoint: LangSmith API endpoint (default: public cloud).
        enabled:  Master toggle — if False, tracing is explicitly disabled.

    Returns:
        True if tracing was successfully enabled, False otherwise.
    """
    if not enabled or not api_key:
        # Ensure tracing is fully disabled so stale env vars don't activate it
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if not enabled:
            logger.info("[LangSmith] Tracing is DISABLED (set LANGCHAIN_TRACING_V2=true to enable).")
        else:
            logger.warning("[LangSmith] Tracing enabled but LANGCHAIN_API_KEY is missing — tracing disabled.")
        return False

    # Propagate all required vars so LangChain SDK picks them up automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_ENDPOINT"] = endpoint

    logger.info("[LangSmith] Tracing ENABLED")
    logger.info(f"[LangSmith] Project  : {project}")
    logger.info(f"[LangSmith] Endpoint : {endpoint}")
    logger.info("[LangSmith] Dashboard: https://smith.langchain.com")
    return True


def get_run_url() -> Optional[str]:
    """Return the LangSmith project dashboard URL if tracing is active."""
    project = os.environ.get("LANGCHAIN_PROJECT", "enterprise-rag-assistant")
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        return f"https://smith.langchain.com/projects/{project}"
    return None


def is_tracing_enabled() -> bool:
    """Check whether LangSmith tracing is currently active."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
