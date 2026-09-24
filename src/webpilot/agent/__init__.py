"""The webpilot agent core: tools, prompts, context management, sub-agents, loop.

Only ``webpilot.types``, ``webpilot.config``, ``webpilot.errors`` and
``webpilot.tokenizer`` are imported from outside this package, so the agent can be
tested (and reasoned about) without a browser, a network or an API key.
"""

from __future__ import annotations

from . import prompts, tools
from .loop import Agent
from .memory import ContextManager
from .subagents import (
    ExtractorSubAgent,
    RecoverySubAgent,
    Summarizer,
    deterministic_summary,
    extract_json_object,
    parse_tool_calls,
)
from .tools import (
    TOOL_NAMES,
    TOOL_SPECS,
    DispatchContext,
    ToolRegistry,
    fallback_render_page,
    validate_args,
)

__all__ = [
    "Agent",
    "ContextManager",
    "DispatchContext",
    "ExtractorSubAgent",
    "RecoverySubAgent",
    "Summarizer",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "ToolRegistry",
    "deterministic_summary",
    "extract_json_object",
    "fallback_render_page",
    "parse_tool_calls",
    "prompts",
    "tools",
    "validate_args",
]
