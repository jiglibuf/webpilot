"""Security layer: the human-in-the-loop gate for destructive actions.

``SecurityPolicy`` classifies every tool call by *meaning* (rule engine over the
element wording, the arguments and the page context) and decides whether it may
run: ``ask``/``allow``/``deny`` confirm modes, a domain allow-list, and an
append-only JSONL audit trail with credentials redacted.

    from webpilot.security import SecurityPolicy

    policy = SecurityPolicy(config, llm=llm if config.llm_risk_check else None)
    risk = policy.classify(call, ctx)
    approved, reason = policy.authorize(call, risk, ctx, ui.confirm)
"""

from .policy import (  # noqa: F401
    REDACTED,
    RULES,
    Rule,
    SecurityPolicy,
    Signals,
    default_rules,
    host_matches,
    looks_secret,
    redact_args,
    redact_secrets,
)

__all__ = [
    "REDACTED",
    "RULES",
    "Rule",
    "SecurityPolicy",
    "Signals",
    "default_rules",
    "host_matches",
    "looks_secret",
    "redact_args",
    "redact_secrets",
]
