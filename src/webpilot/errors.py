"""Exception hierarchy - the only exception types crossing module boundaries."""

from __future__ import annotations


class WebpilotError(Exception):
    """Base class for every error webpilot raises deliberately."""


class ConfigError(WebpilotError):
    """Bad or missing configuration (including missing credentials)."""


class LLMError(WebpilotError):
    """Any provider-side failure: HTTP error, timeout, malformed response."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class BrowserError(WebpilotError):
    """Browser launch/navigation/action failure that is not a plain tool failure."""


class ActionError(WebpilotError):
    """A single tool action failed in a recoverable way."""


class SecurityDenied(WebpilotError):
    """The human or the policy refused an action."""


class BudgetExceeded(WebpilotError):
    """The run ran out of steps, time or tokens."""
