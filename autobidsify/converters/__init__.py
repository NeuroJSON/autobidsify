"""BIDS Converters"""

# execute + validate have no LLM dependency, so they are imported eagerly.
from autobidsify.converters.executor import execute_bids_plan
from autobidsify.converters.validators import validate_bids_compatible

__all__ = [
    "build_bids_plan",
    "execute_bids_plan",
    "validate_bids_compatible",
]


def __getattr__(name):
    """Lazily import planner only when build_bids_plan is actually requested.

    planner pulls in autobidsify.llm (openai/requests), which the execute-only
    consumers (e.g. the ExecVal desktop app) must not be forced to load just to
    import execute_bids_plan. Keeping planner out of the eager import chain lets
    those consumers avoid the LLM client dependencies entirely, while callers
    that need planning (`from autobidsify.converters import build_bids_plan`)
    still work transparently.
    """
    if name == "build_bids_plan":
        from autobidsify.converters.planner import build_bids_plan
        return build_bids_plan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
