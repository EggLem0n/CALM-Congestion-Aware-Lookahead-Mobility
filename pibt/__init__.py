"""Shared PIBT engine for online MAPF-style one-step coordination."""

__all__ = ["PIBTAgentState", "PIBTEngine", "PIBTStepResult"]


def __getattr__(name):
    if name in __all__:
        from .engine import PIBTAgentState, PIBTEngine, PIBTStepResult

        values = {
            "PIBTAgentState": PIBTAgentState,
            "PIBTEngine": PIBTEngine,
            "PIBTStepResult": PIBTStepResult,
        }
        return values[name]
    raise AttributeError(name)
