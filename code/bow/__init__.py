"""Buy or Wait? — AI-powered financial affordability agent.

The package is split so that every number that gets scored is produced by
deterministic code, and the LLM is confined to turning unstructured evidence
(messages, images) into a small, schema-checked set of patches.
"""

__all__ = ["loading", "fx", "recurrence", "forecast", "planner", "verify"]
