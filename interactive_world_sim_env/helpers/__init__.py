"""Isolated, exploratory helpers for the WorldModelEnv wrapper.

Modules in this subpackage have a one-way dependency on the rest of
the wrapper (helper -> env, never env -> helper). They are not part
of the public API surface of `interactive_world_sim_env` and must not
be imported from the top-level `__init__.py`.
"""
