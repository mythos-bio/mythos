"""Scheduler hints for simulators and objectives.

This module provides the SchedulerHints dataclass and SchedulerUnit mixin
for specifying resource requirements and scheduling options that can be
translated to various execution engines (Ray, Dask, local, etc.).
"""

from dataclasses import field
from typing import Any

import chex


@chex.dataclass(frozen=True, kw_only=True)
class SchedulerHints:
    """Engine-agnostic scheduling hints for simulators and objectives.

    These hints describe resource requirements and scheduling preferences
    that can be translated to engine-specific options at runtime.

    Attributes:
        num_cpus: Number of CPUs required. None means unspecified (engine default).
        num_gpus: Number of GPUs required. Fractional values allowed for GPU sharing.
        mem_mb: Memory required in megabytes.
        max_retries: Maximum number of retries on failure.
        custom: Engine-specific options, structured as {"engine_name": {"option": value}}.
            For example: {"ray": {"scheduling_strategy": "SPREAD"}}.
    """

    num_cpus: int | None = None
    num_gpus: float | None = None
    mem_mb: int | None = None
    max_retries: int | None = None
    custom: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self, engine: str, rewrite_options: dict[str, str]|None = None) -> dict[str, Any]:
        """Convert SchedulerHints to a dictionary.

        Args:
            engine: The target execution engine (e.g., "ray", "dask").
            rewrite_options: Optional str->str mapping to rename standard
                options for the target engine.
        """
        rewrite_options = rewrite_options or {}
        def translate(name: str) -> str:
            return rewrite_options.get(name, name)
        option_dict = {
            translate(option): value
            for option, value in self.items()
            if value is not None and option != "custom"
        }
        return {
            **option_dict,
            **self.custom.get(engine, {}),
        }


@chex.dataclass(frozen=True, kw_only=True)
class SchedulerUnit:
    """Mixin for classes that support scheduler hints.

    This mixin provides helper methods for accessing scheduler hints.
    Classes using this mixin must declare a `scheduler_hints` field:

        scheduler_hints: SchedulerHints | None = None

    Example:
        @chex.dataclass(frozen=True, kw_only=True)
        class MySimulator(Simulator, SchedulerUnit):
            scheduler_hints: SchedulerHints | None = None

            # ... rest of simulator implementation
    """

    scheduler_hints: SchedulerHints | None = None
