from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class SupportsPipelineStage(Protocol):
    """
    Minimal interface describing what the viewer needs to know about a
    pipeline stage.
    """

    name: str
    params: Dict[str, object]


@dataclass
class PipelineStageInfo:
    """
    Lightweight, view-friendly representation of a pipeline stage.

    This type is intentionally small so it can be shared between controller and
    viewer without exposing controller internals.
    """

    name: str
    params: Dict[str, object] = field(default_factory=dict)

