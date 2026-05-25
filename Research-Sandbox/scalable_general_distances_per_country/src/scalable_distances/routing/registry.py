from __future__ import annotations

from scalable_distances.core.registry import Registry
from scalable_distances.routing.base import RouterStrategy


class RouterRegistry(Registry[RouterStrategy]):
    """Registry for routing strategy classes."""
