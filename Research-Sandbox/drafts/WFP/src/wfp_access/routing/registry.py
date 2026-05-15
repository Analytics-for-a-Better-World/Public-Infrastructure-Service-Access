from __future__ import annotations

from wfp_access.core.registry import Registry
from wfp_access.routing.base import RouterStrategy


class RouterRegistry(Registry[RouterStrategy]):
    """Registry for routing strategy classes."""
