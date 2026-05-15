from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from wfp_access.artifacts.spec import ArtifactSpec, CachePolicy
from wfp_access.data.schemas import Schema

F = TypeVar("F", bound=Callable)


def artifact(
    *,
    name: str,
    schema: Schema | None = None,
    format_role: str = "table",
    cache_policy: CachePolicy = CachePolicy.CONTENT_ADDRESSED,
) -> Callable[[F], F]:
    """Attach artifact metadata while preserving direct function calls."""

    spec = ArtifactSpec(name=name, schema=schema, format_role=format_role, cache_policy=cache_policy)

    def decorate(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(wrapper, "__artifact_spec__", spec)
        return wrapper  # type: ignore[return-value]

    return decorate


def source_artifact(**kwargs):
    return artifact(cache_policy=CachePolicy.READ_ONLY, **kwargs)


def derived_artifact(**kwargs):
    return artifact(cache_policy=CachePolicy.CONTENT_ADDRESSED, **kwargs)


def report_artifact(**kwargs):
    return artifact(cache_policy=CachePolicy.NEVER, **kwargs)
