from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from scalable_distances.data.schemas import Schema


class CachePolicy(StrEnum):
    NEVER = "never"
    REUSE = "reuse"
    CONTENT_ADDRESSED = "content_addressed"
    READ_ONLY = "read_only"


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    schema: Schema | None = None
    format_role: str = "table"
    cache_policy: CachePolicy = CachePolicy.CONTENT_ADDRESSED
    contract_version: str = "artifact.v1"
