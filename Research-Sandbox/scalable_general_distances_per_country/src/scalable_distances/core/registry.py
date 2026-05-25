from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Tiny named registry for pluggable strategies and codecs."""

    _items: dict[str, type[T]] = field(default_factory=dict)

    def register(self, name: str, item: type[T]) -> None:
        if name in self._items:
            raise ValueError(f"Registry item already exists: {name}")
        self._items[name] = item

    def create(self, name: str, *args, **kwargs) -> T:
        try:
            cls = self._items[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown registry item {name!r}. Known: {known}") from exc
        return cls(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
