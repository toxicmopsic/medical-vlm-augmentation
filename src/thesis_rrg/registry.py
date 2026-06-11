from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        key = key.strip().lower()
        if key in self._items:
            raise KeyError(f"{self.name}: duplicate key '{key}'")
        self._items[key] = value

    def get(self, key: str) -> T:
        key = key.strip().lower()
        if key not in self._items:
            known = ", ".join(sorted(self._items))
            raise KeyError(f"{self.name}: unknown key '{key}'. Known: {known}")
        return self._items[key]

    def keys(self) -> Iterable[str]:
        return self._items.keys()


BuilderFn = Callable[..., object]
