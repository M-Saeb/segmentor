from __future__ import annotations

from threading import Lock


class ObjectClusterIdManager:
    """Singleton allocator for globally unique object-cluster IDs."""

    _instance: "ObjectClusterIdManager | None" = None
    _instance_lock = Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._next_id = 1
                cls._instance._lock = Lock()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ObjectClusterIdManager":
        return cls()

    def reserve_next_ids(self, count: int) -> list[int]:
        """
        Reserve and return a contiguous ID block.

        Example:
            next_id=5, count=3 -> [5, 6, 7], then next_id becomes 8.
        """
        if count < 0:
            raise ValueError("count must be >= 0")
        if count == 0:
            return []

        with self._lock:
            start = self._next_id
            self._next_id += count
        return list(range(start, start + count))

    def reset(self, start: int = 1) -> None:
        if start < 1:
            raise ValueError("start must be >= 1")
        with self._lock:
            self._next_id = start
