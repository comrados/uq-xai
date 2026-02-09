from pathlib import Path
import pickle
from typing import Any

from config.settings import DATA_CACHE_DIR


class Cache:
    """Universal pickle-based cache for arbitrary Python objects.

    Provides a simple key-value store backed by pickle files on disk.
    Useful for caching expensive computations like dataset loading,
    preprocessing, or feature extraction.

    Attributes:
        cache_dir: Directory path where cache files are stored.
    """

    def __init__(self, cache_dir: Path = DATA_CACHE_DIR):
        """Initialize cache with storage directory.

        Args:
            cache_dir: Path to directory for storing cache files.
                Default from config.
        """
        self.cache_dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        """Generate file path for a cache key.

        Args:
            key: Cache key identifying the object.

        Returns:
            Path to the pickle file for this key.
        """
        return self.cache_dir / f"{key}.pkl"

    def save(self, key: str, obj: Any) -> None:
        """Save an object to cache.

        Args:
            key: Unique identifier for the object.
            obj: Python object to cache (must be pickle-serializable).
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(self, key: str) -> Any:
        """Load an object from cache.

        Args:
            key: Unique identifier for the object.

        Returns:
            The cached Python object.

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        with open(self._path(key), "rb") as f:
            return pickle.load(f)

    def exists(self, key: str) -> bool:
        """Check if an object exists in cache.

        Args:
            key: Unique identifier for the object.

        Returns:
            True if the object is cached, False otherwise.
        """
        return self._path(key).exists()

    def load_or_create(self, key: str, create_fn) -> Any:
        """Load object from cache or create and cache it.

        Convenience method that checks cache first, and only calls create_fn
        if the object is not found. Newly created objects are automatically saved.

        Args:
            key: Unique identifier for the object.
            create_fn: Callable that creates the object if not cached.
                Should take no arguments and return the object to cache.

        Returns:
            The object (either loaded from cache or freshly created).
        """
        if self.exists(key):
            return self.load(key)
        obj = create_fn()
        self.save(key, obj)
        return obj