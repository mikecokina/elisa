from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class Buffer:
    """Singleton storage for tables loaded from external resources.

    This class provides shared in-memory buffers for frequently accessed data
    such as limb darkening coefficient tables and atmosphere tables. The
    buffers are size-limited to prevent uncontrolled memory growth.

    Instances of this class share the same state.
    """

    _instance: ClassVar[Buffer | None] = None

    MAX_STORAGE: ClassVar[int] = 300

    LD_CFS_TABLES: ClassVar[dict] = {}
    ATMOSPHERE_TABLES: ClassVar[dict] = {}

    def __new__(cls) -> Buffer:  # noqa: PYI034
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.DEFAULT_SETTINGS = cls.settings_serializer()
        return cls._instance

    @classmethod
    def settings_serializer(cls) -> dict:
        """Return a snapshot of buffer configuration and storages.

        :returns: Dictionary containing buffer storages and limits.
        """
        return {
            "LD_CFS_TABLES": cls.LD_CFS_TABLES,
            "ATMOSPHERE_TABLES": cls.ATMOSPHERE_TABLES,
            "MAX_STORAGE": cls.MAX_STORAGE,
        }

    @classmethod
    def reduce_buffer(cls, storage: MutableMapping) -> MutableMapping:
        """Trim buffer if it exceeds the configured maximum size.

        Oldest inserted items are removed first to maintain the size limit.

        :param storage: Buffer storage mapping.
        :returns: Reduced storage mapping.
        """
        if len(storage) > cls.MAX_STORAGE:
            start_idx = len(storage) - cls.MAX_STORAGE
            for key in list(storage.keys())[:start_idx]:
                del storage[key]
        return storage


buffer = Buffer()
