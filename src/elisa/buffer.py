class Buffer(object):
    """
    Buffers for parameters loaded from external files in singleton format
    """
    _instance = None

    MAX_STORAGE = 300

    LD_CFS_TABLES = dict()
    ATMOSPHERE_TABLES = dict()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Buffer, cls).__new__(cls)
            cls.DEFAULT_SETTINGS = cls.settings_serializer()
        return cls._instance

    @classmethod
    def settings_serializer(cls):
        return {
            "LD_CFS_TABLES": cls.LD_CFS_TABLES,
            "ATMOSPHERE_TABLES": cls.ATMOSPHERE_TABLES,
            "MAX_STORAGE": cls.MAX_STORAGE
        }

    @classmethod
    def reduce_buffer(cls, storage):
        """
        If buffer exceeds allowed size, the first items are deleted.

        :param storage: Dict; buffer
        :return: Dict; reduced buffer
        """
        if len(storage) > cls.MAX_STORAGE:
            for key in storage.keys()[:len(storage)-cls.MAX_STORAGE]:
                del storage[key]
        return storage


buffer = Buffer()
