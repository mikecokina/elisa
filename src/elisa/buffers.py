MAX_STORAGE = 300


LD_CFS_TABLES = dict()
ATMOSPHERE_TABLES = dict()


def reduce_buffer(storage):
    """
    If buffer exceeds allowed size, the first items are deleted.

    :param storage: dict; buffer
    :return: dict; reduced buffer
    """
    if len(storage) > MAX_STORAGE:
        for key in storage.keys()[:len(storage)-MAX_STORAGE]:
            print('HEEEEY')
            del storage[key]
    return storage
