import os
import json


class Registry(object):
    SCHEMA_SEARCH_DIRS = [
        os.path.join(os.path.dirname(__file__), 'schema_files')
    ]

    @classmethod
    def get_schema(cls, fname):
        """
        Gets schema for given topic.

        :param fname: filename for schema
        :return: Dict; json object
        """
        schema_path = cls._get_schema_path(fname)
        with open(schema_path, "r") as f:
            return json.loads(f.read())

    @classmethod
    def _get_schema_path(cls, fname):
        for base_path in cls.SCHEMA_SEARCH_DIRS:

            path = os.path.join(base_path, f'{fname}.sc')
            if os.path.isfile(path):
                return path
        raise LookupError(f'No schema found for fname: {fname}.')
