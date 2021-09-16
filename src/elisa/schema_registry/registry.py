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
            # return json.loads(f.read())
            schema = json.loads(f.read())

        spot_schema_path = cls._get_schema_path('spot')
        with open(spot_schema_path, "r") as f:
            spot_schema = json.loads(f.read())

        mode_schema_path = cls._get_schema_path('pulsation')
        with open(mode_schema_path, "r") as f:
            mode_schema = json.loads(f.read())

        # adding subschemas for spots and pulsations to the base system schemas
        feature_enabled = ['star', 'primary', 'secondary']
        for component in feature_enabled:
            if component in schema['properties']:
                component_schema = schema['properties'][component]
                if 'spots' in component_schema['properties']:
                    component_schema['properties']['spots'] = spot_schema

                if 'pulsations' in component_schema['properties']:
                    component_schema['properties']['pulsations'] = mode_schema

        return schema

    @classmethod
    def _get_schema_path(cls, fname):
        for base_path in cls.SCHEMA_SEARCH_DIRS:

            path = os.path.join(base_path, f'{fname}.sc')
            if os.path.isfile(path):
                return path
        raise LookupError(f'No schema found for fname: {fname}.')
