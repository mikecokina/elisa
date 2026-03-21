from __future__ import annotations

import json
from pathlib import Path

SCHEMA_SEARCH_DIRS: list[Path] = [Path(__file__).resolve().parent / "schema_files"]


class Registry:
    """Schema registry for loading and composing JSON schemas.

    The registry searches schema files in configured directories and composes
    base schemas by injecting shared subschemas (currently spots and pulsations)
    into selected components.

    Schema files are expected to have the ``.sc`` extension.
    """

    @classmethod
    def get_schema(cls, fname: str) -> dict:
        """Return the composed schema for the given topic.

        Loads the base schema for *fname* and injects shared subschemas for
        ``spots`` and ``pulsations`` into supported components where applicable.

        :param fname: Schema topic name (filename stem without extension).
        :returns: Parsed and composed schema as a dictionary.
        :raises LookupError: If the base schema cannot be located.
        :raises json.JSONDecodeError: If any schema file contains invalid JSON.
        """
        schema_path = cls._get_schema_path(fname)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        spot_schema_path = cls._get_schema_path("spot")
        spot_schema = json.loads(spot_schema_path.read_text(encoding="utf-8"))

        mode_schema_path = cls._get_schema_path("pulsation")
        mode_schema = json.loads(mode_schema_path.read_text(encoding="utf-8"))

        # Adding subschemas for spots and pulsations to the base system schemas.
        feature_enabled = ["star", "primary", "secondary"]
        for component in feature_enabled:
            if component in schema.get("properties", {}):
                component_schema = schema["properties"][component]
                properties = component_schema.get("properties", {})

                if "spots" in properties:
                    properties["spots"] = spot_schema

                if "pulsations" in properties:
                    properties["pulsations"] = mode_schema

        return schema

    @staticmethod
    def _get_schema_path(fname: str) -> Path:
        """Resolve schema file path for *fname* in the configured search directories.

        :param fname: Schema topic name (filename stem without extension).
        :returns: Path to the schema file.
        :raises LookupError: If the schema cannot be found in any search directory.
        """
        for base_path in SCHEMA_SEARCH_DIRS:
            path = base_path / f"{fname}.sc"
            if path.is_file():
                return path

        msg = f"No schema found for fname: {fname}."
        raise LookupError(msg)
