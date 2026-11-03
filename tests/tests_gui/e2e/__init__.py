"""Opt-in browser integration tests and their application manifest contracts."""

from pathlib import Path
from typing import TypedDict


class Component(TypedDict):
    """Serialized Gradio control metadata used to locate browser inputs."""

    selector: str
    type: str
    label: str | None
    value: object


class Callback(TypedDict):
    """Input IDs and fitting parameter fields of a registered callback."""

    fields: dict[str, int]
    inputs: list[int]


class Manifest(TypedDict):
    """Controls and fitting callbacks exported by the isolated server."""

    components: dict[str, Component]
    callbacks: dict[str, Callback]


class Application(TypedDict):
    """Running application fixture and its isolated artifact directory."""

    url: str
    work: Path
    manifest: Manifest
    log: Path
