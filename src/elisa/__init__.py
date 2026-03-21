from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elisa import units  # re-export

# prepare units as u for simpler import
u = units

if TYPE_CHECKING:
    # These imports are evaluated only by static type checkers and IDEs
    # (for example: mypy, Pyright, and language server tooling). We intentionally
    # avoid importing these submodules at runtime to prevent two problems:
    #
    # 1. Import-time cost and side effects: some submodules perform heavy work
    #    (I/O, large dependency imports, or interactive prompts). Importing them
    #    in `__init__` would increase startup time and may trigger unwanted
    #    side effects during simple imports (for example in unit tests).
    #
    # 2. Circular imports: many submodules reference package-level names such
    #    as `settings`. Importing everything eagerly at package-import time can
    #    create import cycles where a submodule tries to import `elisa` while
    #    `elisa.__init__` is still initialising. Keeping these imports inside
    #    TYPE_CHECKING prevents those runtime cycles.
    #
    # Additional benefits:
    # - IDEs and static analyzers can resolve the exported symbols (listed in
    #   `__all__`) for autocompletion, go-to-definition, and type checking.
    # - Sphinx and other documentation tools that run static analysis will be
    #   able to find and document these names without executing package code.
    #
    # Maintenance guidance:
    # - Keep the names here in sync with the package public API (the
    #   `_PUBLIC_OBJECTS` map and `__all__`) so tools can correctly resolve
    #   symbols.
    # - Do not move heavy runtime imports into this block; it is for static
    #   use only. For runtime access, use the package's lazy-loading helpers
    #   (see `__getattr__` and `_make_download_manager_instance`).
    #
    # Use `# noqa: F401` on imports that are only present for type-checking or
    # IDE resolution to avoid linter warnings about unused imports.
    from elisa.analytics.dataset.base import LCData, RVData  # noqa: F401
    from elisa.base.star import Star  # noqa: F401
    from elisa.binary_system.system import BinarySystem
    from elisa.conf.settings import settings  # noqa: F401
    from elisa.managers.download_manager import DownloadManager  # noqa: F401
    from elisa.observer.observer import Observer
    from elisa.single_system.system import SingleSystem  # noqa: F401

__version__ = "0.7.0.dev0"

# Map of public names to (module, attribute) for lazy import
_PUBLIC_OBJECTS = {
    "BinarySystem": ("binary_system.system", "BinarySystem"),
    "Observer": ("observer.observer", "Observer"),
    "Star": ("base.star", "Star"),
    "SingleSystem": ("single_system.system", "SingleSystem"),
    "LCData": ("analytics.dataset.base", "LCData"),
    "RVData": ("analytics.dataset.base", "RVData"),
    "settings": ("conf.settings", "settings"),
    "download_manager": ("managers.download_manager", "DownloadManager"),
}

__all__ = tuple(sorted([
    "__version__",
    "BinarySystem",
    "LCData",
    "Observer",
    "RVData",
    "SingleSystem",
    "Star",
    "download_manager",
    "get_default_binary",
    "get_default_observer",
    "get_bolometric_default_observer",
    "settings",
    "u",
]))


def get_default_binary_definition() -> dict:
    """Return the default binary-system JSON parsed from package data.

    :returns: A dictionary representation of the default binary system.
    :rtype: dict
    """
    data_path = Path(__file__).resolve().parent / "data" / "default_binary_system.json"
    return json.loads(data_path.read_text())


def get_default_binary() -> BinarySystem:
    """Create a BinarySystem instance from the package default definition.

    The import is performed lazily to avoid import-time cycles.

    :returns: BinarySystem instance created from package data.
    :rtype: BinarySystem
    """
    from elisa.binary_system.system import BinarySystem  # noqa: PLC0415

    return BinarySystem.from_json(data=get_default_binary_definition())


def get_default_observer() -> Observer:
    """Create an Observer configured with default passbands and the default binary.

    The import is performed lazily.

    :returns: Observer instance.
    :rtype: Observer
    """
    from elisa.observer.observer import Observer  # noqa: PLC0415

    return Observer(
        passband=["Generic.Bessell.U", "Generic.Bessell.V", "Generic.Bessell.R"],
        system=get_default_binary(),
    )


def get_bolometric_default_observer() -> Observer:
    """Create an Observer configured with the bolometric passband.

    :returns: Observer instance using bolometric passband.
    :rtype: Observer
    """
    from elisa.observer.observer import Observer  # noqa: PLC0415

    return Observer(passband=["bolometric"], system=get_default_binary())


def _make_download_manager_instance() -> Any:
    """Create and return a DownloadManager instance using current settings.

    This is lazily executed to avoid importing settings at package import time.
    """
    settings_mod = importlib.import_module(__name__ + ".conf.settings")
    settings_obj = settings_mod.settings
    dm_mod = importlib.import_module(__name__ + ".managers.download_manager")
    download_manager_cls = dm_mod.DownloadManager
    return download_manager_cls(settings_obj)


def __getattr__(name: str) -> Any:
    """Lazy attribute loader for package-level re-exports.

    Imports the requested object from the corresponding submodule on first access
    and caches it in the package globals to avoid repeated imports.
    """
    if name in _PUBLIC_OBJECTS:
        module_name, attr_name = _PUBLIC_OBJECTS[name]
        if name == "download_manager":
            val = _make_download_manager_instance()
            globals()[name] = val
            return val

        module = importlib.import_module(__name__ + "." + module_name)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return available attributes for the package, including lazy exports."""
    return sorted(list(globals().keys()) + list(_PUBLIC_OBJECTS.keys()))


# first time user
_settings_mod = importlib.import_module(__name__ + ".conf.settings")
_settings = _settings_mod.settings
if _settings.FIRST_TIME_USER:
    download = input(
        "Download manager will pull atmospheres and limb darkening tables.\n"
        "Do you want to proceed? [y/N]: ",
    )
    if download.lower() != "y":
        warn_msg = (
            "Please use download manager to pull atmospheres and limb darkening "
            "or do it manually as refered docs."
        )
        print(warn_msg)  # noqa: T201
        sys.exit(0)
    # The download path can trigger DeprecationWarnings from settings parsing
    # or downstream libraries; suppress DeprecationWarning here so the interactive
    # prompt remains clean for the user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _make_download_manager_instance().download_all()
