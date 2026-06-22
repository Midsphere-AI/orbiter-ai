"""Exo meta-package. Installs all Exo sub-packages."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exo-ai")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
