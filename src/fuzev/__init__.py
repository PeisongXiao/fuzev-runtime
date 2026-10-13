"""Engine-independent FuZev contracts and a shared decision head."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fuzev-runtime")
except PackageNotFoundError:
    __version__ = "unknown"
