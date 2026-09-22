"""Phonetic similarity search over a list of words or spans."""

from importlib.metadata import PackageNotFoundError, version

from simphone.g2p import normalize
from simphone.index import PhoneticIndex

try:
    __version__ = version("simphone")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["PhoneticIndex", "normalize"]
