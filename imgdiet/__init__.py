# imgdiet/__init__.py

import importlib.metadata

from .core import save

try:
    __version__ = importlib.metadata.version("imgdiet")
except importlib.metadata.PackageNotFoundError:
    # Fallback if the package is not installed or in dev mode
    __version__ = "0.0.0"

__all__ = [
    "save",
    "__version__"
]
