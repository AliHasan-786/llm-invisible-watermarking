"""Generation pipelines with lazy exports for optional dataset dependencies."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate import CorpusGenerator

__all__ = ["CorpusGenerator"]


def __getattr__(name):
    if name == "CorpusGenerator":
        from .generate import CorpusGenerator

        return CorpusGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
