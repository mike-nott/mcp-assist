"""Fast Path module for simple commands without LLM."""

from .processor import FastPathProcessor, FastPathResult, is_fast_path_candidate
from .loader import KeywordLoader, get_available_languages

__all__ = [
    "FastPathProcessor",
    "FastPathResult",
    "is_fast_path_candidate",
    "KeywordLoader",
    "get_available_languages",
]
