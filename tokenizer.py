"""
Tokenization and stemming shared by the indexer and the search component.

Tokens are alphanumeric sequences ([a-z0-9]+) after lowercasing, with two
additions so common technical terms aren't lost:
  - Language-style names of 1-2 characters followed by "++" or "#" are kept
    whole: "C++" -> "c++", "C#" -> "c#", "g++", "f#" (not just "c").
  - Hyphenated words of 2-3 alphabetic parts are also indexed joined together:
    "Wi-Fi" -> "wi", "fi" AND "wifi", so a query for "wifi" finds pages that
    wrote "Wi-Fi". In a query, a hyphenated word becomes only its joined form
    ("wi-fi" -> "wifi"), which matches pages that wrote either spelling.
No stop-word removal (per the assignment spec). Every token is Porter-stemmed.
The same functions MUST be used on queries so query terms match index terms.
"""

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]?(?:\+\+|#)(?![a-z0-9+#])|[a-z0-9]+")
_HYPHEN_RE = re.compile(r"(?<![a-z0-9-])[a-z]+(?:-[a-z]+){1,2}(?![a-z0-9-])")
_stemmer = PorterStemmer()


@lru_cache(maxsize=500_000)
def stem(token: str) -> str:
  """Porter-stem a single lowercase token (cached: stemming is the hot path)."""
  return _stemmer.stem(token)


def words(text: str, query: bool = False) -> list[str]:
  """Split text into lowercase (unstemmed) tokens."""
  text = text.lower()
  if query:
    # "wi-fi" -> "wifi": matches documents that wrote "wifi" or "Wi-Fi"
    return _TOKEN_RE.findall(_HYPHEN_RE.sub(lambda m: m.group().replace("-", ""), text))
  joined = [m.replace("-", "") for m in _HYPHEN_RE.findall(text)]
  return _TOKEN_RE.findall(text) + joined


def tokenize(text: str, query: bool = False) -> list[str]:
  """Split text into tokens and stem each one."""
  return [stem(t) for t in words(text, query)]
