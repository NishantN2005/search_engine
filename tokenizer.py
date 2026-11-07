"""
Tokenization and stemming shared by the indexer and (later) the search component.

Tokens are all alphanumeric sequences ([a-z0-9]+) after lowercasing.
No stop-word removal (per the assignment spec). Every token is Porter-stemmed.
The same function MUST be used on queries so query terms match index terms.
"""

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


@lru_cache(maxsize=500_000)
def stem(token: str) -> str:
  """Porter-stem a single lowercase token (cached: stemming is the hot path)."""
  return _stemmer.stem(token)


def tokenize(text: str) -> list[str]:
  """Split text into lowercase alphanumeric tokens and stem each one."""
  return [stem(t) for t in _TOKEN_RE.findall(text.lower())]
