import argparse
import heapq
import json
import os
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urldefrag

from html_parser import parse_page

DEFAULT_BATCH_SIZE = 10_000  # documents per partial index



# Corpus traversal
def iter_corpus(corpus_dir):
    """Yield (url, html) for every JSON page, in a deterministic order."""
    for domain in sorted(os.listdir(corpus_dir)):
        domain_dir = os.path.join(corpus_dir, domain)
        if not os.path.isdir(domain_dir):
            continue
        for name in sorted(os.listdir(domain_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(domain_dir, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    page = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  skipping unreadable file {path}: {e}", file=sys.stderr)
                continue
            url = urldefrag(page.get("url", "")).url  # ignore #fragments
            yield url, page.get("content", "") or ""


# Partial index I/O
def format_postings(postings):
    """[(doc_id, tf, imp), ...] -> 'doc_id,tf,imp;doc_id,tf,imp'"""
    return ";".join(f"{d},{tf},{imp}" for d, tf, imp in postings)


def write_partial(index, path):
    """Offload an in-memory index to disk, sorted by term."""
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(index):
            f.write(f"{term}\t{format_postings(index[term])}\n")


def read_partial(path, partial_no):
    """Stream (term, partial_no, postings_str) from a partial index file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            term, postings = line.rstrip("\n").split("\t", 1)
            yield term, partial_no, postings


def merge_partials(partial_paths, index_path, offsets_path):
    """
    K-way merge sorted partial indexes into one index file.

    Documents get increasing ids and partials are written in order, so for a
    given term, concatenating its postings in partial order keeps the final
    postings list sorted by doc_id. heapq.merge breaks ties on term by partial
    number, which gives exactly that order.

    Returns the number of unique terms.
    """
    streams = [read_partial(p, i) for i, p in enumerate(partial_paths)]
    unique_terms = 0

    # Binary mode so f.tell() gives true byte offsets for later seek() calls.
    with open(index_path, "wb") as idx, open(offsets_path, "w", encoding="utf-8") as offs:

        def flush(term, chunks):
            postings = ";".join(chunks)
            df = postings.count(";") + 1
            offs.write(f"{term}\t{idx.tell()}\n")
            idx.write(f"{term}\t{df}\t{postings}\n".encode("utf-8"))

        current_term, chunks = None, []
        for term, _, postings in heapq.merge(*streams):
            if term != current_term:
                if current_term is not None:
                    flush(current_term, chunks)
                    unique_terms += 1
                current_term, chunks = term, []
            chunks.append(postings)
        if current_term is not None:
            flush(current_term, chunks)
            unique_terms += 1

    return unique_terms


