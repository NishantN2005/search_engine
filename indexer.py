import argparse
import hashlib
import heapq
import json
import os
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urldefrag

from html_parser import parse_page

DEFAULT_BATCH_SIZE = 10_000  # documents per partial index
SIMHASH_MAX_DIST = 3         # fingerprints differing in <= 3 of 64 bits are near-duplicates
SIMHASH_MIN_TOKENS = 50      # simhash is unreliable on very short pages


# Duplicate detection
def page_checksum(tokens):
    """MD5 of the page's words. Same words = same checksum."""
    text = " ".join(tokens)
    text_bytes = text.encode("utf-8")
    checksum = hashlib.md5(text_bytes).hexdigest()
    return checksum


def word_hash(word):
    """Turn a word into a 64-bit number (same word = same number)."""
    word_bytes = word.encode("utf-8")
    digest = hashlib.md5(word_bytes).digest()
    first_8_bytes = digest[:8]   # 8 bytes = 64 bits
    return int.from_bytes(first_8_bytes, "big")


def simhash(tf):
    """64-bit fingerprint. Similar pages get similar fingerprints."""
    votes = [0] * 64

    # Each distinct word votes once on every bit: + if its bit is 1, - if it is 0.
    # Votes are NOT weighted by term frequency: we keep stop words, so with tf
    # weights "the"/"of"/"and" decide every bit and unrelated long pages end up
    # with nearly identical fingerprints (false near-duplicates).
    for word in tf:
        h = word_hash(word)
        for i in range(64):
            bit_is_one = (h >> i) & 1
            if bit_is_one:
                votes[i] += 1
            else:
                votes[i] -= 1

    # Bits with more + votes become 1.
    fingerprint = 0
    for i in range(64):
        if votes[i] > 0:
            fingerprint = fingerprint | (1 << i)
    return fingerprint


def bits_different(fp1, fp2):
    """How many of the 64 bits differ between two fingerprints."""
    differing = fp1 ^ fp2   # 1 where bits differ
    return differing.bit_count()


def find_near_duplicate(fp, fingerprints):
    """URL of a kept page within 3 bits of fp, or None."""
    for other_fp, other_url in fingerprints:
        if bits_different(fp, other_fp) <= SIMHASH_MAX_DIST:
            return other_url
    return None



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
    """[(doc_id, tf, imp, title), ...] -> 'doc_id,tf,imp,title;doc_id,tf,imp,title'

    tf    = occurrences anywhere in the page
    imp   = occurrences inside title/h1/h2/h3/b/strong
    title = occurrences inside <title>
    """
    return ";".join(f"{d},{tf},{imp},{title}" for d, tf, imp, title in postings)


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

# Main build
def build_index(corpus_dir, out_dir, batch_size):
    start = time.time()
    partial_dir = os.path.join(out_dir, "partials")
    os.makedirs(partial_dir, exist_ok=True)

    index = defaultdict(list)  # term -> [(doc_id, tf, imp_tf, title_tf), ...]
    partial_paths = []
    doc_id = 0

    def offload():
        """Write the in-memory index to a partial file on disk, then clear it."""
        nonlocal index
        path = os.path.join(partial_dir, f"partial_{len(partial_paths):03d}.txt")
        write_partial(index, path)
        partial_paths.append(path)
        print(f"  offloaded partial index #{len(partial_paths)} "
              f"({len(index):,} terms) after {doc_id:,} docs "
              f"[{time.time() - start:.0f}s]")
        index = defaultdict(list)

    seen_urls = set()
    checksums = {}             # checksum -> url of kept page
    fingerprints = []          # [(simhash, url), ...] of kept pages
    dups = Counter()           # "url" / "exact" / "near" -> pages skipped

    with open(os.path.join(out_dir, "docs.txt"), "w", encoding="utf-8") as docs, \
         open(os.path.join(out_dir, "duplicates.txt"), "w", encoding="utf-8") as dup_log:
        for url, html in iter_corpus(corpus_dir):
            # Duplicates are skipped before getting a doc_id, so ids stay contiguous.
            if url in seen_urls:
                dups["url"] += 1
                continue
            seen_urls.add(url)

            all_tokens, important_tokens, title_tokens = parse_page(html)
            tf = Counter(all_tokens)

            # Exact duplicate?
            checksum = page_checksum(all_tokens)
            if checksum in checksums:
                original = checksums[checksum]
                dups["exact"] += 1
                dup_log.write(f"exact\t{original}\t{url}\n")
                continue

            # Near duplicate?
            fp = None
            if len(all_tokens) >= SIMHASH_MIN_TOKENS:
                fp = simhash(tf)
                original = find_near_duplicate(fp, fingerprints)
                if original is not None:
                    dups["near"] += 1
                    dup_log.write(f"near\t{original}\t{url}\n")
                    continue

            # new page: remember it
            checksums[checksum] = url
            if fp is not None:
                fingerprints.append((fp, url))

            imp = Counter(important_tokens)
            title = Counter(title_tokens)

            for term, count in tf.items():
                index[term].append((doc_id, count, imp.get(term, 0), title.get(term, 0)))

            docs.write(f"{doc_id}\t{url}\t{len(all_tokens)}\n")
            doc_id += 1

            if doc_id % batch_size == 0:
                offload()

    if index:
        offload()

    print(f"Merging {len(partial_paths)} partial indexes ...")
    index_path = os.path.join(out_dir, "index.txt")
    offsets_path = os.path.join(out_dir, "term_offsets.txt")
    unique_terms = merge_partials(partial_paths, index_path, offsets_path)

    kb = lambda p: os.path.getsize(p) / 1024
    stats = {
        "num_documents": doc_id,
        "num_unique_tokens": unique_terms,
        "url_duplicates_skipped": dups["url"],
        "exact_duplicates_skipped": dups["exact"],
        "near_duplicates_skipped": dups["near"],
        "num_partial_indexes": len(partial_paths),
        "index_size_kb": round(kb(index_path), 1),
        "term_offsets_size_kb": round(kb(offsets_path), 1),
        "docs_table_size_kb": round(kb(os.path.join(out_dir, "docs.txt")), 1),
        "partial_indexes_total_kb": round(sum(kb(p) for p in partial_paths), 1),
        "build_seconds": round(time.time() - start, 1),
    }
    stats["total_index_on_disk_kb"] = round(
        stats["index_size_kb"] + stats["term_offsets_size_kb"] + stats["docs_table_size_kb"], 1)

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build the inverted index.")
    ap.add_argument("corpus_dir", help="path to the extracted DEV folder")
    ap.add_argument("--out", default="index_data", help="output directory")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="documents per partial index (default 10000)")
    args = ap.parse_args()
    build_index(args.corpus_dir, args.out, args.batch_size)


if __name__ == "__main__":
    main()



