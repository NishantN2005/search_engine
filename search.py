import argparse
import heapq
import math
import os
import time

from tokenizer import stem, tokenize, words

DEFAULT_INDEX_DIR = "index_data"
DEFAULT_TOP_K = 5

# BM25 ranking (replaces raw tf-idf so long pages don't win just by being long)
BM25_K1 = 1.2           # how quickly extra occurrences of a word stop adding score
BM25_B = 0.75           # how strongly scores are normalized by page length (0 = not at all)
IMPORTANT_WEIGHT = 2.0  # an occurrence in title/h1-h3/b/strong counts as 1 + this
TITLE_WEIGHT = 3.0      # an occurrence in <title> counts this much more on top of that
URL_WEIGHT = 0.5        # a query term in the page's URL adds URL_WEIGHT * idf to the score

# Very common words: a term found in more than this fraction of pages is treated
# like a stop word and ignored when the query also has more specific terms.
COMMON_DF_RATIO = 0.35

# Spelling correction: a query word that is in at most TYPO_MAX_DF pages is
# replaced by a word one edit away that is at least TYPO_MIN_GAIN times more common.
TYPO_MAX_DF = 5
TYPO_MIN_GAIN = 20
TYPO_MIN_LENGTH = 4     # don't "correct" very short words
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

MS2_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]


class SearchEngine:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR):
        self.offsets = self._load_offsets(os.path.join(index_dir, "term_offsets.txt"))
        self.urls, self.doc_lengths = self._load_docs(os.path.join(index_dir, "docs.txt"))
        self.num_docs = len(self.urls)
        self.avg_doc_length = sum(self.doc_lengths) / max(1, self.num_docs)
        # Precomputed per page (once, at startup) so scoring a query is just lookups:
        # the BM25 length factor, and the set of stemmed words in the page's URL.
        self.length_norm = [BM25_K1 * (1 - BM25_B + BM25_B * n / self.avg_doc_length)
                            for n in self.doc_lengths]
        self.url_terms = [frozenset(tokenize(url)) for url in self.urls]
        self.index_file = open(os.path.join(index_dir, "index.txt"), "rb")
        self.last_notes = []  # messages about the last query (spelling fix, partial matches)

    @staticmethod
    def _load_offsets(path):
        #maps each item to the position in index file
        offsets = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                term, off = line.rstrip("\n").split("\t")
                offsets[term] = int(off)
        return offsets

    @staticmethod
    def _load_docs(path):
        #loading document urls and lengths (number of tokens) into lists
        urls, lengths = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                _, url, length = line.rstrip("\n").split("\t")
                urls.append(url)
                lengths.append(int(length))
        return urls, lengths

    def close(self):
        self.index_file.close()

    def get_df(self, term):
        #reads only the "term<TAB>df<TAB>" header of a term's line, not its postings
        offset = self.offsets.get(term)
        if offset is None:
            return 0
        self.index_file.seek(offset)
        header = self.index_file.read(len(term.encode("utf-8")) + 16)
        return int(header.split(b"\t")[1])

    def get_postings(self, term):
        #reads a term's postings as {doc_id: (term_freq, important_freq, title_freq)}
        offset = self.offsets.get(term)
        if offset is None:
            return {}
        self.index_file.seek(offset)
        line = self.index_file.readline().decode("utf-8").rstrip("\n")
        # "d,tf,imp,title;..." -> one flat list of ints (fast: done in C, not a Python loop)
        nums = list(map(int, line.split("\t")[2].replace(";", ",").split(",")))
        return dict(zip(nums[0::4], zip(nums[1::4], nums[2::4], nums[3::4])))

    def idf(self, df):
        #BM25 idf: rare terms count a lot, terms in almost every page count very little
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def _bm25(self, doc_id, term_freq, important_freq, title_freq, idf):
        #BM25 score of one term in one document. Occurrences in important places
        #(title, headings, bold) count as extra occurrences.
        tf = term_freq + IMPORTANT_WEIGHT * important_freq + TITLE_WEIGHT * title_freq
        return idf * tf * (BM25_K1 + 1) / (tf + self.length_norm[doc_id])

    def _drop_common_terms(self, terms, dfs):
        #ignore stop-word-like terms when the query also has more specific ones
        limit = COMMON_DF_RATIO * self.num_docs
        specific = [t for t in terms if dfs[t] <= limit]
        if specific and len(specific) < len(terms):
            return specific
        return terms

    def _spelling_candidates(self, word):
        #all words one edit (delete, swap, replace, insert) away from word
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in ALPHABET]
        inserts = [a + c + b for a, b in splits for c in ALPHABET]
        return set(deletes + swaps + replaces + inserts)

    def _correct_spelling(self, query_words, terms, dfs):
        #replace very rare query words with a much more common word one edit away
        changed = False
        for i, (word, term) in enumerate(zip(query_words, terms)):
            if (dfs[term] > TYPO_MAX_DF or len(word) < TYPO_MIN_LENGTH
                    or not word.isalpha()):
                continue
            best, best_df = None, max(dfs[term], 1) * TYPO_MIN_GAIN - 1
            for candidate in self._spelling_candidates(word):
                candidate_term = stem(candidate)
                if candidate_term == term or candidate_term not in self.offsets:
                    continue
                df = self.get_df(candidate_term)
                if df > best_df:
                    best, best_df = candidate, df
            if best is not None:
                query_words[i], terms[i] = best, stem(best)
                dfs[terms[i]] = best_df
                changed = True
        return changed

    def _rank(self, doc_ids, postings_per_term, top_k):
        #BM25 score for each doc; sort by (how many query terms it has, score)
        scores, matched = {}, {}
        for term, postings in postings_per_term.items():
            idf = self.idf(len(postings))
            # URL bonus is added outside the length normalization, so a tiny page
            # doesn't get a bigger URL boost than a normal page from the same site.
            url_bonus = URL_WEIGHT * idf
            for doc_id in doc_ids:
                entry = postings.get(doc_id)
                if entry is None:
                    continue
                score = self._bm25(doc_id, *entry, idf)
                if term in self.url_terms[doc_id]:
                    score += url_bonus
                scores[doc_id] = scores.get(doc_id, 0) + score
                matched[doc_id] = matched.get(doc_id, 0) + 1
        return heapq.nlargest(top_k, scores.items(), key=lambda item: (matched[item[0]], item[1]))

    def search(self, query, top_k=DEFAULT_TOP_K):
        #returns (number of matches, top_k (url, score) pairs); notes go to self.last_notes
        self.last_notes = []
        query_words = list(dict.fromkeys(words(query, query=True)))  # unique, in order
        terms = [stem(w) for w in query_words]
        if not terms:
            return 0, []
        dfs = {t: self.get_df(t) for t in terms}

        # 1) Exact matching: every (non-common) term must be in the page.
        required = self._drop_common_terms(terms, dfs)
        postings = {t: self.get_postings(t) for t in required}
        matching = self._intersect(postings)

        # 2) Too few results: maybe a typo. Fix very rare words and try again.
        if len(matching) < top_k and self._correct_spelling(query_words, terms, dfs):
            self.last_notes.append("showing results for: " + " ".join(query_words))
            required = self._drop_common_terms(terms, dfs)
            postings = {t: self.get_postings(t) for t in required}
            matching = self._intersect(postings)

        # 3) Still too few: fill with pages that contain most of the terms.
        total = len(matching)
        if total < top_k and len(postings) > 1:
            matching = set().union(*(p.keys() for p in postings.values()))
            self.last_notes.append(f"only {total} page(s) contain every term; "
                                   f"also showing pages that contain most of them")
            total = len(matching)

        ranked = self._rank(matching, postings, top_k)
        return total, [(self.urls[doc_id], score) for doc_id, score in ranked]

    @staticmethod
    def _intersect(postings_per_term):
        #keeps only the docs that contain every query term (starting from the shortest list)
        lists = sorted(postings_per_term.values(), key=len)
        if not lists or not lists[0]:
            return set()
        matching_docs = set(lists[0])
        for postings in lists[1:]:
            matching_docs &= postings.keys()
        return matching_docs


def timed_search(engine, query, top_k):
    # Run a query and also return how long it took in milliseconds.
    t0 = time.perf_counter()
    total, results = engine.search(query, top_k)
    return total, results, (time.perf_counter() - t0) * 1000, engine.last_notes


def print_results(query, total, results, elapsed_ms, notes=()):
    print(f'\n{total:,} result(s) for "{query}" ({elapsed_ms:.1f} ms)')
    for note in notes:
        print(f"  ({note})")
    for rank, (url, score) in enumerate(results, 1):
        print(f"  {rank}. {url}  [score {score:.3f}]")
    print()


def run_report(engine, top_k, out_path="ms2_results.txt"):
    # Run the 4 Milestone 2 queries and save the top URLs for the report.
    lines = [f"Milestone 2 results (top {top_k})",
             f"Documents in index: {engine.num_docs:,}", ""]
    for q in MS2_QUERIES:
        total, results, ms, notes = timed_search(engine, q, top_k)
        print_results(q, total, results, ms, notes)
        lines.append(f'Query: "{q}"  ({total:,} matches, {ms:.1f} ms)')
        lines += [f"  {i}. {url}" for i, (url, _) in enumerate(results, 1)]
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved to {out_path}")
def main():
    ap = argparse.ArgumentParser(description="Search the inverted index.")
    ap.add_argument("--index", default=DEFAULT_INDEX_DIR, help="index directory")
    ap.add_argument("-k", "--top", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("-q", "--query", help="run one query and exit")
    ap.add_argument("--report", action="store_true", help="run the MS2 queries")
    args = ap.parse_args()

    engine = SearchEngine(args.index)
    print(f"Loaded {engine.num_docs:,} docs, {len(engine.offsets):,} terms")
    try:
        if args.report:
            run_report(engine, args.top)
        elif args.query:
            print_results(args.query, *timed_search(engine, args.query, args.top))
        else:
            print("Enter a query (blank line or Ctrl-C to quit).")
            while True:
                try:
                    q = input("search> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not q:
                    break
                print_results(q, *timed_search(engine, q, args.top))
    finally:
        engine.close()


if __name__ == "__main__":
    main()