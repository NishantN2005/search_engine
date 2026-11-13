import argparse
import math
import os
import time

from tokenizer import tokenize

DEFAULT_INDEX_DIR = "index_data"
DEFAULT_TOP_K = 5
IMPORTANT_WEIGHT = 2.0  # extra weight for terms in title/h1-h3/b/strong

MS2_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]


class SearchEngine:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR):
        self.offsets = self._load_offsets(os.path.join(index_dir, "term_offsets.txt"))
        self.urls = self._load_docs(os.path.join(index_dir, "docs.txt"))
        self.num_docs = len(self.urls)
        self.index_file = open(os.path.join(index_dir, "index.txt"), "rb")
    
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
    #loading document urls into a list
        urls = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                urls.append(line.rstrip("\n").split("\t")[1])
        return urls

    def close(self):
        self.index_file.close()

    def get_postings(self, term):
    #reads a term's postings from the index file as {doc_id: (term_freq, important_freq)}
        offset = self.offsets.get(term)
        if offset is None:
            return {}
        self.index_file.seek(offset)
        line = self.index_file.readline().decode("utf-8").rstrip("\n")
        postings = {}
        for posting in line.split("\t")[2].split(";"):
            doc_id, term_freq, important_freq = posting.split(",")
            postings[int(doc_id)] = (int(term_freq), int(important_freq))
        return postings

    @staticmethod
    def _intersect(postings_per_term):
    #keeps only the docs that contain every query term
        matching_docs = set(postings_per_term[0])
        for postings in postings_per_term[1:]:
            matching_docs &= postings.keys()
        return matching_docs

    def _score(self, matching_docs, postings_per_term):
    #tf-idf score for each doc, with a bonus for important words
        scores = {}
        for postings in postings_per_term:
            idf = math.log10(self.num_docs / len(postings))
            for doc_id in matching_docs:
                term_freq, important_freq = postings[doc_id]
                weight = 1 + math.log10(term_freq)
                if important_freq > 0:
                    weight += IMPORTANT_WEIGHT
                scores[doc_id] = scores.get(doc_id, 0) + weight * idf
        return scores

    def search(self, query, top_k=DEFAULT_TOP_K):
    #returns (number of matches, top_k (url, score) pairs)
        query_terms = set(tokenize(query))
        postings_per_term = []
        for term in query_terms:
            postings_per_term.append(self.get_postings(term))

        #no results if the query is empty or any term isn't in the index
        if not postings_per_term or not all(postings_per_term):
            return 0, []

        matching_docs = self._intersect(postings_per_term)
        scores = self._score(matching_docs, postings_per_term)
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked_docs:
            results.append((self.urls[doc_id], score))
        return len(matching_docs), results
