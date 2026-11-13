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
    def init(self, index_dir=DEFAULT_INDEX_DIR):
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