# search_engine
# search_engine

CS121 Assignment 3: inverted-index search engine over the ICS `DEV` corpus.

## Setup

```
pip install nltk beautifulsoup4 lxml
```

## 1. Build the index (Milestone 1)

```
python indexer.py /path/to/DEV --out index_data
```

Writes partial indexes to disk every 10,000 docs, then merges them into
`index_data/index.txt`, along with `term_offsets.txt`, `docs.txt` and `stats.json`.

## 2. Search (Milestone 2)

```
python search.py                        # interactive prompt
python search.py -q "machine learning"  # run one query
python search.py --report               # run the 4 MS2 queries, save to ms2_results.txt
```

Options: `--index DIR` (default `index_data`), `-k N` number of results (default 5).
