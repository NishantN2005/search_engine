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

## 3. Final search engine (Milestone 3)

Rebuild the index using the command in step 1, then try a query:

```
python search.py -q "machine lerning" -k 10
```

The indexer skips duplicate and near-duplicate pages. Search uses BM25 ranking
with extra weight for matches in titles and URLs. It skips very common words
when more specific terms are present and tries spelling corrections and
partial matches when there are too few results.

See [TEST.txt](TEST.txt) for the 20 test queries and what changed for the ones
that performed poorly.

## 4. Web interface (extra credit)

```
python web_gui.py                     # opens http://localhost:8000 in your browser
python web_gui.py --port 8080 --no-browser
```

Options: `--index DIR` (default `index_data`), `--port N` (default 8000),
`--no-browser` to skip opening a tab. Only uses the Python standard library.

Type a query and press Enter. Each result shows its rank, URL and BM25
score, 10 per page (up to 10 pages). A note appears when a spelling was
corrected or partial matches were added. Stop the server with Ctrl-C.
