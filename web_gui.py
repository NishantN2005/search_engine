"""
Web GUI for the search engine (extra credit).

    python web_gui.py                          # then open http://localhost:8000
    python web_gui.py --index index_data --port 8000 --no-browser

Uses only the Python standard library (http.server). The page is a plain
HTML form; submitting it requests /?q=...&page=N, and the server runs
SearchEngine.search() and returns the results as HTML.
"""

import argparse
import html
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote_plus, urlparse

from search import DEFAULT_INDEX_DIR, SearchEngine

RESULTS_PER_PAGE = 10
MAX_PAGES = 10            # at most 100 results can be paged through
MAX_QUERY_LENGTH = 200

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; max-width: 800px; margin: 20px auto; padding: 0 15px; }}
  .result {{ margin-bottom: 12px; }}
  .info {{ color: #666; font-size: small; }}
</style>
</head>
<body>
<h2><a href="/">ICS Search</a></h2>
<form action="/" method="get">
  <input type="text" name="q" value="{query}" size="50" autofocus>
  <input type="submit" value="Search">
</form>
{body}
</body>
</html>
"""


class SearchHandler(BaseHTTPRequestHandler):
    engine = None                   # set in main()
    engine_lock = threading.Lock()  # the engine shares one open index file, so one query at a time

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self._send(404, "Not found")
            return
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0].strip()[:MAX_QUERY_LENGTH]
        try:
            page = max(1, min(MAX_PAGES, int(params.get("page", ["1"])[0])))
        except ValueError:
            page = 1

        body = self._results_html(query, page) if query else ""
        title = f"{query} - ICS Search" if query else "ICS Search"
        self._send(200, PAGE.format(title=html.escape(title), query=html.escape(query), body=body))

    def _results_html(self, query, page):
        # Ask for enough results to fill this page, then show only this page's slice.
        with self.engine_lock:
            t0 = time.perf_counter()
            total, results = self.engine.search(query, page * RESULTS_PER_PAGE)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            notes = list(self.engine.last_notes)

        lines = [f'<p class="info">{total:,} results ({elapsed_ms:.1f} ms)</p>']
        for note in notes:
            lines.append(f"<p><i>{html.escape(note)}</i></p>")
        if not results:
            lines.append(f"<p>No pages match <b>{html.escape(query)}</b>.</p>")

        start = (page - 1) * RESULTS_PER_PAGE
        for rank, (url, score) in enumerate(results[start:], start + 1):
            link = html.escape(url)
            lines.append(f'<div class="result">{rank}. <a href="{link}">{link}</a>'
                         f'<br><span class="info">score {score:.3f}</span></div>')

        pages = math.ceil(min(total, MAX_PAGES * RESULTS_PER_PAGE) / RESULTS_PER_PAGE)
        nav = []
        if page > 1:
            nav.append(f'<a href="/?q={quote_plus(query)}&page={page - 1}">&lt; Previous</a>')
        if page < pages:
            nav.append(f'<a href="/?q={quote_plus(query)}&page={page + 1}">Next &gt;</a>')
        if nav:
            lines.append(f"<p>Page {page} of {pages} &nbsp; " + " &nbsp; ".join(nav) + "</p>")
        return "\n".join(lines)

    def _send(self, status, text):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep the console quiet


def main():
    ap = argparse.ArgumentParser(description="Web interface for the search engine.")
    ap.add_argument("--index", default=DEFAULT_INDEX_DIR, help="index directory")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    print("Loading index ...")
    SearchHandler.engine = SearchEngine(args.index)
    print(f"Loaded {SearchHandler.engine.num_docs:,} docs, {len(SearchHandler.engine.offsets):,} terms")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), SearchHandler)
    url = f"http://localhost:{args.port}/"
    print(f"Search GUI running at {url}  (Ctrl-C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
        SearchHandler.engine.close()


if __name__ == "__main__":
    main()
