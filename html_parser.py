import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

from tokenizer import tokenize

# Some "pages" are plain text or XML; bs4 warns about them but parses fine.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

IMPORTANT_TAGS = ["title", "h1", "h2", "h3", "b", "strong"]
NON_VISIBLE_TAGS = ["script", "style", "noscript", "template"]

def parse_page(html: str) -> tuple[list[str], list[str], list[str]]:
    """Return (all_tokens, important_tokens, title_tokens) for a page's raw content."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return tokenize(html), [], []
    for tag in soup(NON_VISIBLE_TAGS):
        tag.decompose()

    important_tokens: list[str] = []
    for tag in soup.find_all(IMPORTANT_TAGS):
        if tag.find_parent(IMPORTANT_TAGS) is not None:
            continue
        important_tokens.extend(tokenize(tag.get_text(" ")))

    # The <title> is the strongest signal of what a page is about, so it is
    # also counted on its own (it is still part of important_tokens too).
    title = soup.find("title")
    title_tokens = tokenize(title.get_text(" ")) if title is not None else []

    all_tokens = tokenize(soup.get_text(" "))
    return all_tokens, important_tokens, title_tokens