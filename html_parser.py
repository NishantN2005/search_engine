import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

from tokenizer import tokenize

# Some "pages" are plain text or XML; bs4 warns about them but parses fine.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

IMPORTANT_TAGS = ["title", "h1", "h2", "h3", "b", "strong"]
NON_VISIBLE_TAGS = ["script", "style", "noscript", "template"]
