from typing import Tuple
import re

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _clean_text(text: str) -> str:
    # Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_and_extract(url: str, max_chars: int = 8000, timeout: int = 15) -> Tuple[str, str]:
    """
    Fetch a URL and return (title, cleaned_text) up to max_chars.
    Uses a simple heuristic: keep headings and paragraph text, drop script/style/nav.
    """
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-contenty elements
    for tag in soup(["script", "style", "noscript", "svg", "img", "video", "footer", "nav", "aside"]):
        tag.decompose()

    title = (soup.title.string if soup.title and soup.title.string else "").strip()

    # Prefer visible text from headings and paragraphs
    texts = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        t = el.get_text(separator=" ", strip=True)
        if t and len(t.split()) >= 3:  # skip very short fragments
            texts.append(t)

    content = _clean_text("\n".join(texts))
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars] + "\n... [truncated]"
    return title, content

