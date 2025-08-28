from typing import List, Dict


def web_search(query: str, max_results: int = 5, region: str = "us-en", safesearch: str = "moderate") -> List[Dict]:
    """
    Perform a DuckDuckGo search and return a list of dicts with keys: title, url, snippet.
    Requires duckduckgo_search to be installed.
    """
    try:
        from duckduckgo_search import DDGS
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "duckduckgo_search is required. Install with `pip install duckduckgo-search`."
        ) from e

    results: List[Dict] = []
    # Use a context manager to share session and be polite
    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            region=region,
            safesearch=safesearch,
            timelimit="y",
            max_results=max_results,
        ):
            results.append(
                {
                    "title": r.get("title") or r.get("source") or "",
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": r.get("body") or r.get("snippet") or "",
                }
            )
    # Filter any malformed entries
    return [r for r in results if r.get("url")]

