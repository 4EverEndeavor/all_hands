from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import os

from . import search as search_mod
from . import fetch as fetch_mod


def _require_ollama():
    try:
        import ollama  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The 'ollama' package is required. Install with `pip install ollama`."
        ) from e


@dataclass
class SourceDoc:
    index: int
    url: str
    title: str
    snippet: str
    content: str


class WebAnswerAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        per_page_chars: int = 8000,
        verbose: bool = False,
    ) -> None:
        _require_ollama()
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.temperature = temperature
        self.per_page_chars = per_page_chars
        self.verbose = verbose

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def search_and_fetch(self, query: str, max_results: int = 5) -> List[SourceDoc]:
        self._log(f"Searching DuckDuckGo for: {query}")
        hits: List[Dict] = search_mod.web_search(query, max_results=max_results)
        docs: List[SourceDoc] = []
        for i, h in enumerate(hits, start=1):
            url = h.get("url")
            try:
                title, content = fetch_mod.fetch_and_extract(url, max_chars=self.per_page_chars)
            except Exception as e:
                self._log(f"! Failed to fetch {url}: {e}")
                continue
            doc = SourceDoc(
                index=i,
                url=url,
                title=title or h.get("title") or url,
                snippet=h.get("snippet") or "",
                content=content,
            )
            self._log(f"Fetched [{i}] {doc.title} -> {doc.url}")
            docs.append(doc)
        return docs

    def synthesize(self, question: str, docs: List[SourceDoc]) -> str:
        import ollama

        if not docs:
            return "No sources could be fetched to answer the question."

        sources_block_lines: List[str] = []
        for d in docs:
            sources_block_lines.append(
                f"Source [{d.index}] URL: {d.url}\nTitle: {d.title}\nExcerpt:\n{d.content}\n"
            )
        sources_block = "\n\n".join(sources_block_lines)

        system_prompt = (
            "You are a careful research assistant. Answer the user's question "
            "using only the provided web excerpts. Be concise and neutral. "
            "Cite sources inline like [1], [2] referring to the numbered sources. "
            "If the sources are insufficient, say so clearly."
        )

        user_prompt = (
            f"Question: {question}\n\n" 
            f"You have access to the following sources. Use them to answer and cite.\n\n"
            f"{sources_block}\n\n"
            f"Return a direct answer first, then a short list of sources by number with URLs."
        )

        self._log(f"Querying Ollama model: {self.model}")
        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": float(self.temperature)},
        )
        return resp["message"]["content"].strip()

    def answer(self, question: str, max_results: int = 5) -> str:
        docs = self.search_and_fetch(question, max_results=max_results)
        return self.synthesize(question, docs)

