"""
Web tools for real-time web searching and webpage fetching.

Provides access to live online documentation, scientific references,
and web content across all agents (both Hybrid RAG and Baseline Agent).
"""

import asyncio
import re
import urllib.parse
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup


# User-agent header for web requests
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web for scientific information, code docs, or general queries.

    Tries DDGS (DuckDuckGo Search) first, with fallback to DuckDuckGo Lite HTML
    and Wikipedia OpenSearch API.

    Parameters
    ----------
    query : str
        The search string.
    max_results : int, default 5
        Maximum number of results to return.

    Returns
    -------
    list[dict[str, str]]
        List of dicts with keys: ``title``, ``href``, ``body``, ``source``.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # 1. Primary: DDGS library
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(cleaned_query, max_results=max_results))
            if raw_results:
                return [
                    {
                        "title": str(item.get("title", "")).strip(),
                        "href": str(item.get("href", "")).strip(),
                        "body": str(item.get("body", "")).strip(),
                        "source": "ddgs",
                    }
                    for item in raw_results
                ]
    except Exception as exc:
        print(f"[web_tools] DDGS search failed, trying fallback: {exc}")

    # 2. Fallback: DuckDuckGo Lite HTML
    try:
        resp = httpx.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": cleaned_query},
            headers={
                **_HEADERS,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            follow_redirects=True,
            timeout=10.0,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", class_="result-link")
            snippets = soup.find_all("td", class_="result-snippet")
            results = []
            for i in range(min(len(links), len(snippets), max_results)):
                title = links[i].get_text(strip=True)
                href = links[i].get("href", "")
                snippet = snippets[i].get_text(strip=True)
                if title and href:
                    results.append({
                        "title": title,
                        "href": href,
                        "body": snippet,
                        "source": "ddg_lite",
                    })
            if results:
                return results
    except Exception as exc:
        print(f"[web_tools] DDG Lite search fallback failed: {exc}")

    # 3. Fallback: Wikipedia OpenSearch API
    try:
        wiki_url = (
            "https://en.wikipedia.org/w/api.php?action=opensearch&search="
            f"{urllib.parse.quote(cleaned_query)}&limit={max_results}&namespace=0&format=json"
        )
        wresp = httpx.get(wiki_url, headers=_HEADERS, timeout=8.0)
        if wresp.status_code == 200:
            data = wresp.json()
            # Wikipedia opensearch format: [query, [titles], [descriptions], [urls]]
            if len(data) >= 4 and data[1]:
                titles, descs, urls = data[1], data[2], data[3]
                wiki_results = []
                for t, d, u in zip(titles, descs, urls):
                    wiki_results.append({
                        "title": t,
                        "href": u,
                        "body": d or f"Wikipedia article on {t}",
                        "source": "wikipedia",
                    })
                if wiki_results:
                    return wiki_results
    except Exception as exc:
        print(f"[web_tools] Wikipedia search fallback failed: {exc}")

    return []


async def aweb_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Asynchronously execute a web search without blocking the event loop."""
    return await asyncio.to_thread(web_search, query, max_results)


def fetch_web_page(url: str, max_chars: int = 4000) -> dict[str, Any]:
    """Fetch and extract clean readable text from a URL.

    Parameters
    ----------
    url : str
        Target web page URL.
    max_chars : int, default 4000
        Maximum character count of the returned content to fit into context.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``url``, ``title``, ``content``, ``status``, ``error``.
    """
    cleaned_url = url.strip()
    if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
        cleaned_url = f"https://{cleaned_url}"

    try:
        resp = httpx.get(
            cleaned_url,
            headers=_HEADERS,
            follow_redirects=True,
            timeout=12.0,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip unhelpful elements (scripts, styles, navigations, footers)
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else cleaned_url

        # Extract text separated by clean newlines
        raw_text = soup.get_text(separator="\n", strip=True)
        # Collapse multiple blank lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        cleaned_content = "\n".join(lines)

        if len(cleaned_content) > max_chars:
            cleaned_content = cleaned_content[:max_chars] + "\n... [Content truncated]"

        return {
            "url": cleaned_url,
            "title": title,
            "content": cleaned_content,
            "status": "success",
            "error": None,
        }
    except Exception as exc:
        return {
            "url": cleaned_url,
            "title": "",
            "content": "",
            "status": "error",
            "error": str(exc),
        }


async def afetch_web_page(url: str, max_chars: int = 4000) -> dict[str, Any]:
    """Asynchronously fetch a webpage without blocking the event loop."""
    return await asyncio.to_thread(fetch_web_page, url, max_chars)


def format_web_search_markdown(query: str, results: list[dict[str, str]]) -> str:
    """Format search results into clean GitHub-flavored markdown."""
    if not results:
        return f"*(No live web search results found for `{query}`)*"

    lines = [f"#### 🌐 Live Web Search Results: `{query}`\n"]
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        href = item.get("href", "#")
        body = item.get("body", "").replace("\n", " ")
        lines.append(f"{i}. **[{title}]({href})**\n   {body}\n")

    return "\n".join(lines).strip()


def format_web_page_markdown(page: dict[str, Any]) -> str:
    """Format fetched webpage content into clean GitHub-flavored markdown."""
    url = page.get("url", "")
    title = page.get("title", url)
    content = page.get("content", "")
    error = page.get("error")

    if error:
        return f"⚠️ **Failed to fetch [{title}]({url}):** `{error}`"

    return f"#### 📄 Webpage: [{title}]({url})\n\n```text\n{content}\n```"


def extract_web_tool_calls(text: str) -> list[dict[str, str]]:
    """Extract explicit web tool call tags emitted by LLMs.

    Supported patterns:
      - ``<web_search>query</web_search>``
      - ``[web_search: query]``
      - ``<fetch_url>url</fetch_url>``
      - ``[fetch_url: url]``
    """
    calls: list[dict[str, str]] = []

    # <web_search>query</web_search>
    for m in re.finditer(r'<web_search>(.*?)</web_search>', text, re.IGNORECASE | re.DOTALL):
        q = m.group(1).strip()
        if q:
            calls.append({"tool": "web_search", "argument": q})

    # [web_search: query]
    for m in re.finditer(r'\[web_search:\s*([^\]]+)\]', text, re.IGNORECASE):
        q = m.group(1).strip()
        if q and not any(c["argument"] == q for c in calls):
            calls.append({"tool": "web_search", "argument": q})

    # <fetch_url>url</fetch_url>
    for m in re.finditer(r'<fetch_url>(.*?)</fetch_url>', text, re.IGNORECASE | re.DOTALL):
        u = m.group(1).strip()
        if u:
            calls.append({"tool": "fetch_url", "argument": u})

    # [fetch_url: url]
    for m in re.finditer(r'\[fetch_url:\s*([^\]]+)\]', text, re.IGNORECASE):
        u = m.group(1).strip()
        if u and not any(c["argument"] == u for c in calls):
            calls.append({"tool": "fetch_url", "argument": u})

    return calls


def detect_web_intent(query: str) -> tuple[bool, str, Optional[str]]:
    """Detect if a user message naturally requests web search or URL fetching.

    Returns
    -------
    tuple[bool, str, Optional[str]]
        ``(needs_web, action_type, extracted_target)``
        where ``action_type`` is ``"search"`` or ``"fetch"``.
    """
    q_low = query.lower()

    # 1. URL pattern
    url_match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', query)
    if url_match:
        return True, "fetch", url_match.group(0)

    # 2. Web search triggers
    search_triggers = (
        "search the web", "search online", "web search", "google", "search for",
        "look up online", "browse the web", "browse online", "find online",
        "latest documentation", "current documentation", "latest release",
        "latest version", "check the website", "search internet",
    )
    for trigger in search_triggers:
        if trigger in q_low:
            # Strip the trigger to isolate the core search topic
            clean_topic = re.sub(
                r'^(?:please\s+)?(?:search\s+the\s+web\s+for|search\s+online\s+for|web\s+search\s+for|look\s+up\s+online|search\s+for|google)\s*',
                '',
                query,
                flags=re.IGNORECASE,
            ).strip()
            return True, "search", clean_topic or query

    return False, "none", None
