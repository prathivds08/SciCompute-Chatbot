"""
Test suite for Web Tools integration across Baseline and RAG agents.
"""

import asyncio
import sys
from pathlib import Path

# Ensure backend can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.tools.web_tools import (
    web_search,
    aweb_search,
    fetch_web_page,
    afetch_web_page,
    format_web_search_markdown,
    format_web_page_markdown,
    extract_web_tool_calls,
    detect_web_intent,
)
from backend.agents.baseline_agent import BaselineAgent


def test_intent_detection():
    print("[1/5] Testing web intent detection...")
    
    # 1. URL pattern
    needs_web, action, target = detect_web_intent("Check out https://wiki.fysik.dtu.dk/ase/ for docs")
    assert needs_web is True, "Expected URL detection to trigger web intent"
    assert action == "fetch"
    assert "https://wiki.fysik.dtu.dk/ase/" in target

    # 2. Search query pattern
    needs_web, action, target = detect_web_intent("Search the web for Quantum ESPRESSO latest release")
    assert needs_web is True, "Expected search query to trigger web intent"
    assert action == "search"
    assert "quantum espresso" in target.lower()

    # 3. Regular query pattern
    needs_web, action, target = detect_web_intent("Calculate water mass density profile with ASE")
    assert needs_web is False, "Regular query should not falsely trigger web intent"

    print("  -> Intent detection PASSED.")


def test_tool_call_extraction():
    print("[2/5] Testing tool call tag extraction...")
    
    text = (
        "Let me look up the latest documentation for DeePMD.\n"
        "<web_search>DeePMD-kit latest version</web_search>\n"
        "And also fetch the site:\n"
        "<fetch_url>https://deepmd.org</fetch_url>"
    )
    calls = extract_web_tool_calls(text)
    assert len(calls) == 2, f"Expected 2 tool calls, got {len(calls)}"
    assert calls[0]["tool"] == "web_search"
    assert calls[0]["argument"] == "DeePMD-kit latest version"
    assert calls[1]["tool"] == "fetch_url"
    assert calls[1]["argument"] == "https://deepmd.org"

    print("  -> Tool call tag extraction PASSED.")


def test_web_search_and_fetch():
    print("[3/5] Testing live web search and webpage fetching...")

    # Search
    results = web_search("Atomic Simulation Environment ASE python", max_results=3)
    assert isinstance(results, list)
    assert len(results) > 0, "Expected non-empty search results"
    formatted_search = format_web_search_markdown("Atomic Simulation Environment ASE python", results)
    assert "#### 🌐 Live Web Search Results" in formatted_search
    print(f"  -> Found {len(results)} search results. Formatted markdown length: {len(formatted_search)}")

    # Fetch
    page = fetch_web_page("https://httpbin.org/html", max_chars=1000)
    assert page["status"] == "success"
    assert "Herman Melville - Moby-Dick" in page["content"]
    formatted_page = format_web_page_markdown(page)
    assert "#### 📄 Webpage:" in formatted_page
    print("  -> Live web search and fetch PASSED.")


async def test_baseline_agent_web_integration():
    print("[4/5] Testing BaselineAgent web tool workflow...")
    agent = BaselineAgent()
    assert hasattr(agent, "agenerate_response_with_tools")
    assert hasattr(agent, "generate_response_with_tools")

    # Check that proactive web search intent triggers web tool execution
    test_query = "Search the web for Quantum ESPRESSO latest version"
    needs_web, action_type, target = detect_web_intent(test_query)
    assert needs_web is True

    search_res = await aweb_search(target, max_results=3)
    assert len(search_res) > 0
    formatted = format_web_search_markdown(target, search_res)
    assert "Quantum ESPRESSO" in formatted or "quantum" in formatted.lower()

    print("  -> BaselineAgent web tools integration PASSED.")


def test_system_prompt_web_tools():
    print("[5/5] Testing BaselineAgent system prompt for web tools...")
    from backend.agents.baseline_agent import BASELINE_ZERO_SHOT_SYSTEM_PROMPT
    assert "<web_tools>" in BASELINE_ZERO_SHOT_SYSTEM_PROMPT
    assert "<web_search>" in BASELINE_ZERO_SHOT_SYSTEM_PROMPT
    assert "<fetch_url>" in BASELINE_ZERO_SHOT_SYSTEM_PROMPT
    print("  -> System prompt contains <web_tools> block PASSED.")


def main():
    test_intent_detection()
    test_tool_call_extraction()
    test_web_search_and_fetch()
    asyncio.run(test_baseline_agent_web_integration())
    test_system_prompt_web_tools()
    print("\nALL WEB TOOLS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
