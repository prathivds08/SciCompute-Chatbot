"""Tools package for scientific computation, code execution, and web research."""

from backend.tools.code_tools import (
    execute_code,
    extract_code_blocks,
    extract_docker_tool_calls,
    detect_docker_intent,
    is_followup_execution_request,
    resolve_fallback_analysis_script,
    format_execution_result_markdown,
    get_executor,
    sanitize_markdown_text,
    sanitize_scientific_code,
)
from backend.tools.web_tools import (
    aweb_search,
    afetch_web_page,
    detect_web_intent,
    extract_web_tool_calls,
    fetch_web_page,
    format_web_page_markdown,
    format_web_search_markdown,
    web_search,
)

__all__ = [
    "execute_code",
    "extract_code_blocks",
    "extract_docker_tool_calls",
    "detect_docker_intent",
    "is_followup_execution_request",
    "resolve_fallback_analysis_script",
    "format_execution_result_markdown",
    "get_executor",
    "sanitize_markdown_text",
    "sanitize_scientific_code",
    "web_search",
    "aweb_search",
    "fetch_web_page",
    "afetch_web_page",
    "format_web_search_markdown",
    "format_web_page_markdown",
    "extract_web_tool_calls",
    "detect_web_intent",
]
