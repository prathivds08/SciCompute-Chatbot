"""
Tools for extracting, executing, and formatting scientific code and sandbox results.
"""

import re
import time
from typing import Any

from backend.sandbox.executor import CodeExecutor
from backend.tools.deepmd_analysis_tools import (
    save_analysis_script,
    build_water_density_profile_script,
    build_rdf_script,
    build_extxyz_to_deepmd_raw_script,
    build_loss_curve_script,
    build_parity_plot_script,
)

# Global singleton executor
_executor = CodeExecutor()


def get_executor() -> CodeExecutor:
    return _executor


def sanitize_scientific_code(code: str) -> str:
    """Auto-repair known LLM hallucinations in scientific Python (ASE & NumPy)."""
    # 1. Fix: read(..., index=":") when used for single structure inspection or assigned to atoms/structure/slab
    code = re.sub(
        r'(\b(?:atoms|atom|slab|structure|system|mol)\s*=\s*(?:ase\.io\.)?read\([^)]*?)index\s*=\s*[\'"]:\'"]',
        r'\g<1>index=0',
        code,
    )

    # If code calls Atoms methods (.get_positions, .cell, etc.) on an object
    has_direct_atoms_methods = bool(
        re.search(
            r'\b\w+\.(?:get_positions|get_scaled_positions|cell|get_cell|get_potential_energy|get_forces)\b',
            code,
        )
    )
    if has_direct_atoms_methods:
        # Check if code actually iterates over frames: e.g. 'for frame in frames:'
        m = re.search(r'(\b\w+)\s*=\s*(?:ase\.io\.)?read\([^)]*index\s*=\s*[\'"]:\'"]', code)
        if m:
            var_name = m.group(1)
            # Only keep index=":" if the code actually loops over this specific variable: 'for ... in var_name:'
            if not re.search(rf'for\s+\w+\s+in\s+{re.escape(var_name)}\b', code):
                code = re.sub(
                    rf'(\b{re.escape(var_name)}\s*=\s*(?:ase\.io\.)?read\([^)]*?)index\s*=\s*[\'"]:\'"]',
                    r'\g<1>index=0',
                    code,
                )
        # Catch any helper functions returning read(..., index=":") when Atoms methods are used
        code = re.sub(
            r'(\b(?:ase\.io\.)?read\([^)]*?)index\s*=\s*[\'"]:\'"]',
            r'\g<1>index=0',
            code,
        )

    # 2. Fix: ambiguous truth value error in numpy array comparisons
    code = re.sub(
        r'if\s+any\s*\(\s*(\w+)\s*([<>=!]+)\s*([0-9.eE+-]+)\s+for\s+\1\s+in\s+(\w+)\s*\)\s*:',
        r'if np.any(\4 \2 \3):',
        code,
    )
    code = re.sub(
        r'any\s*\(\s*(\w+)\s*([<>=!]+)\s*([0-9.eE+-]+)\s+for\s+\1\s+in\s+(\w+)\s*\)',
        r'np.any(\4 \2 \3)',
        code,
    )
    code = re.sub(
        r'if\s+any\s*\(\s*(\w+)\s*<\s*([0-9.eE+-]+)\s+for\s+\1\s+in\s+(\w+)\s*\)\s*:',
        r'if np.any(\3 < \2):',
        code,
    )

    # 3. Fix: ASE method hallucinations
    code = re.sub(r'(\b\w+)\.get_count\(\)', r'len(\1)', code)
    code = re.sub(r'(\b\w+)\.num_atoms\(\)', r'len(\1)', code)
    code = re.sub(r'(\b\w+)\.get_energy\(\)', r'\1.get_potential_energy()', code)

    return code


def sanitize_markdown_text(text: str) -> str:
    """Sanitize Python code blocks inside markdown text so user sees working code."""
    def _replace_block(m):
        lang = m.group(1)
        body = m.group(2)
        if lang.strip().lower() in ("python", "py", ""):
            body = sanitize_scientific_code(body)
        return f"```{lang}\n{body}\n```"

    return re.sub(r"```(\w*)\r?\n([\s\S]*?)```", _replace_block, text)


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """Extract code blocks with their specified language from markdown text.

    Returns a list of dicts: [{"language": "python", "code": "..."}]
    """
    pattern = r"```(\w*)\r?\n([\s\S]*?)```"
    matches = re.findall(pattern, text)
    blocks = []
    for lang, code in matches:
        clean_lang = lang.strip().lower() or "code"
        clean_code = code.strip()
        if clean_code:
            if clean_lang in ("python", "py"):
                clean_code = sanitize_scientific_code(clean_code)
            blocks.append({"language": clean_lang, "code": clean_code})
    return blocks


def execute_code(
    code: str,
    language: str = "python",
    timeout: int | None = None,
    session_id: str | None = None,
    save_script_name: str | None = None,
) -> dict[str, Any]:
    """Execute code in the sandbox and return the structured result.
    
    If save_script_name is specified (or by default for python scripts),
    the code is persisted to backend/data/sandbox/scripts/{session_id}/<name>.py first.
    """
    if language.strip().lower() in ("python", "py"):
        code = sanitize_scientific_code(code)
    saved_meta = None
    if save_script_name or language.strip().lower() in ("python", "py"):
        fname = save_script_name or f"analysis_{int(time.time())}.py"
        sid = session_id or "default"
        try:
            saved_meta = save_analysis_script(
                code=code,
                script_name=fname,
                session_id=sid,
                description="Auto-generated DeePMD / scientific analysis script",
            )
        except Exception as exc:
            print(f"[code_tools] Failed to register script artifact: {exc}")

    result = _executor.execute(
        code=code,
        language=language,
        timeout=timeout,
        session_id=session_id,
    )
    if saved_meta:
        result["saved_script"] = saved_meta

    return result


def format_execution_result_markdown(result: dict[str, Any]) -> str:
    """Format execution output into clean GitHub-flavored markdown with status, container badge, and artifacts."""
    success = result.get("success", False)
    duration = result.get("duration", 0.0)
    returncode = result.get("returncode", 0)
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "").strip()
    artifacts = result.get("artifacts", [])
    executed_in_docker = result.get("executed_in_docker", False)
    saved_script = result.get("saved_script")

    exec_env = "🐳 Isolated Docker Container" if executed_in_docker else "💻 Local Sandbox"
    status_str = f"**Status:** {'✅ Succeeded' if success else '❌ Failed (Exit Code ' + str(returncode) + ')'} `({duration}s)` • *{exec_env}*"

    sections = [f"\n### 💻 Execution Results\n{status_str}"]

    if saved_script:
        sections.append(f"📁 **Script Saved:** `{saved_script['name']}` ({saved_script['size_bytes']} bytes)")

    if stdout:
        sections.append(f"**Output (stdout):**\n```text\n{stdout}\n```")

    if stderr:
        sections.append(f"**Stderr / Diagnostics:**\n```text\n{stderr}\n```")

    if not stdout and not stderr and success:
        sections.append("*Script completed cleanly with no console output.*")

    image_artifacts = [a for a in artifacts if a.get("type") == "image"]
    if image_artifacts:
        sections.append(f"**Generated Graphical Representation ({len(image_artifacts)}):**")
        for img in image_artifacts:
            # Embed image data url directly in markdown
            sections.append(f"![{img.get('name', 'plot')}]({img.get('data_url')})")

    file_artifacts = [a for a in artifacts if a.get("type") == "file"]
    if file_artifacts:
        sections.append(f"**Generated Data Files ({len(file_artifacts)}):**")
        for f in file_artifacts:
            preview = f.get("preview", "")
            if preview:
                sections.append(f"- `{f['name']}` ({f['size_bytes']} bytes):\n```text\n{preview.strip()}\n```")
            else:
                sections.append(f"- `{f['name']}` ({f['size_bytes']} bytes)")

    return "\n\n".join(sections)

