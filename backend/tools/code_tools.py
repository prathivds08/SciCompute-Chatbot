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
    # 1. Fix read(..., index=":") ONLY when the assigned variable is not iterated over
    #    or has Atoms methods called directly on the list variable itself.
    for match in list(re.finditer(r'(\b(\w+)\s*=\s*(?:ase\.io\.)?read\([^)]*?)index\s*=\s*[\'"]:\'"]([^)]*\))', code)):
        full_expr, prefix, var_name, suffix = match.group(0), match.group(1), match.group(2), match.group(3)
        is_iterated = bool(re.search(rf'\bfor\s+[\w\s,]+\s+in\s+(?:enumerate\s*\(\s*)?{re.escape(var_name)}\b', code))
        direct_call_on_list = bool(
            re.search(
                rf'\b{re.escape(var_name)}\.(?:get_positions|get_scaled_positions|cell|get_cell|get_potential_energy|get_forces|get_chemical_symbols|get_all_distances|get_distances|positions|arrays|info)\b',
                code,
            )
        )
        if direct_call_on_list or not is_iterated:
            fixed_expr = f"{prefix}index=0{suffix}"
            code = code.replace(full_expr, fixed_expr, 1)

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
    """Strip internal <think> blocks and sanitize Python code blocks inside markdown text."""
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)

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
            # If language is 'code' or empty but content has python indicators, classify as python
            if clean_lang in ("python", "py"):
                clean_code = sanitize_scientific_code(clean_code)
            elif clean_lang in ("code", "") and any(
                clean_code.startswith(k) or f"\n{k}" in clean_code
                for k in ("import ", "from ", "def ", "print(", "class ")
            ):
                clean_lang = "python"
                clean_code = sanitize_scientific_code(clean_code)
            blocks.append({"language": clean_lang, "code": clean_code})
    return blocks


def extract_docker_tool_calls(text: str) -> list[str]:
    """Extract <docker_run>...</docker_run> or <call_docker>...</call_docker> blocks from LLM responses."""
    pattern = r"<(?:docker_run|docker_exec|run_docker|call_docker|docker)>([\s\S]*?)</(?:docker_run|docker_exec|run_docker|call_docker|docker)>"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    calls = []
    for raw in matches:
        clean = re.sub(r"^```(?:python|py)?\r?\n", "", raw.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"\r?\n```$", "", clean)
        clean = sanitize_scientific_code(clean.strip())
        if clean:
            calls.append(clean)
    return calls


def detect_docker_intent(text: str) -> bool:
    """Detect if the user query requests running, executing, calculating, or analyzing in Docker."""
    clean_text = text.strip()
    txt = clean_text.lower()

    # 1. Container keywords and synonyms
    container_synonyms = [
        "docker", "container", "sandbox", "dockers", "containers", "sandboxes",
    ]
    has_container = any(c in txt for c in container_synonyms)

    # 2. Execution action verbs and synonyms
    exec_verbs = [
        "run", "running", "execute", "executing", "exec", "execution",
        "eval", "evaluate", "evaluating", "evaluation",
        "launch", "launching", "trigger", "triggering",
        "compute", "computing", "computation",
        "calculate", "calculating", "calculation",
        "simulate", "simulating", "simulation",
        "test", "testing", "test-run", "test run",
        "perform", "process", "plot", "plotting",
        "generate", "generating",
    ]

    # If the user explicitly mentions both an execution verb and a container synonym
    if has_container and any(v in txt for v in exec_verbs):
        return True

    # 3. Direct container phrasing patterns
    docker_phrases = [
        "in docker", "inside docker", "using docker", "use docker", "call docker",
        "via docker", "on docker", "through docker", "with docker", "docker tool",
        "docker tools", "docker run", "docker exec", "run in container", "inside container",
        "in the container", "on the container", "via container", "using container",
        "in sandbox", "inside sandbox", "in the sandbox", "with container",
        "to docker", "by docker", "within docker", "within container",
    ]
    if any(p in txt for p in docker_phrases):
        return True

    # 4. Explicit action phrases requesting execution of code, script, or computation
    exec_phrases = [
        "run the code", "run this code", "run that code", "run code",
        "execute the code", "execute this code", "execute that code", "execute code",
        "run it", "run this", "run that", "execute it", "execute this", "execute that",
        "run the script", "run this script", "run script", "execute the script", "execute script",
        "run by yourself", "run by itself", "execute by itself", "run and give", "execute and give",
        "run and return", "execute and return", "run and show", "execute and show",
        "run and print", "execute and print", "run and display", "execute and display",
        "return the output", "return output", "show the output", "show output", "give the output",
        "give output", "print the output", "print output", "display the output",
        "run python", "execute python", "eval python", "evaluate python",
        "calculate and run", "compute and run", "test and run", "give me the result by running",
        "water density profile", "density profile", "radial distribution", "rdf",
        "survival probability", "hydroxylation", "boundary atoms", "find the total force",
        "calculate the force", "compute the force", "convert to raw", "raw_to_set",
        "generate plot", "plot the",
    ]
    if any(p in txt for p in exec_phrases):
        return True

    # 5. Short imperative execution commands (e.g. "run", "execute", "please run", "run it now")
    words = [w.strip(".,!?:;\"'()[]{}") for w in txt.split()]
    verb_hits = set(words).intersection({
        "run", "execute", "exec", "eval", "evaluate", "launch", "simulate", "calculate", "compute", "test"
    })
    if verb_hits and len(words) <= 12:
        return True

    return False


def is_followup_execution_request(text: str) -> bool:
    """Detect if the user prompt is a follow-up command requesting execution of existing code.

    Matches commands like:
      - "run", "execute", "exec"
      - "run in docker", "execute in docker", "run it in docker and return the output"
      - "can you run this in docker?", "please execute the code"
      - "run the above script", "now run it"
    """
    clean_text = text.strip()
    # If the user provides a full markdown code block in this message, it is handled as a direct code submission
    if "```" in clean_text:
        return False

    txt = clean_text.lower()
    # Strip common conversational preambles
    stripped = re.sub(
        r"^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|now\s+|kindly\s+|hey\s+|ok\s+|okay\s+|just\s+)+",
        "",
        txt,
    ).strip()
    stripped = stripped.rstrip(".,!?:;\"' ")

    # Direct short command triggers
    direct_commands = {
        "run", "run it", "run this", "run that", "run code", "run the code", "run this code",
        "run that code", "run script", "run the script", "run this script", "run that script",
        "run above", "run above code", "run the above code", "run above script", "run previous code",
        "execute", "execute it", "execute this", "execute that", "execute code", "execute the code",
        "execute this code", "execute that code", "execute script", "execute the script",
        "execute this script", "execute above", "execute above code", "execute the above code",
        "exec", "exec it", "exec this", "exec code", "exec script",
        "eval", "eval it", "eval this", "evaluate", "evaluate it", "evaluate this",
        "launch", "launch it", "launch this", "start", "start execution",
        "run in docker", "run it in docker", "run this in docker", "run that in docker",
        "run the code in docker", "run code in docker", "run script in docker", "run the script in docker",
        "run in container", "run it in container", "run this in container", "run inside docker",
        "execute in docker", "execute it in docker", "execute this in docker", "execute that in docker",
        "execute the code in docker", "execute code in docker", "execute script in docker",
        "execute in container", "execute it in container", "execute inside docker",
        "docker run", "docker exec", "call docker", "use docker", "run container", "execute container",
        "run and return output", "execute and return output", "run and give output", "execute and give output",
        "run it and show output", "execute it and show output",
    }
    if stripped in direct_commands:
        return True

    # Regex pattern for conversational execution requests
    verbs = r"(?:run|execute|exec|eval|evaluate|launch|simulate|test|start)"
    targets = r"(?:it|this|that|code|the\s+code|this\s+code|that\s+code|script|the\s+script|this\s+script|python|the\s+above\s+code|the\s+previous\s+code|above|previous)?"
    containers = r"(?:in\s+docker|in\s+the\s+docker|inside\s+docker|on\s+docker|via\s+docker|using\s+docker|with\s+docker|in\s+container|in\s+the\s+container|inside\s+container|in\s+sandbox|inside\s+sandbox)?"
    outputs = r"(?:and\s+return(?:\s+the)?\s+output|and\s+show(?:\s+the)?\s+output|and\s+give(?:\s+the)?\s+output|and\s+print(?:\s+the)?\s+output|and\s+display(?:\s+the)?\s+output|and\s+return(?:\s+the)?\s+results?|and\s+show(?:\s+the)?\s+results?)?"

    pattern = rf"^\s*{verbs}(?:\s+{targets})?(?:\s+{containers})?(?:\s+{outputs})?\s*$"
    if re.match(pattern, stripped):
        return True

    # Sentence check: <= 16 words, containing an execution verb and a target or container reference
    words = [w.strip(".,!?:;\"'()[]{}") for w in txt.split()]
    if len(words) <= 16:
        has_verb = bool(set(words).intersection({
            "run", "execute", "exec", "eval", "evaluate", "launch", "simulate", "test"
        }))
        has_target_or_container = bool(set(words).intersection({
            "it", "this", "that", "code", "script", "above", "previous",
            "docker", "container", "sandbox", "output", "results", "result"
        }))
        if has_verb and has_target_or_container:
            return True

    return False


def resolve_fallback_analysis_script(user_query: str) -> str | None:
    """Return a verified DeePMD analysis script template when execution is requested without LLM code."""
    txt = user_query.lower()
    if "density" in txt or "water density" in txt or "rho(z)" in txt:
        return build_water_density_profile_script()
    if "rdf" in txt or "radial distribution" in txt:
        return build_rdf_script()
    if "convert to raw" in txt or "raw_to_set" in txt or "raw format" in txt:
        return build_extxyz_to_deepmd_raw_script()
    if "loss curve" in txt or "training loss" in txt:
        return build_loss_curve_script()
    if "parity" in txt or "parity plot" in txt or "force parity" in txt:
        return build_parity_plot_script()
    return None


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

