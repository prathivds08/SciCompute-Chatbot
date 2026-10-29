"""
FastAPI application — main entry point for the Scientific Computing Chatbox.

Provides:
  - POST /api/chat         — send a message, get an agent response
  - GET  /api/sessions     — list all chat sessions
  - GET  /api/sessions/{id}/messages — get messages for a session
  - GET  /api/jobs         — list tracked SLURM / simulation jobs
  - WS   /ws/chat          — WebSocket for streaming chat responses
  - POST /api/init         — (re-)initialise the RAG knowledge base

Lifespan handler bootstraps Ollama, ChromaDB, Kùzu, and SQLite on startup.
"""

import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agents.baseline_agent import BaselineAgent
from backend.config import settings
from backend.db.models import Database
from backend.rag.embeddings import EmbeddingModel
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.knowledge_graph import KnowledgeGraph
from backend.rag.vectorstore import VectorStore
from backend.sandbox.docker_runner import DockerRunner
from backend.tools.deepmd_analysis_tools import (
    save_analysis_script,
    build_water_density_profile_script,
    build_rdf_script,
    build_extxyz_to_deepmd_raw_script,
    build_loss_curve_script,
    build_parity_plot_script,
)
from backend.tools import (
    execute_code,
    extract_code_blocks,
    extract_docker_tool_calls,
    detect_docker_intent,
    is_followup_execution_request,
    resolve_fallback_analysis_script,
    format_execution_result_markdown,
    sanitize_markdown_text,
    web_search,
    aweb_search,
    fetch_web_page,
    afetch_web_page,
    format_web_search_markdown,
    format_web_page_markdown,
    extract_web_tool_calls,
    detect_web_intent,
)

# ------------------------------------------------------------------ #
#  Global singletons (initialised during lifespan)
# ------------------------------------------------------------------ #

db: Database | None = None
vectorstore: VectorStore | None = None
knowledge_graph: KnowledgeGraph | None = None
retriever: HybridRetriever | None = None
baseline_agent: BaselineAgent = BaselineAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global db, vectorstore, knowledge_graph, retriever, baseline_agent  # noqa: PLW0603

    print("[startup] Initialising database ...")
    db = Database().connect()

    print("[startup] Initialising embedding model ...")
    emb = EmbeddingModel()

    print("[startup] Initialising ChromaDB vector store ...")
    vectorstore = VectorStore(embedding_model=emb)

    print("[startup] Initialising Kuzu knowledge graph ...")
    knowledge_graph = KnowledgeGraph()

    print("[startup] Populating domain knowledge ...")
    counts = knowledge_graph.populate_domain_knowledge()
    print(f"[startup]   -> KG entities: {counts}")

    print("[startup] Ingesting project READMEs ...")
    ingest_results = vectorstore.ingest_project_readmes()
    print(f"[startup]   -> Ingested: {ingest_results}")

    print("[startup] Initialising hybrid retriever ...")
    retriever = HybridRetriever(vectorstore, knowledge_graph, db)

    print("[startup] Initialising zero-shot baseline agent ...")
    baseline_agent = BaselineAgent()

    print(f"[startup] OK Backend ready - model={settings.MODEL_NAME}, "
          f"vectorDB chunks={vectorstore.count}")

    yield  # app runs

    print("[shutdown] Closing connections ...")
    if db:
        db.close()
    if knowledge_graph:
        knowledge_graph.close()


# ------------------------------------------------------------------ #
#  App
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Scientific Computing Chatbox",
    description="AI-powered assistant for DFT, DeePMD Training, LAMMPS MD, and Analysis workflows",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Request / response models
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    mode: str = "rag"  # "rag" (default with hybrid retrieval) or "baseline" (zero-shot, no RAG)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    agent_used: str | None = None
    tool_results: list[dict[str, Any]] | None = None


class ExecuteCodeRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int | None = None
    session_id: str | None = None
    save_script_name: str | None = None


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebPageRequest(BaseModel):
    url: str
    max_chars: int = 4000


class CreateAnalysisScriptRequest(BaseModel):
    session_id: str | None = None
    analysis_type: str  # "density", "rdf", "raw_conversion", "loss_curve", "parity", "custom"
    extxyz_filename: str = "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    script_name: str | None = None
    custom_code: str | None = None


class RunAnalysisRequest(BaseModel):
    session_id: str | None = None
    script_path: str | None = None
    code: str | None = None
    timeout: int | None = None


class DockerStatusResponse(BaseModel):
    docker_available: bool
    docker_message: str
    image_available: bool
    image_message: str
    image_name: str
    require_docker: bool


class InitResponse(BaseModel):
    vectordb_chunks: int
    kg_entities: dict[str, int]
    status: str


# ------------------------------------------------------------------ #
#  Execution Intent Helpers
# ------------------------------------------------------------------ #

def _is_execution_request(text: str) -> bool:
    """Check if the user prompt requests running or executing code in Docker."""
    return detect_docker_intent(text)


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #

@app.get("/api/status")
async def api_status():
    return {
        "service": "Scientific Computing Chatbox",
        "status": "running",
        "model": settings.MODEL_NAME,
        "vectordb_chunks": vectorstore.count if vectorstore else 0,
    }


@app.post("/api/sandbox/execute")
async def execute_code_endpoint(req: ExecuteCodeRequest):
    """Directly execute a code snippet in the sandbox and return stdout, stderr, and plots."""
    result = execute_code(
        code=req.code,
        language=req.language,
        timeout=req.timeout,
        session_id=req.session_id,
        save_script_name=req.save_script_name,
    )
    return result


@app.get("/api/docker/status", response_model=DockerStatusResponse)
async def docker_status_endpoint():
    """Inspect Docker daemon readiness, target sandbox image, and isolation policies."""
    runner = DockerRunner()
    avail, msg = runner.is_docker_available()
    img_avail, img_msg = runner.is_image_available() if avail else (False, "Docker daemon not reachable")
    return DockerStatusResponse(
        docker_available=avail,
        docker_message=msg,
        image_available=img_avail,
        image_message=img_msg,
        image_name=runner.image_name,
        require_docker=settings.REQUIRE_DOCKER,
    )


@app.post("/api/deepmd/create-script")
async def create_deepmd_script_endpoint(req: CreateAnalysisScriptRequest):
    """Generate and persist a DeePMD analysis Python script to disk and register it in SQLite."""
    sid = req.session_id or "default"
    atype = req.analysis_type.lower()

    if atype in ("density", "water_density", "density_profile"):
        code = build_water_density_profile_script(req.extxyz_filename)
        default_name = f"water_density_{req.extxyz_filename.replace('.extxyz', '')}.py"
        desc = f"Interfacial water density profile for {req.extxyz_filename}"
    elif atype in ("rdf", "radial_distribution"):
        code = build_rdf_script(req.extxyz_filename)
        default_name = f"rdf_analysis_{req.extxyz_filename.replace('.extxyz', '')}.py"
        desc = f"Radial distribution function analysis for {req.extxyz_filename}"
    elif atype in ("raw_conversion", "raw", "deepmd_raw"):
        code = build_extxyz_to_deepmd_raw_script(req.extxyz_filename)
        default_name = f"convert_raw_{req.extxyz_filename.replace('.extxyz', '')}.py"
        desc = f"DeePMD raw and set.000 conversion for {req.extxyz_filename}"
    elif atype in ("loss_curve", "lcurve", "loss"):
        code = build_loss_curve_script()
        default_name = "deepmd_loss_curve_analysis.py"
        desc = "DeePMD training loss and RMSE convergence analyzer"
    elif atype in ("parity", "parity_plot"):
        code = build_parity_plot_script(req.extxyz_filename)
        default_name = f"parity_plot_{req.extxyz_filename.replace('.extxyz', '')}.py"
        desc = f"DeePMD potential parity plot for {req.extxyz_filename}"
    elif atype == "custom" and req.custom_code:
        code = req.custom_code
        default_name = req.script_name or f"custom_deepmd_analysis_{int(time.time())}.py"
        desc = "Custom user/agent DeePMD analysis script"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported analysis type: '{req.analysis_type}'. Supported: density, rdf, raw_conversion, loss_curve, parity, custom.")

    sname = req.script_name or default_name
    script_meta = save_analysis_script(
        code=code,
        script_name=sname,
        session_id=sid,
        description=desc,
    )
    return {
        "success": True,
        "script": script_meta,
        "code": code,
    }


@app.post("/api/deepmd/run-analysis")
async def run_deepmd_analysis_endpoint(req: RunAnalysisRequest):
    """Execute a DeePMD analysis script strictly inside the isolated Docker container."""
    sid = Path(req.session_id or "default").name
    code = req.code

    if not code and req.script_path:
        p = Path(req.script_path).resolve()
        allowed_root = settings.SCRIPTS_DIR.resolve()
        if not p.is_relative_to(allowed_root) or p.suffix != ".py":
            raise HTTPException(status_code=403, detail="Access denied: script_path must reside inside SCRIPTS_DIR.")
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Script file not found: {req.script_path}")
        code = p.read_text(encoding="utf-8")

    if not code:
        raise HTTPException(status_code=400, detail="Must provide either 'code' or valid 'script_path'.")

    result = execute_code(
        code=code,
        language="python",
        timeout=req.timeout,
        session_id=sid,
        save_script_name=Path(req.script_path).name if req.script_path else None,
    )
    return result


@app.get("/api/sandbox/scripts")
async def list_sandbox_scripts(session_id: str | None = None):
    """List all saved Python analysis scripts."""
    base_dir = settings.SCRIPTS_DIR
    if not base_dir.exists():
        return []

    scripts = []
    safe_sid = Path(session_id).name if session_id else None
    target_dirs = [base_dir / safe_sid] if safe_sid and (base_dir / safe_sid).exists() else [d for d in base_dir.iterdir() if d.is_dir()]
    for sdir in target_dirs:
        for f in sdir.glob("*.py"):
            scripts.append({
                "session_id": sdir.name,
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "modified_at": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(),
            })
    return sorted(scripts, key=lambda x: x["modified_at"], reverse=True)



@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a user message and get an agent response.

    Supports mode='rag' (retrieval-augmented) and mode='baseline' (zero-shot, no RAG).
    Automatically detects requests to execute code and returns live simulation/computation results.
    """
    assert db is not None

    # Ensure session exists
    sid = req.session_id or str(uuid.uuid4())
    if not db.get_session(sid):
        db.create_session(sid, title=req.message[:60])

    # Case 1: Check if the user directly provided a code block in their prompt to execute
    user_blocks = extract_code_blocks(req.message)
    executable_user_blocks = [b for b in user_blocks if b["language"] in ("python", "py", "code")]
    if executable_user_blocks and detect_docker_intent(req.message):
        target_code = executable_user_blocks[0]["code"]
        db.add_message(sid, "user", req.message)
        exec_res = execute_code(
            code=target_code,
            language="python",
            session_id=sid,
            save_script_name=f"user_analysis_{int(time.time())}.py",
        )
        formatted_res = format_execution_result_markdown(exec_res)
        reply_text = "Executed the user-provided Python script in the isolated Docker container:\n\n" + formatted_res
        docker_tool_item = {
            "tool": "docker_run",
            "success": exec_res.get("success", False),
            "duration": exec_res.get("duration", 0.0),
            "returncode": exec_res.get("returncode", 0),
            "stdout": exec_res.get("stdout", ""),
            "stderr": exec_res.get("stderr", ""),
            "artifacts": exec_res.get("artifacts", []),
            "formatted": formatted_res,
        }
        db.add_message(sid, "assistant", reply_text, agent_used="docker_executor", tool_calls=[docker_tool_item])
        return ChatResponse(
            session_id=sid,
            reply=reply_text,
            agent_used="docker_executor",
            tool_results=[docker_tool_item],
        )

    # Case 2: Check if this is a follow-up command to run code from the conversation history
    if is_followup_execution_request(req.message):
        past_msgs = db.get_messages(sid)
        target_block = None
        # Scan backward for the most recent message with an executable Python code block
        for pm in reversed(past_msgs):
            blocks = extract_code_blocks(pm.get("content", ""))
            executable = [b for b in blocks if b["language"] in ("python", "py", "code")]
            if executable:
                target_block = executable[0]
                break

        # Fallback to standard template if no code in history but specific analysis requested
        if not target_block:
            fallback_template = resolve_fallback_analysis_script(req.message)
            if fallback_template:
                target_block = {"language": "python", "code": fallback_template}

        if target_block:
            db.add_message(sid, "user", req.message)
            exec_res = execute_code(
                code=target_block["code"],
                language="python",
                session_id=sid,
                save_script_name=f"deepmd_analysis_{int(time.time())}.py",
            )
            formatted_res = format_execution_result_markdown(exec_res)
            reply_text = f"Running Python code in the isolated Docker container:\n\n" + formatted_res
            docker_tool_item = {
                "tool": "docker_run",
                "success": exec_res.get("success", False),
                "duration": exec_res.get("duration", 0.0),
                "returncode": exec_res.get("returncode", 0),
                "stdout": exec_res.get("stdout", ""),
                "stderr": exec_res.get("stderr", ""),
                "artifacts": exec_res.get("artifacts", []),
                "formatted": formatted_res,
            }
            db.add_message(sid, "assistant", reply_text, agent_used="docker_executor", tool_calls=[docker_tool_item])
            return ChatResponse(
                session_id=sid,
                reply=reply_text,
                agent_used="docker_executor",
                tool_results=[docker_tool_item],
            )

    wants_execution = _is_execution_request(req.message)

    # Retrieve prior conversation history (excluding the current uncommitted turn)
    past_msgs = db.get_messages(sid)
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in past_msgs
        if m.get("role") in ("user", "assistant")
    ]

    # 1. Zero-shot Baseline mode (No RAG access, but with Live Web Tools & Docker)
    if req.mode == "baseline":
        assert baseline_agent is not None
        try:
            reply_text, web_tool_results, agent_used = await baseline_agent.agenerate_response_with_tools(
                req.message,
                conversation_history=conversation_history,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama LLM service error: {exc}",
            ) from exc

        reply_text = sanitize_markdown_text(reply_text)
        tool_results = list(web_tool_results) if web_tool_results else None

        docker_calls = extract_docker_tool_calls(reply_text)
        if not docker_calls and wants_execution:
            blocks = extract_code_blocks(reply_text)
            executable_blocks = [b for b in blocks if b["language"] in ("python", "py", "code")]
            if executable_blocks:
                docker_calls = [executable_blocks[0]["code"]]
            else:
                fallback_template = resolve_fallback_analysis_script(req.message)
                if fallback_template:
                    docker_calls = [fallback_template]

        if docker_calls:
            agent_used = "docker_executor"
            for code_snippet in docker_calls:
                exec_res = execute_code(
                    code=code_snippet,
                    language="python",
                    session_id=sid,
                    save_script_name=f"baseline_analysis_{int(time.time())}.py",
                )
                formatted_res = format_execution_result_markdown(exec_res)
                reply_text += "\n\n---\n" + formatted_res
                if tool_results is None:
                    tool_results = []
                tool_results.append({
                    "tool": "docker_run",
                    "success": exec_res.get("success", False),
                    "duration": exec_res.get("duration", 0.0),
                    "returncode": exec_res.get("returncode", 0),
                    "stdout": exec_res.get("stdout", ""),
                    "stderr": exec_res.get("stderr", ""),
                    "artifacts": exec_res.get("artifacts", []),
                    "formatted": formatted_res,
                })

        db.add_message(sid, "user", req.message)
        db.add_message(sid, "assistant", reply_text, agent_used=agent_used, tool_calls=tool_results)
        return ChatResponse(
            session_id=sid,
            reply=reply_text,
            agent_used=agent_used,
            tool_results=tool_results,
        )

    # 2. RAG mode (VectorStore + Knowledge Graph context + Live Web Tools)
    assert retriever is not None
    context = retriever.retrieve(req.message)
    context_str = retriever.format_context_for_prompt(context)

    # Proactive web intent detection for RAG
    needs_web, web_action, web_target = detect_web_intent(req.message)
    web_tool_results: list[dict[str, Any]] = []
    web_context_blocks: list[str] = []

    if needs_web and web_target:
        if web_action == "search":
            search_res = await aweb_search(web_target, max_results=5)
            formatted_web = format_web_search_markdown(web_target, search_res)
            web_tool_results.append({
                "tool": "web_search",
                "query": web_target,
                "results": search_res,
                "formatted": formatted_web,
            })
            web_context_blocks.append(formatted_web)
        elif web_action == "fetch":
            page_res = await afetch_web_page(web_target, max_chars=4000)
            formatted_web = format_web_page_markdown(page_res)
            web_tool_results.append({
                "tool": "fetch_url",
                "url": web_target,
                "page": page_res,
                "formatted": formatted_web,
            })
            web_context_blocks.append(formatted_web)

    web_tools_guidelines = (
        "\nLIVE WEB TOOLS & INTERNET ACCESS:\n"
        "You have real-time access to the live web and online documentation:\n"
        "- Web search: emit `<web_search>query</web_search>` to look up live documentation, releases, or papers.\n"
        "- Webpage fetch: emit `<fetch_url>https://example.com</fetch_url>` to read text from a URL.\n"
        "When <live_web_context> is provided, synthesize its information directly into your answer."
    )

    docker_tools_guidelines = (
        "\nLIVE DOCKER CONTAINER EXECUTION TOOL:\n"
        "You have access to an isolated, secure Docker execution container (`deepmd-analysis-sandbox`).\n"
        "The structure dataset is mounted read-only at `/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` "
        "(or `DATA_DIR / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'`).\n\n"
        "To execute scientific analysis, compute interfacial properties (forces, density profiles, RDF, "
        "DeePMD dataset conversions, etc.), or generate graphical plots:\n"
        "Emit your complete Python script inside `<docker_run>...</docker_run>` tags:\n"
        "<docker_run>\n"
        "# Complete, self-contained Python script\n"
        "import os, sys, math\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from ase.io import read\n"
        "...\n"
        "</docker_run>\n\n"
        "The backend will execute your script in the secure Docker container, capture stdout and generated figures, "
        "and return the live execution output."
    )

    data_isolation_policy = (
        "\nDATA PRIVACY & ISOLATION POLICY:\n"
        "- You do NOT have direct access to the raw coordinates, forces, or internal data of 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'.\n"
        "- Never attempt to hallucinate, dump, or display raw atomic coordinate tables in text responses.\n"
        "- The structure dataset is mounted read-only inside the isolated Docker sandbox at '/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz' (or DATA_DIR / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz').\n"
        "- When analysis is requested, write a self-contained Python script to load and compute on the file inside Docker, printing summary metrics and generating figures.\n"
    )

    execution_instruction = ""
    if wants_execution:
        execution_instruction = (
            "\nCRITICAL DIRECTIVE -- USER REQUESTED CODE EXECUTION IN DOCKER:\n"
            "The user explicitly used an execution keyword (e.g. 'run', 'execute', 'calculate', 'compute', 'simulate') in Docker.\n"
            "You MUST NOT just explain theory or output an unexecutable snippet.\n"
            "You MUST output a complete, executable Python script inside a `<docker_run>...</docker_run>` block (or ```python ... ``` block).\n"
            "- List 'Required Imports:' before opening the block.\n"
            "- Start with the Mandatory Standard Scientific Prelude.\n"
            "- For interfacial and DeePMD analysis on Anatase TiO2, call the structure file at "
            "'/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz' (or DATA_DIR / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz').\n"
            "- Always use print() statements to report key numerical metrics and findings to stdout.\n"
            "- If graphical representations or plots are requested, use matplotlib.pyplot to save figures to 'figure_output.png'.\n"
            "The backend will automatically execute your script inside the isolated Docker container and return the live execution outputs.\n"
        )

    ase_guidelines = (
        "\nASE CODING RULES -- MANDATORY SPOONFED TEMPLATE (CRITICAL API RETURN TYPES):\n"
        "---------------------------------------------------------------------------------\n"
        "1. STRUCTURE I/O (from ase.io import read, write):\n"
        "   - ALWAYS use index=0 to load a single Atoms object: atoms = read(str(file_path), index=0)\n"
        "   - NEVER pass index=':' when inspecting or analyzing a structure! 'read(path, index=\":\")' returns list[Atoms], NOT an Atoms object.\n"
        "     Calling atoms.get_positions() on a list crashes with: AttributeError: 'list' object has no attribute 'get_positions'.\n"
        "   - To write/save structures, construct a new `ase.Atoms` object and call `write()`:\n"
        "     from ase import Atoms\n"
        "     from ase.io import write\n"
        "     new_atoms = Atoms(symbols=symbols, positions=positions, cell=atoms.cell, pbc=atoms.pbc)\n"
        "     write('boundary_atoms.extxyz', new_atoms)\n"
        "   - FATAL ERROR: The variable `atoms` returned by read() is an instance. NEVER call `atoms(...)` (TypeError: 'Atoms' object is not callable).\n"
        "   - ALWAYS use len(atoms) for atom count (NEVER atoms.get_count(), atoms.num_atoms(), or atoms.count()).\n"
        "2. DISTANCES (from ase.geometry import get_distances):\n"
        "   - Exact Signature: disp_vecs, dist_matrix = get_distances(p1, p2=None, cell=atoms.cell, pbc=atoms.pbc)\n"
        "   - FATAL ERROR: NEVER pass 'mic=True' to get_distances()! It raises TypeError: get_distances() got an unexpected keyword argument 'mic'.\n"
        "     Minimum Image Convention is automatically applied when cell and pbc are supplied.\n"
        "   - Return Value: get_distances() returns a 2-tuple (displacement_vectors, distances). Extract scalar distances with:\n"
        "     dist_matrix = get_distances(p1, p2, cell=atoms.cell, pbc=atoms.pbc)[1]\n"
        "   - For all-pairs distance on Atoms: dists = atoms.get_all_distances(mic=True) (directly returns (N, N) array).\n"
        "3. NEIGHBOR LISTS (from ase.neighborlist import neighbor_list):\n"
        "   - Exact Signature: i_idx, j_idx, d_ij = neighbor_list('ijd', atoms, cutoff=...)\n"
        "   - Pairwise cutoffs dict MUST include both directions: {('O', 'H'): 1.20, ('H', 'O'): 1.20, ('Ti', 'O'): 2.50, ('O', 'Ti'): 2.50}.\n"
        "   - Separate water O from lattice O: water oxygens have covalent O-H bonds (< 1.20 A):\n"
        "     i_oh, _, _ = neighbor_list('ijd', atoms, {('O', 'H'): 1.20, ('H', 'O'): 1.20})\n"
        "     water_o_idx = np.unique(i_oh[symbols[i_oh] == 'O'])\n"
        "     lattice_o_idx = np.setdiff1d(np.where(symbols == 'O')[0], water_o_idx)\n"
        "4. UNITS & CONVERSIONS (from ase.units import Hartree):\n"
        "   - Potential energy in Hartree: energy_Ha = atoms.get_potential_energy() / Hartree\n"
        "   - Forces in Hartree/Bohr: forces_Ha_Bohr = atoms.get_forces() / (Hartree / Bohr)\n"
        "   - FATAL ERROR: Never import Avogadro from ase.units (does NOT exist; use NA = 6.02214076e23).\n"
    )

    numpy_guidelines = (
        "\nNUMPY ARRAY COMPARISON & DISTANCE RULES (CRITICAL - PREVENT RUNTIME ERRORS):\n"
        "- NEVER evaluate NumPy array expressions in Python boolean contexts (e.g. 'if any(distance < 1e-5 for distance in distances):' or 'if arr < 1e-5:').\n"
        "  This causes: 'ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()'.\n"
        "- ALWAYS use NumPy vectorized checks: np.any(distances < 1e-5) or (distances < 1e-5).any() or np.all(...).\n"
        "- For pairwise distances with PBC: dists = atoms.get_all_distances(mic=True); np.fill_diagonal(dists, np.inf); has_overlap = bool(np.any(dists < 1e-5)).\n"
        "- For neighbor searching: from ase.neighborlist import neighbor_list; idx_i, idx_j, d_ij = neighbor_list('ijd', atoms, cutoff=3.0).\n"
    )

    anatase_physics_and_models = (
        "\nANATASE TiO2 / WATER INTERFACE -- PHYSICS & APPROPRIATE MODELS:\n"
        "1. Surface Physics, Coordination & Adsorption Chemistry:\n"
        "   - Anatase TiO2 (101) is the lowest surface-energy, thermodynamically dominant facet (~0.44 J/m2).\n"
        "   - Surface Coordination: 5-fold coordinated Ti (Ti_5c, Lewis acid site), 6-fold bulk-like Ti (Ti_6c), 2-fold bridging oxygen (O_2c / O_br, Lewis base & H-bond acceptor), 3-fold lattice oxygen (O_3c).\n"
        "   - Water Adsorption Mode: On stoichiometric pristine Anatase (101), water adsorbs predominantly MOLECULARLY (intact H2O) onto Ti_5c sites (d(Ti_5c - Ow) ~ 2.15-2.25 A, E_ads ~ -0.7 to -0.9 eV/molecule). Water protons Hw form directed hydrogen bonds with adjacent bridging oxygens O_br (d(O_br ... Hw) ~ 1.75-1.95 A). Dissociation is endothermic on pristine (101) and occurs preferentially at oxygen vacancies (V_O) or step defects.\n"
        "   - Interfacial Layering & Density Profiles rho(z): 1st contact layer at z ~ 2.1-2.3 A above surface Ti plane; 2nd hydration layer at z ~ 3.8-4.2 A; bulk liquid water plateau at ~ 1.0 g/cm3 (~0.033 molecules/A^3).\n"
        "   - RDF Signatures: Ti_5c-Ow peak at 2.18-2.25 A; O_br-Hw interfacial H-bond peak at 1.75-1.95 A; Ow-Ow peak at 2.75-2.82 A; Ow-Hw intra-molecular peak at 0.98-1.02 A.\n"
        "2. Appropriate Models & Parameters:\n"
        "   - DeePMD-kit Potential: Use 'se_e2_a' (Smooth Edition 2-Body with Angular information, rc=6.0 A, rcs=0.5 A, neuron [25, 50, 100]) or 'DPA-2'. Strict type mapping: type_map = ['Ti', 'O', 'H'] (0: Ti, 1: O, 2: H). Loss weights: start with high force weight (pref_f=1000, pref_e=0.02) and decay to balanced weights (pref_f=1, pref_e=1). DP-GEN active learning model deviation thresholds: sigma_f < 0.05 eV/A (accurate), 0.05 <= sigma_f <= 0.15-0.20 eV/A (candidate for DFT), > 0.20 eV/A (discard).\n"
        "   - DFT Reference (Quantum ESPRESSO): PBE + Hubbard U with U_eff ~ 3.0-3.5 eV on Ti-3d states to correct self-interaction errors and properly position the band gap. Grimme DFT-D3/D4 or vdW functional (revPBE-D3) to capture liquid water dispersion. ecutwfc >= 60-85 Ry, ecutrho >= 480-680 Ry, >= 4x4x1 k-mesh, dipole correction along z.\n"
        "   - LAMMPS Molecular Dynamics: units metal, atom_style atomic, pair_style deepmd, timestep 0.5 fs (0.0005 ps) for H2O (1.0 fs for D2O), NVT Nosé-Hoover thermostat at 300 K (Tdamp = 0.05-0.1 ps).\n"
    )

    surface_boundary_rules = (
        "\nANATASE TiO2 SURFACE VS. SIMULATION BOX BOUNDARY RULES (WHY BOUNDARY ATOM COUNT IS WAY LESS):\n"
        "1. PHYSICAL SLAB BOUNDARY (SOLID-LIQUID INTERFACE FACING WATER):\n"
        "   - The number of atoms on the physical boundary is WAY LESS than 60 or 98!\n"
        "   - Surface Ti active sites (Ti_5c): ONLY 6 atoms on top surface, 6 atoms on bottom surface = 12 Ti atoms total!\n"
        "   - Surface bridging oxygens (O_2c): 6 top + 6 bottom = 12 O atoms.\n"
        "   - Total solid atoms at the physical surface boundary: ONLY 24 atoms (12 Ti + 12 O)!\n"
        "   - The 60 count represents the ENTIRE Titanium inventory of the crystal slab (all bulk interior layers + surfaces).\n"
        "2. GEOMETRIC SIMULATION BOX BOUNDARIES (PBC BOX FACES):\n"
        "   - In periodic slab simulations (e.g. c ~ 36.8 Angstrom), x and y have PBC representing an infinite continuous surface (NO physical walls in x and y).\n"
        "   - The previous '98 atoms' output was an artifact of using an arbitrary 5% window (which sweeps 27% of the entire 3D volume).\n"
        "   - When checking actual proximity to box faces with a realistic physical skin:\n"
        "     * Within 0.2 A of box faces: only ~4 to 6 atoms sit on boundary faces!\n"
        "     * Within 0.5 A of box faces: only ~12 to 18 atoms!\n"
        "   - Under any realistic physical metric, the number of atoms on the boundary is WAY LESS!\n"
        "3. VECTORIZED PBC DISTANCE CALCULATION:\n"
        "   - scaled = atoms.get_scaled_positions(wrap=True)\n"
        "   - dist_to_face = np.min(np.minimum(scaled, 1.0 - scaled) * atoms.cell.lengths(), axis=1)\n"
        "   - boundary_idx = np.where(dist_to_face <= cutoff_angstrom)[0]\n"
        "4. PRIVACY & CONSOLE HYGIENE:\n"
        "   - NEVER dump loops of hundreds of raw atomic coordinate lines. Print aggregate counts, species fractions, and boundary z-coordinates.\n"
    )

    code_generation_mandate = (
        "\nCODE GENERATION & MANDATORY IMPORT PRELUDE:\n"
        "- Before writing a ```python ... ``` block, state a 1-line 'Required Imports:' list naming every module and function used.\n"
        "- Every Python script MUST be 100% self-contained and MUST start with these explicit top-level imports:\n"
        "```python\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import math\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from ase.io import read, write\n"
        "from ase.geometry import get_distances\n"
        "from ase.neighborlist import neighbor_list, NeighborList\n"
        "from ase.units import Hartree, Bohr, mol, kJ, fs\n"
        "```\n"
        "- Structure reading MUST follow: 'atoms = read(str(DATA_PATH), index=0)' for single structures.\n"
        "- Dual-path fallback for dataset loading:\n"
        "    CONTAINER_PATH = Path('/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz')\n"
        "    LOCAL_PATH = Path(__file__).resolve().parent / 'data' / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'\n"
        "    DATA_PATH = CONTAINER_PATH if CONTAINER_PATH.exists() else LOCAL_PATH\n"
    )

    # ---- Upgrade 6: Dynamic intent-based system prompt modularisation ----
    # Only inject the heavyweight physics/boundary blocks when the query
    # is actually about Anatase TiO2 / interfacial analysis. For DFT
    # convergence (Pt), LAMMPS batch scripts, or training questions,
    # those ~900 tokens of Anatase context actively hurt qwen2.5-coder:7b
    # by confusing Ti/IrO2 parameters with Pt or Water ones.
    q_low = req.message.lower()
    is_anatase_or_interface = any(
        k in q_low for k in (
            "anatase", "tio2", "titanium", "extxyz", "boundary", "surface",
            "slab", "hydration", "density", "rdf", "radial", "survival",
            "hydroxyl", "interfacial", "water interface",
        )
    )
    is_code_or_ase = wants_execution or any(
        k in q_low for k in (
            "code", "script", "python", "ase", "numpy", "plot", "read",
            "extxyz", "cif", "calculate", "compute", "write", "generate",
        )
    )

    prompt_sections: list[str] = [
        (
            "You are a scientific computing assistant specialising in DFT (Quantum ESPRESSO), "
            "DeePMD-kit model training, LAMMPS molecular dynamics, and post-processing analysis "
            "of metal-oxide/water interfaces."
        ),
        data_isolation_policy,
        web_tools_guidelines,
        docker_tools_guidelines,
    ]

    if is_code_or_ase:
        prompt_sections.extend([ase_guidelines, numpy_guidelines, code_generation_mandate])
    if is_anatase_or_interface:
        prompt_sections.extend([surface_boundary_rules, anatase_physics_and_models])
    if wants_execution:
        prompt_sections.append(execution_instruction)

    if web_context_blocks:
        joined_web = "\n\n".join(web_context_blocks)
        prompt_sections.append(
            f"<live_web_context>\n{joined_web}\n</live_web_context>"
        )

    prompt_sections.append(
        "Prioritize the facts, parameters, and code patterns inside <retrieved_context>. "
        "When citing pseudopotentials, cutoffs, type_maps, or script templates, "
        "use the exact values from <retrieved_context>.\n\n"
        f"{context_str}"
    )

    system_prompt = "\n\n".join(prompt_sections)

    user_content = req.message
    if is_code_or_ase:
        user_content = (
            f"{req.message}\n\n"
            "[Mandatory Code Checklist:\n"
            "1. First list 'Required Imports:' before the code block.\n"
            "2. Start the ```python block with explicit imports: `import os, sys, json, math`, `from pathlib import Path`, "
            "`import numpy as np`, `import matplotlib.pyplot as plt`, `from ase.io import read, write`, "
            "`from ase.geometry import get_distances`, `from ase.neighborlist import neighbor_list`, `from ase.units import Hartree`.\n"
            "3. Use `read(str(DATA_PATH), index=0)` for single structures, `len(atoms)`, `np.any()`, and save figures to `figure_output.png`.]"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": user_content},
    ]

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            ollama_resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    # Upgrade 7: Tuned inference options for qwen2.5-coder:7b
                    # - temperature 0.1: eliminates stochastic hallucinations in grounded RAG QA
                    # - num_ctx 16384: covers system prompt + 5 chunks + history + full script
                    #   while halving KV-cache overhead vs 32768, improving time-to-first-token
                    # - top_p / top_k / repeat_penalty: standard quality settings
                    "options": {
                        "num_ctx": settings.LLM_NUM_CTX,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "top_k": 20,
                        "repeat_penalty": 1.05,
                    },
                },
            )
            if ollama_resp.status_code >= 400:
                import sys
                print(
                    f"[Chat API] Ollama returned {ollama_resp.status_code} for model '{settings.MODEL_NAME}':\n"
                    f"  Response: {ollama_resp.text[:500]}",
                    file=sys.stderr,
                )
            ollama_resp.raise_for_status()
            reply_text = ollama_resp.json()["message"]["content"]
            reply_text = sanitize_markdown_text(reply_text)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama LLM service error: {exc}",
        ) from exc

    agent_used = "rag_web_agent" if web_tool_results else "qa_rag"
    tool_results = list(web_tool_results) if web_tool_results else None

    # Check for reactive tool calls emitted by the model
    model_web_calls = extract_web_tool_calls(reply_text)
    if model_web_calls:
        agent_used = "rag_web_agent"
        new_web_outputs: list[str] = []
        for call in model_web_calls:
            t_type = call.get("tool")
            arg = call.get("argument", "")
            if t_type == "web_search":
                s_res = await aweb_search(arg, max_results=5)
                fmt = format_web_search_markdown(arg, s_res)
                if tool_results is None:
                    tool_results = []
                tool_results.append({
                    "tool": "web_search",
                    "query": arg,
                    "results": s_res,
                    "formatted": fmt,
                })
                new_web_outputs.append(fmt)
            elif t_type == "fetch_url":
                p_res = await afetch_web_page(arg, max_chars=4000)
                fmt = format_web_page_markdown(p_res)
                if tool_results is None:
                    tool_results = []
                tool_results.append({
                    "tool": "fetch_url",
                    "url": arg,
                    "page": p_res,
                    "formatted": fmt,
                })
                new_web_outputs.append(fmt)

        if new_web_outputs:
            web_followup_messages = [
                *messages,
                {"role": "assistant", "content": reply_text},
                {
                    "role": "user",
                    "content": (
                        "Live Web Tool Results:\n\n"
                        + "\n\n".join(new_web_outputs)
                        + "\n\nPlease provide your final answer to the user synthesizing these results. "
                        "Do not output raw tool tags."
                    ),
                },
            ]
            async with httpx.AsyncClient(timeout=None) as client:
                synth_resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": settings.MODEL_NAME,
                        "messages": web_followup_messages,
                        "stream": False,
                        "options": {
                            "num_ctx": settings.LLM_NUM_CTX,
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "top_k": 20,
                            "repeat_penalty": 1.05,
                        },
                    },
                )
                if synth_resp.status_code < 400:
                    reply_text = sanitize_markdown_text(
                        synth_resp.json()["message"]["content"]
                    )

    # Clean any residual tool tags
    reply_text = re.sub(
        r'<(?:web_search|fetch_url)>.*?</(?:web_search|fetch_url)>',
        '',
        reply_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    docker_tool_calls = extract_docker_tool_calls(reply_text)
    if not docker_tool_calls and wants_execution:
        blocks = extract_code_blocks(reply_text)
        executable_blocks = [b for b in blocks if b["language"] in ("python", "py", "code")]
        if executable_blocks:
            docker_tool_calls = [executable_blocks[0]["code"]]
        else:
            fallback_template = resolve_fallback_analysis_script(req.message)
            if fallback_template:
                docker_tool_calls = [fallback_template]

    if docker_tool_calls:
        agent_used = "docker_executor"
        for code_snippet in docker_tool_calls:
            exec_res = execute_code(
                code=code_snippet,
                language="python",
                session_id=sid,
                save_script_name=f"deepmd_analysis_{int(time.time())}.py",
            )

            # Upgrade 7: 1-step self-correction loop.
            # If the container execution failed with a real Python traceback
            # (not a policy-blocked execution), ask the model to repair its code once.
            if not exec_res["success"] and "EXECUTION BLOCKED" not in exec_res.get("stderr", ""):
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": reply_text},
                    {
                        "role": "user",
                        "content": (
                            "Your script failed in the Docker container with the following error:\n"
                            f"```text\n{exec_res['stderr'][:2000]}\n```\n"
                            "Provide the corrected, complete, self-contained Python script "
                            "in a <docker_run>...</docker_run> block."
                        ),
                    },
                ]
                async with httpx.AsyncClient(timeout=None) as repair_client:
                    repair_resp = await repair_client.post(
                        f"{settings.OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": settings.MODEL_NAME,
                            "messages": repair_messages,
                            "stream": False,
                            "options": {"num_ctx": settings.LLM_NUM_CTX, "temperature": 0.05},
                        },
                    )
                    if repair_resp.status_code < 400:
                        repaired_text = sanitize_markdown_text(
                            repair_resp.json()["message"]["content"]
                        )
                        repaired_docker = extract_docker_tool_calls(repaired_text)
                        if not repaired_docker:
                            repaired_blocks = [
                                b for b in extract_code_blocks(repaired_text)
                                if b["language"] in ("python", "py")
                            ]
                            if repaired_blocks:
                                repaired_docker = [repaired_blocks[0]["code"]]
                        if repaired_docker:
                            reply_text = repaired_text
                            exec_res = execute_code(
                                code=repaired_docker[0],
                                language="python",
                                session_id=sid,
                                save_script_name=f"deepmd_analysis_repaired_{int(time.time())}.py",
                            )
                            agent_used = "docker_executor"

            formatted_res = format_execution_result_markdown(exec_res)

            # If the model emitted an explicit <docker_run> tag and succeeded, synthesize the results
            has_explicit_docker_tag = bool(extract_docker_tool_calls(reply_text))
            if has_explicit_docker_tag and exec_res.get("success"):
                synth_messages = [
                    *messages,
                    {"role": "assistant", "content": reply_text},
                    {
                        "role": "user",
                        "content": (
                            "Live Docker Container Execution Results:\n\n"
                            f"Status: Succeeded ({exec_res.get('duration', 0.0)}s)\n"
                            f"Console Output (stdout):\n```text\n{exec_res.get('stdout', '')[:4000]}\n```\n"
                            + (f"Generated {len(exec_res.get('artifacts', []))} visual plots.\n" if exec_res.get("artifacts") else "")
                            + "\nPlease provide your final response to the user synthesizing these live computation results. "
                            "Report the exact numerical findings calculated inside Docker. "
                            "Do not output raw tool tags."
                        ),
                    },
                ]
                async with httpx.AsyncClient(timeout=None) as client:
                    synth_resp = await client.post(
                        f"{settings.OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": settings.MODEL_NAME,
                            "messages": synth_messages,
                            "stream": False,
                            "options": {
                                "num_ctx": settings.LLM_NUM_CTX,
                                "temperature": 0.1,
                            },
                        },
                    )
                    if synth_resp.status_code < 400:
                        reply_text = sanitize_markdown_text(synth_resp.json()["message"]["content"])

            # Clean any residual docker tool tags
            reply_text = re.sub(
                r'<(?:docker_run|docker_exec|run_docker|call_docker|docker)>[\s\S]*?</(?:docker_run|docker_exec|run_docker|call_docker|docker)>',
                '',
                reply_text,
                flags=re.IGNORECASE,
            ).strip()

            reply_text += "\n\n---\n" + formatted_res
            if tool_results is None:
                tool_results = []
            tool_results.append({
                "tool": "docker_run",
                "success": exec_res.get("success", False),
                "duration": exec_res.get("duration", 0.0),
                "returncode": exec_res.get("returncode", 0),
                "stdout": exec_res.get("stdout", ""),
                "stderr": exec_res.get("stderr", ""),
                "artifacts": exec_res.get("artifacts", []),
                "formatted": formatted_res,
            })

    # Persist user message and assistant reply together
    db.add_message(sid, "user", req.message)
    db.add_message(sid, "assistant", reply_text, agent_used=agent_used, tool_calls=tool_results)

    return ChatResponse(
        session_id=sid,
        reply=reply_text,
        agent_used=agent_used,
        tool_results=tool_results,
    )


@app.post("/api/chat/baseline", response_model=ChatResponse)
async def chat_baseline(req: ChatRequest):
    """Dedicated endpoint for zero-shot baseline evaluations without RAG."""
    req.mode = "baseline"
    return await chat(req)


@app.post("/api/tools/web_search")
async def api_web_search(req: WebSearchRequest):
    """Execute a live web search using DuckDuckGo with fallback cascade."""
    results = await aweb_search(req.query, max_results=req.max_results)
    markdown = format_web_search_markdown(req.query, results)
    return {
        "query": req.query,
        "results": results,
        "markdown": markdown,
    }


@app.post("/api/tools/fetch_web_page")
async def api_fetch_web_page(req: WebPageRequest):
    """Fetch and extract clean readable text from a URL."""
    page = await afetch_web_page(req.url, max_chars=req.max_chars)
    markdown = format_web_page_markdown(page)
    return {
        "url": req.url,
        "page": page,
        "markdown": markdown,
    }


@app.get("/api/sessions")
async def list_sessions():
    assert db is not None
    return db.list_sessions()


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    assert db is not None
    return db.get_messages(session_id)


@app.get("/api/jobs")
async def list_jobs(session_id: str | None = None, status: str | None = None):
    assert db is not None
    return db.get_jobs(session_id=session_id, status=status)


@app.post("/api/init", response_model=InitResponse)
async def reinitialise_knowledge():
    """Force re-ingestion of READMEs and KG population."""
    assert vectorstore is not None and knowledge_graph is not None

    ingest = vectorstore.ingest_project_readmes()
    kg_counts = knowledge_graph.populate_domain_knowledge()

    return InitResponse(
        vectordb_chunks=vectorstore.count,
        kg_entities=kg_counts,
        status="ok",
    )


# ------------------------------------------------------------------ #
#  WebSocket (streaming — Phase 4 will wire real streaming)
# ------------------------------------------------------------------ #

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """WebSocket endpoint for streaming chat. Placeholder for Phase 4."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                req = ChatRequest(**msg)
                response = await chat(req)
                await ws.send_text(response.model_dump_json())
            except Exception as exc:
                await ws.send_text(json.dumps({"error": str(exc)}))
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------ #
#  Static files (Frontend)
# ------------------------------------------------------------------ #

frontend_dir = settings.WORKSPACE_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    async def root_fallback():
        return await api_status()


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        reload_dirs=["backend"],
        reload_excludes=["*.db*", "*.wal*", "*.shm*", "backend/data/*"],
    )
