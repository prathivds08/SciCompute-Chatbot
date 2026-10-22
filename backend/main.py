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
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

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

from backend.tools import (
    execute_code,
    extract_code_blocks,
    format_execution_result_markdown,
    sanitize_markdown_text,
)

from backend.sandbox.docker_runner import DockerRunner
from backend.tools.deepmd_analysis_tools import (
    save_analysis_script,
    build_water_density_profile_script,
    build_rdf_script,
    build_extxyz_to_deepmd_raw_script,
    build_loss_curve_script,
    build_parity_plot_script,
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
    """Check if the user prompt requests running or executing code."""
    txt = text.lower()
    triggers = [
        "run the code", "run code", "run it", "run this", "execute it", "execute the code",
        "execute this", "run by yourself", "run and give", "calculate and run", "run the script",
        "execute the script", "run and show", "test and run", "give me the result by running",
        "run it by yourself", "run by itself", "execute by itself", "run and tell me",
        "execute and give", "run the python", "execute the python", "run this script",
    ]
    words = [w.strip(".,!?:;\"'") for w in txt.split()]
    if set(words).intersection({"run", "execute"}) and len(words) <= 5:
        return True
    return any(t in txt for t in triggers)


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
    sid = req.session_id or "default"
    code = req.code

    if not code and req.script_path:
        p = Path(req.script_path)
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
    target_dirs = [base_dir / session_id] if session_id and (base_dir / session_id).exists() else [d for d in base_dir.iterdir() if d.is_dir()]
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

    # Check if this is a follow-up command to run code from the previous assistant message
    stripped_cmd = req.message.strip().lower().rstrip(".,!?")
    if stripped_cmd in {
        "run", "run it", "run this", "run that", "execute", "execute it", "execute this",
        "execute that", "run the code", "execute the code", "run script", "execute script"
    }:
        past_msgs = db.get_messages(sid)
        # Scan backward for the most recent assistant message with an executable code block
        for pm in reversed(past_msgs):
            if pm.get("role") == "assistant":
                blocks = extract_code_blocks(pm.get("content", ""))
                executable = [b for b in blocks if b["language"] in ("python", "py", "bash", "sh")]
                if executable:
                    target_block = executable[0]
                    # Persist user trigger message
                    db.add_message(sid, "user", req.message)
                    exec_res = execute_code(
                        code=target_block["code"],
                        language=target_block["language"],
                        session_id=sid,
                        save_script_name=f"deepmd_analysis_{int(time.time())}.py",
                    )
                    formatted_res = format_execution_result_markdown(exec_res)
                    reply_text = f"Running previous `{target_block['language']}` code in the sandbox:\n\n" + formatted_res
                    db.add_message(sid, "assistant", reply_text, agent_used="code_executor", tool_calls=[exec_res])
                    return ChatResponse(
                        session_id=sid,
                        reply=reply_text,
                        agent_used="code_executor",
                        tool_results=[exec_res],
                    )

    # Persist user message
    db.add_message(sid, "user", req.message)

    wants_execution = _is_execution_request(req.message)

    # Retrieve prior conversation history (excluding the current user message)
    past_msgs = db.get_messages(sid)
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in past_msgs[:-1]
        if m.get("role") in ("user", "assistant")
    ]

    # 1. Zero-shot Baseline mode (No RAG access)
    if req.mode == "baseline":
        assert baseline_agent is not None
        reply_text = await baseline_agent.agenerate_response(
            req.message,
            conversation_history=conversation_history,
        )
        reply_text = sanitize_markdown_text(reply_text)
        agent_used = "baseline_zero_shot"
        tool_results = None

        if wants_execution:
            blocks = extract_code_blocks(reply_text)
            executable_blocks = [b for b in blocks if b["language"] in ("python", "py", "bash", "sh")]
            if executable_blocks:
                target_block = executable_blocks[0]
                exec_res = execute_code(
                    code=target_block["code"],
                    language=target_block["language"],
                    session_id=sid,
                )
                formatted_res = format_execution_result_markdown(exec_res)
                reply_text += "\n\n---\n" + formatted_res
                agent_used = "code_executor"
                tool_results = [exec_res]

        db.add_message(sid, "assistant", reply_text, agent_used=agent_used, tool_calls=tool_results)
        return ChatResponse(
            session_id=sid,
            reply=reply_text,
            agent_used=agent_used,
            tool_results=tool_results,
        )

    # 2. RAG mode (VectorStore + Knowledge Graph context)
    assert retriever is not None
    context = retriever.retrieve(req.message)
    context_str = retriever.format_context_for_prompt(context)

    # Generate response via Ollama
    import httpx

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
            "\nIMPORTANT: The user has requested to run/execute code or perform analysis. Provide a complete, self-contained, "
            "executable Python script in a ```python ... ``` block.\n"
            "- For interfacial and DeePMD analysis on Anatase TiO2, call the structure file at "
            "'/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz' (or DATA_DIR / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz').\n"
            "- Always use print() statements to report key numerical metrics and findings to stdout.\n"
            "- If graphical representations or plots are requested, use matplotlib.pyplot to create clear, titled, labeled figures.\n"
            "The backend will automatically save your script as a Python file and run it inside the isolated Docker container.\n"
        )

    ase_guidelines = (
        "\nASE CODING RULES -- MANDATORY SPOONFED TEMPLATE (CRITICAL API RETURN TYPES):\n"
        "---------------------------------------------------------------------------------\n"
        "CANONICAL SINGLE STRUCTURE LOADING PATTERN (ALWAYS USE THIS FOR EXTXYZ INSPECTION):\n"
        "```python\n"
        "from ase.io import read\n"
        "# ALWAYS use index=0 to load a single Atoms object:\n"
        "atoms = read(str(file_path), index=0)\n"
        "# Now all Atoms methods work directly:\n"
        "positions = atoms.get_positions()            # Shape (N, 3) in Angstrom\n"
        "symbols = atoms.get_chemical_symbols()        # List of element symbols ['Ti', 'O', 'H']\n"
        "cell = atoms.cell                            # 3x3 Lattice Cell\n"
        "scaled = atoms.get_scaled_positions(wrap=True) # Fractional coordinates in [0, 1)\n"
        "atom_count = len(atoms)                      # ALWAYS use len(atoms)\n"
        "```\n"
        "CRITICAL ERROR PREVENTION (STRICTLY FORBIDDEN ANTI-PATTERNS):\n"
        "- NEVER pass index=':' when inspecting or analyzing a structure!\n"
        "  'read(path, index=\":\")' returns a Python list[Atoms], NOT an Atoms object.\n"
        "  Calling atoms.get_positions() or atoms.cell on a list CRASHES with:\n"
        "  AttributeError: 'list' object has no attribute 'get_positions'\n"
        "- If inspecting a structure, boundary atoms, surface atoms, or properties: ALWAYS USE index=0.\n"
        "- NEVER use atoms.get_count(), atoms.num_atoms(), or atoms.count() (DO NOT EXIST; use len(atoms)).\n"
        "- Potential energy: atoms.get_potential_energy() [eV]. Convert to Hartree: energy_eV / Hartree.\n"
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
        "\nCODE GENERATION MANDATE:\n"
        "- If providing Python code, always provide a complete, self-contained, immediately executable script.\n"
        "- Structure reading MUST follow: 'atoms = ase.io.read(path, index=0)'. NEVER use index=':' for single structures.\n"
        "- Dual-path fallback for dataset loading:\n"
        "    CONTAINER_PATH = Path('/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz')\n"
        "    LOCAL_PATH = Path(__file__).resolve().parent / 'data' / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'\n"
        "    DATA_PATH = CONTAINER_PATH if CONTAINER_PATH.exists() else LOCAL_PATH\n"
    )

    system_prompt = (
        "You are a scientific computing assistant specialising in DFT (Quantum ESPRESSO), "
        "DeePMD-kit model training, LAMMPS molecular dynamics, and post-processing analysis "
        "of metal-oxide/water interfaces.\n\n"
        f"{data_isolation_policy}"
        f"{ase_guidelines}"
        f"{numpy_guidelines}"
        f"{surface_boundary_rules}"
        f"{anatase_physics_and_models}"
        f"{code_generation_mandate}"
        f"{execution_instruction}"
        "Use the following retrieved context to help answer the user's question. "
        "If the context doesn't contain the answer, say so and use your own knowledge.\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context_str}\n--- END CONTEXT ---"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": req.message},
    ]

    async with httpx.AsyncClient(timeout=None) as client:
        ollama_resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": 32768,
                    "temperature": 0.6,
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

    agent_used = "qa_rag"
    tool_results = None

    if wants_execution:
        blocks = extract_code_blocks(reply_text)
        executable_blocks = [b for b in blocks if b["language"] in ("python", "py", "bash", "sh")]
        if executable_blocks:
            target_block = executable_blocks[0]
            exec_res = execute_code(
                code=target_block["code"],
                language=target_block["language"],
                session_id=sid,
                save_script_name=f"deepmd_analysis_{int(time.time())}.py",
            )
            formatted_res = format_execution_result_markdown(exec_res)
            reply_text += "\n\n---\n" + formatted_res
            agent_used = "code_executor"
            tool_results = [exec_res]

    # Persist assistant reply
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
            msg = json.loads(data)

            # For now, just echo back through the REST chat logic
            req = ChatRequest(**msg)
            response = await chat(req)
            await ws.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------ #
#  Static files (Frontend)
# ------------------------------------------------------------------ #

frontend_dir = settings.WORKSPACE_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


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
    )
