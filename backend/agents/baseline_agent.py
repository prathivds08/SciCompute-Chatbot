"""
Baseline Agent — Zero-shot scientific computing assistant without RAG access.

This agent serves as the experimental baseline for comparative evaluations
against the full hybrid-RAG pipeline (ChromaDB + Kùzu). It purposefully
has NO access to:
  - ChromaDB (VectorStore) retrieved chunks
  - Kùzu (KnowledgeGraph) domain triples
  - Any external document retrieval system

It relies exclusively on the LLM's pre-trained parametric knowledge using
an enhanced, maximally-capable zero-shot system prompt tuned for reasoning
models such as DeepSeek-R1 (and Qwen / Qwen-Coder variants).

Usage
-----
Instantiate directly (settings are pulled from backend.config)::

    agent = BaselineAgent()
    reply = await agent.agenerate_response(user_message, history)

Or override model / URL for testing::

    agent = BaselineAgent(
        model_name="deepseek-r1:7b",
        base_url="http://localhost:11434",
    )
"""

import re
from typing import Any

import httpx

from backend.config import settings
from backend.tools.code_tools import sanitize_markdown_text
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


# ---------------------------------------------------------------------------
# Enhanced Zero-Shot System Prompt (Optimized for qwen2.5-coder:7b)
# ---------------------------------------------------------------------------
# Design principles:
#
#  1. IDENTITY & EXPERTISE      -- XML-delimited role anchoring in computational
#                                 materials science and scientific Python.
#  2. VISIBLE CoT REASONING     -- structured "Plan -> Code -> Verify" chain of
#                                 thought instead of leaked <think> tags.
#  3. DOMAIN TAXONOMY           -- DFT (Quantum ESPRESSO), DeePMD-kit (DP-GEN),
#                                 LAMMPS MD, and ASE trajectory analysis.
#  4. POSITIVE CODE RECIPES     -- copy-ready, verified Python/ASE/NumPy idioms
#                                 without negative anti-pattern contamination.
#  5. COMPACT TOKEN FOOTPRINT   -- ~1,450 tokens with zero redundancy for high
#                                 GQA attention fidelity on 7B coder architectures.
#  6. DATA ISOLATION POLICY     -- Docker sandbox path only (/data/Anatase_...),
#                                 never dump raw coordinates in responses.
#  7. LIVE WEB TOOLS ACCESS     -- Web search and webpage fetch tools for up-to-date
#                                 documentation and external scientific references.
# ---------------------------------------------------------------------------
BASELINE_ZERO_SHOT_SYSTEM_PROMPT = """
<role>
You are Dr. Compute, a principal computational materials scientist and scientific Python engineer specializing in:
1. Density Functional Theory (DFT) with Quantum ESPRESSO (pw.x).
2. Machine-Learned Interatomic Potentials with DeePMD-kit and DP-GEN.
3. Molecular Dynamics (MD) with LAMMPS.
4. Interfacial trajectory post-processing with Python, ASE, NumPy, SciPy, Pandas, and Matplotlib.
</role>

<web_tools>
LIVE INTERNET ACCESS & WEB TOOLS:
You have real-time access to the live web and online documentation through two tools:
1. Web Search: emit `<web_search>query</web_search>` to search DuckDuckGo / web for latest releases, documentation, packages, or scientific articles.
2. Webpage Fetch: emit `<fetch_url>https://example.com</fetch_url>` to read the full text content of a specific webpage or URL.

When you need current information (e.g. latest version of software, new package APIs, online GitHub repos, external documentation), emit the tool call tag. The system will execute the query and provide the results in `<live_web_context>`, enabling you to provide complete, factually grounded answers.
</web_tools>

<docker_tools>
LIVE DOCKER CONTAINER EXECUTION TOOL:
You have access to an isolated, secure Docker execution container (`deepmd-analysis-sandbox`).
The structure dataset is mounted read-only at `/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`.

To execute calculations, compute interfacial properties (forces, density profiles, RDF, etc.), or generate plots:
Emit your Python script inside `<docker_run>...</docker_run>` tags:
<docker_run>
import os, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
...
</docker_run>

The system executes this script inside the secure Docker container and returns the stdout and graphical plot artifacts.
</docker_tools>

<data_privacy_policy>
MANDATORY DATA ISOLATION RULES:
1. You do NOT have direct access to raw atomic coordinates, forces, or xyz blocks of "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz".
2. Never fabricate, dump, or print per-atom coordinate tables in prose or stdout.
3. Inside the Docker execution sandbox, the dataset is mounted read-only at:
   /data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz
4. Always write self-contained Python scripts that load the file inside Docker, compute aggregate physical metrics (counts, energies, density profiles, RDF peaks, residence times), print concise summary tables to stdout, and save figures to "figure_output.png".
</data_privacy_policy>

<import_rules_and_prelude>
CRITICAL IMPORT & DEPENDENCY RULES (ZERO IMPORT/NAME ERRORS):
1. NEVER write partial or notebook-continuation snippets that assume `np`, `plt`, `Path`, `os`, `read`, or `neighbor_list` are already imported.
2. You are ENCOURAGED to include the full Standard Scientific Prelude at the top of every Python script (extra standard imports are completely allowed and preferred over missing imports):

   import os
   import sys
   import json
   import math
   from pathlib import Path

   import numpy as np
   import pandas as pd
   import scipy
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   import ase
   from ase import Atoms
   from ase.io import read, write
   from ase.geometry import get_distances
   from ase.neighborlist import neighbor_list, NeighborList, natural_cutoffs
   from ase.units import Hartree, Bohr, mol, kJ, fs, kB

3. EXACT SUBMODULE LOOKUP TABLE (use these exact import paths):
   - Structure I/O       : from ase.io import read, write
   - QE I/O              : from ase.io.espresso import write_espresso_in, read_espresso_out
   - LAMMPS I/O          : from ase.io.lammpsdata import read_lammps_data, write_lammps_data
   - Neighbor searches   : from ase.neighborlist import neighbor_list, NeighborList
   - PBC distance matrix : from ase.geometry import get_distances, find_mic
   - Physical units      : from ase.units import Hartree, Bohr, mol, kJ, fs, kB
   - Avogadro constant   : NA = 6.02214076e23 (define as float constant directly)
   - DeePMD evaluation   : from deepmd.infer import DeepPot
</import_rules_and_prelude>

<ase_api_rules>
CRITICAL ASE FUNCTION SIGNATURES & USAGE (PREVENT FATAL RUNTIME CRASHES):
1. from ase.io import read, write
   - Reading: atoms = read(str(data_path), index=0) for single structure. NEVER use index=":" unless looping over trajectory frames!
   - Writing: To save structures, always construct a new `Atoms` instance from the class `ase.Atoms`:
     from ase import Atoms
     boundary_atoms = Atoms(symbols=symbols, positions=positions, cell=atoms.cell, pbc=atoms.pbc)
     write("boundary_atoms.extxyz", boundary_atoms)
   - FATAL ERROR: `atoms` returned by `read()` is an INSTANCE, NOT a class. Calling `atoms(...)` crashes with: TypeError: 'Atoms' object is not callable.

2. from ase.geometry import get_distances
   - Exact Signature: disp_vecs, dist_matrix = get_distances(p1, p2=None, cell=None, pbc=None)
   - FATAL ERROR: NEVER pass `mic=True` to `get_distances()`! It does NOT accept a `mic` keyword argument and crashes with:
     TypeError: get_distances() got an unexpected keyword argument 'mic'.
     Minimum Image Convention is automatically applied when `cell` and `pbc` are passed.
   - Return Value: Returns a 2-tuple (displacement_vectors, distances). The scalar distance matrix of shape (len(p1), len(p2)) is:
     dist_matrix = get_distances(p1, p2, cell=atoms.cell, pbc=atoms.pbc)[1]
   - Alternative: On an Atoms instance, `atoms.get_all_distances(mic=True)` directly returns an (N, N) distance array.

3. from ase.neighborlist import neighbor_list
   - Exact Signature: i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff=...)
   - Quantities: 'i' (source indices), 'j' (neighbor indices), 'd' (distances), 'D' (displacement vectors).
   - Cutoff Dict: When specifying pair cutoffs, always provide BOTH symmetric permutations, e.g.:
     {("O", "H"): 1.20, ("H", "O"): 1.20, ("Ti", "O"): 2.50, ("O", "Ti"): 2.50}
   - Water O vs. Lattice O: Water oxygens have covalent O-H bonds (< 1.20 A):
     i_oh, _, _ = neighbor_list("ijd", atoms, {("O", "H"): 1.20, ("H", "O"): 1.20})
     water_o_idx = np.unique(i_oh[symbols[i_oh] == "O"])
     lattice_o_idx = np.setdiff1d(np.where(symbols == "O")[0], water_o_idx)

4. from ase.units import Hartree
   - Energy conversion: energy_Ha = atoms.get_potential_energy() / Hartree
   - Force conversion: forces_Ha_Bohr = atoms.get_forces() / (Hartree / Bohr)
   - FATAL ERROR: Never write `from ase.units import Avogadro` (does NOT exist; use `NA = 6.02214076e23`).
</ase_api_rules>

<domain_knowledge>
1. ANATASE TiO2 (101) / WATER INTERFACE SPECIFICATIONS:
   - Dataset Composition (426 atoms total, orthorhombic PBC box ~ 10.25 x 11.31 x 36.80 Angstrom):
     * 60 Ti atoms (180-atom stoichiometric TiO2 slab: 60 Ti + 120 lattice O).
     * 202 O atoms total = 120 lattice O (O_3c bulk + O_2c bridging) + 82 water O (O_w).
     * 164 H atoms = 82 intact H2O molecules.
   - Surface Sites & Physical Boundary vs. PBC Box Faces:
     * X and Y axes are periodic and continuous (no physical walls in X or Y).
     * Physical solid-liquid interfaces lie strictly along the surface normal (Z-axis).
     * Active surface sites: ONLY 6 top Ti_5c + 6 bottom Ti_5c = 12 surface Ti_5c atoms total, plus 12 surface bridging oxygens (O_2c / O_br) = 24 solid surface boundary atoms (far fewer than the 60 total Ti atoms in the full slab).
     * Geometric PBC box faces: With a physical skin cutoff of 0.2 Angstrom (or 0.5 Angstrom) from wrapped cell faces, only ~4-6 atoms (or ~12-18 atoms) lie on box faces.
   - Interfacial Physics & Signatures:
     * Pristine Anatase (101) adsorbs water predominantly MOLECULARLY (intact H2O) at 5-fold Ti_5c Lewis acid sites: d(Ti_5c - O_w) = 2.15-2.25 Angstrom, E_ads = -0.7 to -0.9 eV.
     * Water protons H_w donate H-bonds to 2-fold bridging oxygens O_br: d(O_br - H_w) = 1.75-1.95 Angstrom.
     * Dissociation (Ti_5c-OH + O_br-H) is endothermic on pristine (101) and occurs primarily at oxygen vacancies (V_O) or steps.
     * Z-Density Profile rho(z): 1st hydration peak at z ~ 2.1-2.3 Angstrom above Ti_5c plane; 2nd hydration peak at z ~ 3.8-4.2 Angstrom; bulk liquid water plateau at ~1.0 g/cm^3 (0.033 molecules/Angstrom^3).
     * Pair RDF g(r) Peaks: Ti_5c-O_w at 2.18-2.25 Angstrom; O_br-H_w at 1.75-1.95 Angstrom; O_w-O_w at 2.75-2.82 Angstrom; intramolecular O_w-H_w at 0.98-1.02 Angstrom.

2. QUANTUM ESPRESSO (DFT) STANDARDS:
   - Platinum (Pt.cif, 4-atom FCC): pseudo = "Pt_ONCV_PBE-1.2.upf", ecutwfc = 60-120 Ry, ecutrho = 4 * ecutwfc, smearing = "marzari-vanderbilt", degauss = 0.02 Ry, nbnd = 44, k-points = 4x4x4 to 12x12x12.
   - Anatase TiO2 / Water Slab: PBE + Hubbard U (U_eff = 3.0-3.5 eV on Ti-3d), Grimme DFT-D3/D4 or revPBE-D3 dispersion, ecutwfc >= 60-85 Ry, ecutrho >= 480-680 Ry, k-mesh >= 4x4x1, dipole correction along z (dipfield = .true.).

3. DEEPMD-KIT & DP-GEN TRAINING STANDARDS:
   - Raw files: box.raw (N_frames, 9), coord.raw (N_frames, 3*N), force.raw (N_frames, 3*N), energy.raw (N_frames, 1), type.raw (N,), type_map.raw -> converted to set.000/*.npy (float32 for box/coord/force, float64 for energy).
   - Type maps: Anatase/Water = ["Ti", "O", "H"]; Pure Water = ["H", "O"] (sel: [120, 64]); Graphene/Water Interface = ["O", "H", "C"].
   - Descriptor: "se_e2_a" (rcut = 6.0 A, rcut_smth = 0.5-3.0 A, neuron = [120, 120, 120], axis_neuron = 12, fitting_net = [120, 120, 120] with resnet_dt = true) or "DPA-2".
   - Learning rate & Loss: exp decay (start_lr = 0.005, decay_steps = 5000, decay_rate = 0.95); loss start_pref_e = 0.02, limit_pref_e = 8, start_pref_f = 1000, limit_pref_f = 1.
   - DP-GEN model deviation (sigma_f across 4 seeded models): sigma_f < 0.05 eV/A (accurate), 0.05 <= sigma_f <= 0.20 eV/A (label with DFT), sigma_f > 0.20 eV/A (failed/unphysical).

4. LAMMPS MOLECULAR DYNAMICS STANDARDS:
   - Core setup: units metal, boundary p p p, atom_style atomic, pair_style deepmd graph1.pb graph2.pb graph3.pb out_file md.out out_freq 100, pair_coeff * *.
   - Timestep & Thermostat: dt = 0.0005 ps (0.5 fs) for H2O (1.0 fs for D2O); NVT Nose-Hoover (fix 1 all nvt temp 300 300 0.05) at 300 K (Water/Anatase) or 330 K (Graphene-Water Interface).
   - Batch chaining: Batch 1 uses read_data + velocity create; Batch N >= 2 uses read_restart ../batch_{N-1}/lammps.restart; end each batch with timer timeout and write_restart lammps.restart; chain via sbatch --dependency=afterok:$PREV_JOBID.
</domain_knowledge>

<canonical_code_patterns>
Every code example below includes its full top-level imports. Always follow these exact patterns:

1. COMPLETE TEMPLATE: STRUCTURE INSPECTION, SURFACE SITES, PBC DISTANCES & DENSITY PROFILE:
   import os
   import sys
   from pathlib import Path
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from ase.io import read
   from ase.geometry import get_distances
   from ase.neighborlist import neighbor_list
   from ase.units import Hartree

   def main() -> None:
       container_path = Path("/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
       local_path = Path(__file__).resolve().parent / "data" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
       data_path = container_path if container_path.exists() else local_path
       if not data_path.exists():
           raise FileNotFoundError(f"Structure file not found: {data_path}")

       # Use index=0 for single structure (returns ase.Atoms); use index=":" ONLY when looping 'for frame in frames:'
       atoms = read(str(data_path), index=0)
       n_atoms = len(atoms)
       positions = atoms.get_positions()
       scaled = atoms.get_scaled_positions(wrap=True)
       symbols = np.array(atoms.get_chemical_symbols())
       cell_lengths = atoms.cell.lengths()
       c_len = float(cell_lengths[2])

       # Vectorized PBC distance check
       dists = atoms.get_all_distances(mic=True)
       np.fill_diagonal(dists, np.inf)
       has_overlap = bool(np.any(dists < 1e-5))

       # Center slab z-coordinates if Ti wraps across z=0 / z=c
       ti_mask = symbols == "Ti"
       o_mask = symbols == "O"
       ti_z_raw = positions[ti_mask, 2]
       if bool(np.any(ti_z_raw < 5.0)) and bool(np.any(ti_z_raw > (c_len - 5.0))):
           z_centered = (positions[:, 2] + c_len / 2.0) % c_len
       else:
           z_centered = positions[:, 2]

       # Identify physical surface Ti_5c sites (top and bottom 1.5 A of slab)
       ti_z_min, ti_z_max = float(z_centered[ti_mask].min()), float(z_centered[ti_mask].max())
       surface_ti_idx = np.where(ti_mask & ((z_centered >= ti_z_max - 1.5) | (z_centered <= ti_z_min + 1.5)))[0]

       # Separate water O (82 O_w bonded to H within 1.20 A) from lattice O (120 O_lat)
       i_oh, _, _ = neighbor_list("ijd", atoms, {("O", "H"): 1.20, ("H", "O"): 1.20})
       water_o_idx = np.unique(i_oh[symbols[i_oh] == "O"])
       lattice_o_idx = np.setdiff1d(np.where(o_mask)[0], water_o_idx)

       # Geometric PBC box face proximity (unique indices via boolean mask)
       dist_to_face = np.min(np.minimum(scaled, 1.0 - scaled) * cell_lengths, axis=1)
       box_face_idx = np.where(dist_to_face <= 0.2)[0]

       # Water mass density profile rho(z) [g/cm^3]
       dz = 0.15
       bins = np.arange(0.0, c_len + dz, dz)
       bin_centers = 0.5 * (bins[:-1] + bins[1:])
       hist_ow, _ = np.histogram(z_centered[water_o_idx], bins=bins)
       bin_vol_cm3 = (cell_lengths[0] * cell_lengths[1] * dz) * 1e-24
       NA = 6.02214076e23
       rho_water = (hist_ow * 18.01528 / NA) / bin_vol_cm3

       print(f"Total atoms: {n_atoms} | Overlap(<1e-5 A): {has_overlap}")
       print(f"Surface Ti_5c count: {len(surface_ti_idx)} | Box-face(<=0.2A) count: {len(box_face_idx)}")
       print(f"Lattice O: {len(lattice_o_idx)} | Water O: {len(water_o_idx)} | Max rho(z): {rho_water.max():.2f} g/cm^3")

       plt.figure(figsize=(8, 4.5), dpi=150)
       plt.plot(bin_centers, rho_water, lw=2.0, label="Water Mass Density (g/cm^3)")
       plt.xlabel("z (Angstrom)")
       plt.ylabel("Density (g/cm^3)")
       plt.legend()
       plt.tight_layout()
       plt.savefig("figure_output.png", dpi=150, bbox_inches="tight")
       plt.close("all")

   if __name__ == "__main__":
       main()
</canonical_code_patterns>

<output_contract>
1. FOR CONCEPTUAL / THEORETICAL QUESTIONS (No Code Requested):
   - Provide a clear, rigorous scientific explanation grounded in physical equations, coordination chemistry, and simulation best practices.

2. FOR CODE GENERATION OR DATA ANALYSIS REQUESTS (MANDATORY 3-STEP STRUCTURE):
   - Step 1 (Pre-Code Plan & Import Manifest): Before opening the code fence, write 2-4 concise bullet points stating:
     * Physical methodology & equations used.
     * Required Imports Manifest: Explicitly list every module and function that the script will import (`os`, `sys`, `Path`, `np`, `plt`, `read`, `neighbor_list`, `get_distances`, `Hartree`, etc.).
   - Step 2 (Complete Executable Script): Emit a single, 100% self-contained Python script inside a ```python ... ``` fence that:
     * Opens with the Mandatory Standard Scientific Prelude from <import_rules_and_prelude> plus any task-specific imports listed in Step 1.
     * Wraps execution in `def main() -> None:` with `if __name__ == "__main__": main()`.
     * Uses `CONTAINER_PATH` / `LOCAL_PATH` fallback, `len(atoms)`, `atoms.get_potential_energy()`, and `np.any()` / `np.all()`.
     * Saves plots to `figure_output.png` via `plt.savefig(...)` and `plt.close("all")`, and prints only aggregate metrics to stdout.
   - Step 3 (Verification Summary): Briefly state what the script outputs and how physical constraints (PBC, units, stoichiometry) were handled.
</output_contract>
""".strip()


class BaselineAgent:
    """
    Zero-shot LLM agent with no access to external RAG knowledge bases.

    This agent queries the configured Ollama-compatible model using only
    the enhanced zero-shot system prompt defined above.  It is the control
    condition in experiments comparing RAG-augmented vs. baseline response
    quality for scientific computing queries.

    Parameters
    ----------
    model_name : str | None
        Ollama model tag to use (e.g. ``"deepseek-r1:7b"``).  Falls back to
        ``settings.MODEL_NAME`` when None.
    base_url : str | None
        Ollama server URL.  Falls back to ``settings.OLLAMA_BASE_URL``.
    system_prompt : str | None
        Override the built-in zero-shot prompt.  Use only for testing.
    timeout : float | None
        HTTP timeout in seconds (``None`` = no timeout, recommended for
        large models on first load).
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Resolve model name — prefer explicit argument, fall back to config
        self.model_name: str = model_name or settings.MODEL_NAME

        # Strip trailing slash from base URL to avoid double-slash in API paths
        self.base_url: str = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")

        # Allow callers to inject a custom prompt (e.g., for A/B testing)
        self.system_prompt: str = system_prompt or BASELINE_ZERO_SHOT_SYSTEM_PROMPT

        # None disables the timeout — safe for slow first-load of 7B+ models
        self.timeout: float | None = timeout

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def build_messages(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Construct the full messages array for the Ollama /api/chat endpoint.

        The array layout is:
          ``[system_prompt, *conversation_history, current_user_message]``

        Parameters
        ----------
        message : str
            The current user message to append at the end of the array.
        conversation_history : list[dict[str, str]] | None
            Prior turns in the format
            ``[{"role": "user"|"assistant", "content": "..."}]``.
            Silently ignored when ``None`` or empty.

        Returns
        -------
        list[dict[str, str]]
            A list of message dicts ready for the Ollama chat API.
        """
        # Always open with the enhanced zero-shot system prompt
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

         # Replay conversation history if provided, filtering to valid roles only
        if conversation_history:
            for item in conversation_history:
                role = item.get("role", "user")
                content = item.get("content", "")

                # Guard against unexpected roles (e.g., "tool", "system")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})

        # Recency anchor for code-generation queries on qwen2.5-coder:7b
        q_low = message.lower()
        code_triggers = (
            "code", "script", "python", "run", "execute", "plot",
            "calculate", "compute", "extxyz", "rdf", "density", "boundary", "analyze",
        )
        if any(t in q_low for t in code_triggers):
            augmented_message = (
                f"{message}\n\n"
                "[Mandatory Code Checklist:\n"
                "1. First list 'Required Imports:' before the code block.\n"
                "2. Start the ```python block with explicit imports: `import os, sys, json, math`, `from pathlib import Path`, "
                "`import numpy as np`, `import matplotlib.pyplot as plt`, `from ase.io import read, write`, "
                "`from ase.geometry import get_distances`, `from ase.neighborlist import neighbor_list`, `from ase.units import Hartree`.\n"
                "3. Use `read(str(DATA_PATH), index=0)` for single structures, `len(atoms)`, `np.any()`, and save figures to `figure_output.png`.]"
            )
        else:
            augmented_message = message

        # Append the current user turn last so the model responds to it
        messages.append({"role": "user", "content": augmented_message})

        return messages


    # ------------------------------------------------------------------
    # Prompt sanitisation
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_prompt(text: str) -> str:
        """
        Strip characters that can trigger a 400 Bad Request from Ollama
        when using models with strict chat templates (e.g. deepseek-r1).

        DeepSeek-R1's Ollama template parser rejects certain Unicode ranges:
          - Box-drawing characters (U+2500..U+257F)
          - Block elements        (U+2580..U+259F)
          - Enclosed alphanumerics / misc symbols

        These are only used for visual decoration in the prompt; replacing
        them with plain-ASCII equivalents preserves all semantic content.

        Parameters
        ----------
        text : str
            Raw system-prompt or user-message string.

        Returns
        -------
        str
            A version of ``text`` safe for all Ollama model templates.
        """
        # Map specific decoration characters to ASCII equivalents
        replacements = {
            # Box-drawing horizontals / verticals -> plain dashes / pipes
            "\u2500": "-", "\u2501": "-",  # light/heavy horizontal
            "\u2502": "|", "\u2503": "|",  # light/heavy vertical
            "\u250c": "+", "\u2510": "+", "\u2514": "+", "\u2518": "+",  # corners
            "\u251c": "+", "\u2524": "+", "\u252c": "+", "\u2534": "+",  # tees
            "\u253c": "+",                                                  # cross
            # Double/heavy box-drawing (U+2550-U+256C range)
            "\u2550": "=",  # double horizontal (used for section dividers)
            "\u2551": "|",  # double vertical
            "\u2552": "+", "\u2553": "+", "\u2554": "+",  # double corners
            "\u2555": "+", "\u2556": "+", "\u2557": "+",
            "\u2558": "+", "\u2559": "+", "\u255a": "+",
            "\u255b": "+", "\u255c": "+", "\u255d": "+",
            "\u255e": "+", "\u255f": "+", "\u2560": "+",
            "\u2561": "+", "\u2562": "+", "\u2563": "+",
            "\u2564": "+", "\u2565": "+", "\u2566": "+",
            "\u2567": "+", "\u2568": "+", "\u2569": "+",
            "\u256a": "+", "\u256b": "+", "\u256c": "+",
            # Misc symbols used in the prompt
            "\u2717": "[X]",  # ballot X  (SYNTAX checklist)
            "\u2022": "*",    # bullet
            "\u2014": "--",   # em-dash
            "\u2013": "-",    # en-dash
            "\u2192": "->",   # rightwards arrow
            "\u2190": "<-",   # leftwards arrow
            "\u00b2": "2",    # superscript 2 (TiO2)
            "\u2082": "2",    # subscript 2
            "\u00b3": "3",    # superscript 3
            "\u207b": "-",    # superscript minus
            "\u00c5": "Angstrom",  # Angstrom sign
            "\u03c1": "rho",       # Greek rho (density)
            "\u03b1": "alpha",     # Greek alpha
            "\u03b2": "beta",      # Greek beta
            "\u03b3": "gamma",     # Greek gamma
            "\u2264": "<=",        # less-than-or-equal
            "\u2265": ">=",        # greater-than-or-equal
            "\u00d7": "x",         # multiplication sign
            "\u00b0": "deg",       # degree symbol
            "\u2019": "'",         # right single quotation mark
            "\u201c": '"',         # left double quotation mark
            "\u201d": '"',         # right double quotation mark
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    # ------------------------------------------------------------------
    # Synchronous generation with Web Tools
    # ------------------------------------------------------------------

    def generate_response_with_tools(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        max_tool_iterations: int = 2,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """
        Synchronously query the model with the zero-shot prompt and live web tools.

        If the user query has explicit web intent (e.g. searching the web, URLs),
        or if the model emits `<web_search>...</web_search>` or `<fetch_url>...</fetch_url>`,
        the web tools are executed and fed back into context.

        Parameters
        ----------
        message : str
            The user's query.
        conversation_history : list[dict[str, str]] | None
            Prior dialogue turns.
        max_tool_iterations : int, default 2
            Maximum iterative tool-call steps.

        Returns
        -------
        tuple[str, list[dict[str, Any]], str]
            (final_reply_text, tool_results, agent_used)
        """
        tool_results: list[dict[str, Any]] = []
        agent_used = "baseline_zero_shot"

        # 1. Proactive web intent detection on user query
        needs_web, action_type, target = detect_web_intent(message)
        web_context_blocks: list[str] = []

        if needs_web and target:
            agent_used = "baseline_web_agent"
            if action_type == "search":
                search_res = web_search(target, max_results=5)
                formatted = format_web_search_markdown(target, search_res)
                tool_results.append({
                    "tool": "web_search",
                    "query": target,
                    "results": search_res,
                    "formatted": formatted,
                })
                web_context_blocks.append(formatted)
            elif action_type == "fetch":
                page_res = fetch_web_page(target, max_chars=4000)
                formatted = format_web_page_markdown(page_res)
                tool_results.append({
                    "tool": "fetch_url",
                    "url": target,
                    "result": page_res,
                    "formatted": formatted,
                })
                web_context_blocks.append(formatted)

        # Build initial prompt messages
        augmented_message = message
        if web_context_blocks:
            joined_context = "\n\n".join(web_context_blocks)
            augmented_message = (
                f"<live_web_context>\n"
                f"{joined_context}\n"
                f"</live_web_context>\n\n"
                f"{message}"
            )

        messages = self.build_messages(augmented_message, conversation_history)
        for msg in messages:
            msg["content"] = self._sanitize_prompt(msg["content"])

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": 16384,
                    "temperature": 0.15,
                    "top_p": 0.90,
                    "top_k": 20,
                    "repeat_penalty": 1.05,
                },
            },
            timeout=self.timeout,
        )

        if resp.status_code >= 400:
            import sys
            print(
                f"[BaselineAgent] Ollama returned {resp.status_code} for model "
                f"'{self.model_name}'.\n"
                f"  Response body: {resp.text[:500]}",
                file=sys.stderr,
            )
        resp.raise_for_status()

        content = resp.json()["message"]["content"]
        content = sanitize_markdown_text(content)

        # 2. Check for reactive tool calls emitted by the model
        iteration = 0
        while iteration < max_tool_iterations:
            model_tool_calls = extract_web_tool_calls(content)
            if not model_tool_calls:
                break

            agent_used = "baseline_web_agent"
            iteration += 1
            new_outputs: list[str] = []

            for call in model_tool_calls:
                t_type = call.get("tool")
                arg = call.get("argument", "")
                if t_type == "web_search":
                    s_res = web_search(arg, max_results=5)
                    fmt = format_web_search_markdown(arg, s_res)
                    tool_results.append({
                        "tool": "web_search",
                        "query": arg,
                        "results": s_res,
                        "formatted": fmt,
                    })
                    new_outputs.append(fmt)
                elif t_type == "fetch_url":
                    p_res = fetch_web_page(arg, max_chars=4000)
                    fmt = format_web_page_markdown(p_res)
                    tool_results.append({
                        "tool": "fetch_url",
                        "url": arg,
                        "result": p_res,
                        "formatted": fmt,
                    })
                    new_outputs.append(fmt)

            if not new_outputs:
                break

            followup_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Live Web Tool Results:\n\n"
                        + "\n\n".join(new_outputs)
                        + "\n\nPlease provide your final answer to the user synthesizing these results. "
                        "Do not output raw tool tags."
                    ),
                },
            ]
            for msg in followup_messages:
                msg["content"] = self._sanitize_prompt(msg["content"])

            f_resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": followup_messages,
                    "stream": False,
                    "options": {
                        "num_ctx": 16384,
                        "temperature": 0.15,
                        "top_p": 0.90,
                        "top_k": 20,
                        "repeat_penalty": 1.05,
                    },
                },
                timeout=self.timeout,
            )
            if f_resp.status_code >= 400:
                break
            f_resp.raise_for_status()
            content = sanitize_markdown_text(f_resp.json()["message"]["content"])

        cleaned_reply = re.sub(
            r'<(?:web_search|fetch_url)>.*?</(?:web_search|fetch_url)>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if not cleaned_reply and content:
            cleaned_reply = content

        return cleaned_reply, tool_results, agent_used

    def generate_response(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Synchronously query the model with the zero-shot prompt and web tools.

        Parameters
        ----------
        message : str
            The user's query.
        conversation_history : list[dict[str, str]] | None
            Optional prior conversation turns for multi-turn dialogue.

        Returns
        -------
        str
            The assistant's response text from the model.
        """
        reply_text, _, _ = self.generate_response_with_tools(
            message=message,
            conversation_history=conversation_history,
        )
        return reply_text

    # ------------------------------------------------------------------
    # Asynchronous generation with Web Tools (preferred inside FastAPI)
    # ------------------------------------------------------------------

    async def agenerate_response_with_tools(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        max_tool_iterations: int = 2,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """
        Asynchronously query the model with zero-shot prompt and live web tools.

        If the user query has explicit web intent (e.g. searching the web, URLs),
        or if the model emits `<web_search>...</web_search>` or `<fetch_url>...</fetch_url>`,
        the web tools are executed and fed back into context.

        Parameters
        ----------
        message : str
            The user's query.
        conversation_history : list[dict[str, str]] | None
            Prior dialogue turns.
        max_tool_iterations : int, default 2
            Maximum iterative tool-call steps.

        Returns
        -------
        tuple[str, list[dict[str, Any]], str]
            (final_reply_text, tool_results, agent_used)
        """
        tool_results: list[dict[str, Any]] = []
        agent_used = "baseline_zero_shot"

        # 1. Proactive web intent detection on user query
        needs_web, action_type, target = detect_web_intent(message)
        web_context_blocks: list[str] = []

        if needs_web and target:
            agent_used = "baseline_web_agent"
            if action_type == "search":
                search_res = await aweb_search(target, max_results=5)
                formatted = format_web_search_markdown(target, search_res)
                tool_results.append({
                    "tool": "web_search",
                    "query": target,
                    "results": search_res,
                    "formatted": formatted,
                })
                web_context_blocks.append(formatted)
            elif action_type == "fetch":
                page_res = await afetch_web_page(target, max_chars=4000)
                formatted = format_web_page_markdown(page_res)
                tool_results.append({
                    "tool": "fetch_url",
                    "url": target,
                    "result": page_res,
                    "formatted": formatted,
                })
                web_context_blocks.append(formatted)

        # Build initial prompt messages
        augmented_message = message
        if web_context_blocks:
            joined_context = "\n\n".join(web_context_blocks)
            augmented_message = (
                f"<live_web_context>\n"
                f"{joined_context}\n"
                f"</live_web_context>\n\n"
                f"{message}"
            )

        messages = self.build_messages(augmented_message, conversation_history)
        for msg in messages:
            msg["content"] = self._sanitize_prompt(msg["content"])

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": 16384,
                        "temperature": 0.15,
                        "top_p": 0.90,
                        "top_k": 20,
                        "repeat_penalty": 1.05,
                    },
                },
            )
            if resp.status_code >= 400:
                import sys
                print(
                    f"[BaselineAgent] Ollama returned {resp.status_code} for model "
                    f"'{self.model_name}'.\n"
                    f"  Response body: {resp.text[:500]}",
                    file=sys.stderr,
                )
            resp.raise_for_status()

            content = resp.json()["message"]["content"]
            content = sanitize_markdown_text(content)

            # 2. Check for reactive tool calls emitted by the model
            iteration = 0
            while iteration < max_tool_iterations:
                model_tool_calls = extract_web_tool_calls(content)
                if not model_tool_calls:
                    break

                agent_used = "baseline_web_agent"
                iteration += 1
                new_outputs: list[str] = []

                for call in model_tool_calls:
                    t_type = call.get("tool")
                    arg = call.get("argument", "")
                    if t_type == "web_search":
                        s_res = await aweb_search(arg, max_results=5)
                        fmt = format_web_search_markdown(arg, s_res)
                        tool_results.append({
                            "tool": "web_search",
                            "query": arg,
                            "results": s_res,
                            "formatted": fmt,
                        })
                        new_outputs.append(fmt)
                    elif t_type == "fetch_url":
                        p_res = await afetch_web_page(arg, max_chars=4000)
                        fmt = format_web_page_markdown(p_res)
                        tool_results.append({
                            "tool": "fetch_url",
                            "url": arg,
                            "result": p_res,
                            "formatted": fmt,
                        })
                        new_outputs.append(fmt)

                if not new_outputs:
                    break

                # Send tool results back to LLM for final synthesis
                followup_messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Live Web Tool Results:\n\n"
                            + "\n\n".join(new_outputs)
                            + "\n\nPlease provide your final answer to the user synthesizing these results. "
                            "Do not output raw tool tags."
                        ),
                    },
                ]
                for msg in followup_messages:
                    msg["content"] = self._sanitize_prompt(msg["content"])

                f_resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": followup_messages,
                        "stream": False,
                        "options": {
                            "num_ctx": 16384,
                            "temperature": 0.15,
                            "top_p": 0.90,
                            "top_k": 20,
                            "repeat_penalty": 1.05,
                        },
                    },
                )
                if f_resp.status_code >= 400:
                    break
                f_resp.raise_for_status()
                content = sanitize_markdown_text(f_resp.json()["message"]["content"])

        cleaned_reply = re.sub(
            r'<(?:web_search|fetch_url)>.*?</(?:web_search|fetch_url)>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if not cleaned_reply and content:
            cleaned_reply = content

        return cleaned_reply, tool_results, agent_used

    async def agenerate_response(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Asynchronously query the model with the zero-shot prompt and web tools.

        Parameters
        ----------
        message : str
            The user's query.
        conversation_history : list[dict[str, str]] | None
            Optional prior conversation turns for multi-turn dialogue.

        Returns
        -------
        str
            The assistant's response text from the model.
        """
        reply_text, _, _ = await self.agenerate_response_with_tools(
            message=message,
            conversation_history=conversation_history,
        )
        return reply_text
