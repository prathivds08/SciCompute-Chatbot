"""
Kùzu embedded knowledge graph — stores structured domain facts.

Schema:
  Material  -[HAS_PSEUDO]->   Pseudopotential
  Material  -[PART_OF]->      System
  System    -[USES_WORKFLOW]-> Workflow
  System    -[TRAINED_WITH]->  DPModel
  System    -[HAS_TRAJECTORY]-> Trajectory
  Workflow  -[HAS_STEP]->     Step

Pre-populated with domain knowledge from the existing project notebooks
(pseudopotential mappings, type maps, temperatures, workflow sequences).
"""

from pathlib import Path
from typing import Any, Optional

import kuzu

from backend.config import settings


class KnowledgeGraph:
    """Manages the Kùzu embedded graph database for domain knowledge."""

    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = db_dir or settings.KUZU_DB_DIR
        # Kùzu manages the DB path itself; only ensure the parent dir exists
        self.db_dir.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self.db_dir))
        self._conn = kuzu.Connection(self._db)
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    #  Schema
    # ------------------------------------------------------------------ #

    def _ensure_schema(self) -> None:
        """Create node and relationship tables if they don't already exist."""
        # ---- Node tables ----
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS Material (
                symbol STRING,
                name STRING,
                cif_path STRING,
                PRIMARY KEY (symbol)
            )
        """)
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS Pseudopotential (
                filename STRING,
                pp_type STRING,
                xc_functional STRING,
                PRIMARY KEY (filename)
            )
        """)
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS System (
                name STRING,
                type_map STRING,
                temperature DOUBLE,
                lammps_style STRING,
                PRIMARY KEY (name)
            )
        """)
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS Workflow (
                name STRING,
                description STRING,
                PRIMARY KEY (name)
            )
        """)
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS Step (
                name STRING,
                script_template STRING,
                default_params STRING,
                PRIMARY KEY (name)
            )
        """)
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS DPModel (
                model_id STRING,
                seed INT64,
                graph_path STRING,
                rcut DOUBLE,
                descriptor_type STRING,
                PRIMARY KEY (model_id)
            )
        """)
        self._safe_exec("""
            CREATE NODE TABLE IF NOT EXISTS Trajectory (
                traj_id STRING,
                path STRING,
                format STRING,
                n_frames INT64,
                PRIMARY KEY (traj_id)
            )
        """)

        # ---- Relationship tables ----
        self._safe_exec("CREATE REL TABLE IF NOT EXISTS HAS_PSEUDO     (FROM Material TO Pseudopotential)")
        self._safe_exec("CREATE REL TABLE IF NOT EXISTS PART_OF        (FROM Material TO System)")
        self._safe_exec("CREATE REL TABLE IF NOT EXISTS USES_WORKFLOW  (FROM System   TO Workflow)")
        self._safe_exec("CREATE REL TABLE IF NOT EXISTS HAS_STEP       (FROM Workflow TO Step)")
        self._safe_exec("CREATE REL TABLE IF NOT EXISTS TRAINED_WITH   (FROM System   TO DPModel)")
        self._safe_exec("CREATE REL TABLE IF NOT EXISTS HAS_TRAJECTORY (FROM System   TO Trajectory)")

    # ------------------------------------------------------------------ #
    #  Domain data population
    # ------------------------------------------------------------------ #

    def populate_domain_knowledge(self) -> dict[str, int]:
        """Insert the baseline domain knowledge from the existing notebooks.

        Returns a dict of {entity_type: count_inserted}.
        """
        counts: dict[str, int] = {}

        # -- Materials & Pseudopotentials --
        materials_pseudos = [
            ("Pt", "Platinum",   "Pt.cif",  "Pt_ONCV_PBE-1.2.upf",  "ONCV",  "PBE"),
            ("Ir", "Iridium",    "",         "Ir_ONCV_PBE-1.2.upf",  "ONCV",  "PBE"),
            ("H",  "Hydrogen",   "",         "H_ONCV_PBE-1.2.upf",   "ONCV",  "PBE"),
            ("O",  "Oxygen",     "",         "O_ONCV_PBE-1.2.upf",   "ONCV",  "PBE"),
            ("C",  "Carbon",     "",         "C_ONCV_PBE-1.2.upf",   "ONCV",  "PBE"),
        ]
        for sym, name, cif, pp_file, pp_type, xc in materials_pseudos:
            self._safe_exec(
                "MERGE (m:Material {symbol: $sym}) "
                "ON CREATE SET m.name = $name, m.cif_path = $cif "
                "ON MATCH SET m.name = $name, m.cif_path = $cif",
                {"sym": sym, "name": name, "cif": cif},
            )
            self._safe_exec(
                "MERGE (p:Pseudopotential {filename: $fn}) "
                "ON CREATE SET p.pp_type = $pt, p.xc_functional = $xc "
                "ON MATCH SET p.pp_type = $pt, p.xc_functional = $xc",
                {"fn": pp_file, "pt": pp_type, "xc": xc},
            )
            self._safe_exec(
                "MATCH (m:Material {symbol: $sym}), (p:Pseudopotential {filename: $fn}) "
                "MERGE (m)-[:HAS_PSEUDO]->(p)",
                {"sym": sym, "fn": pp_file},
            )
        counts["materials"] = len(materials_pseudos)

        # -- Systems --
        systems = [
            ("Water",     '["H", "O"]',        300.0, "simple"),
            ("Interface", '["O", "H", "C"]',   330.0, "complex"),
            ("IrO2_Water", '["Ir", "O", "H"]', 300.0, ""),
        ]
        for name, tmap, temp, style in systems:
            self._safe_exec(
                "MERGE (s:System {name: $name}) "
                "ON CREATE SET s.type_map = $tmap, s.temperature = $temp, s.lammps_style = $style "
                "ON MATCH SET s.type_map = $tmap, s.temperature = $temp, s.lammps_style = $style",
                {"name": name, "tmap": tmap, "temp": temp, "style": style},
            )
        counts["systems"] = len(systems)

        # -- Workflows & Steps --
        workflows = [
            ("dft", "DFT convergence tests with Quantum ESPRESSO", [
                ("parse_structure",     "CIF parsing via ASE"),
                ("generate_inputs",     "Generate pw.in + qe.sub for each parameter value"),
                ("submit_sweep",        "Submit all jobs via submit.sh"),
                ("parse_convergence",   "Parse pw.out files and extract energies"),
                ("plot_convergence",    "Generate convergence plot"),
            ]),
            ("training", "DeePMD-kit model training", [
                ("scan_raw_data",       "Find *_Water directories under Raw_Files/"),
                ("create_test_split",   "Sample hold-out test set"),
                ("raw_to_set",          "Convert .raw files to set.*/ numpy arrays"),
                ("generate_configs",    "Create input.json for each seeded model"),
                ("submit_training",     "Submit SLURM training jobs"),
            ]),
            ("md", "LAMMPS molecular dynamics with DeepMD potentials", [
                ("resolve_graphs",      "Locate frozen .pb graph files"),
                ("validate_configs",    "Check starting configuration files"),
                ("generate_batches",    "Create batch_1..batch_N job directories"),
                ("create_submit_chain", "Write submit_from.sh with SLURM dependencies"),
                ("submit_md",           "Launch the MD batch chain"),
            ]),
            ("analysis", "Post-processing of MD trajectories", [
                ("load_trajectory",     "Read .extxyz trajectory via ASE"),
                ("identify_sites",      "Classify surface/bulk/water sites"),
                ("compute_density_rdf", "Density profiles and radial distribution functions"),
                ("compute_survival",    "Survival probability and hydroxyl fractions"),
                ("generate_plots",      "Create individual + combined 2x2 figures"),
                ("generate_report",     "Write summary_report.txt"),
            ]),
        ]
        step_count = 0
        for wf_name, wf_desc, steps in workflows:
            self._safe_exec(
                "MERGE (w:Workflow {name: $name}) "
                "ON CREATE SET w.description = $description "
                "ON MATCH SET w.description = $description",
                {"name": wf_name, "description": wf_desc},
            )
            for step_name, step_desc in steps:
                full_step = f"{wf_name}__{step_name}"
                self._safe_exec(
                    "MERGE (s:Step {name: $name}) "
                    "ON CREATE SET s.script_template = $description, s.default_params = '' "
                    "ON MATCH SET s.script_template = $description",
                    {"name": full_step, "description": step_desc},
                )
                self._safe_exec(
                    "MATCH (w:Workflow {name: $wf}), (s:Step {name: $step}) "
                    "MERGE (w)-[:HAS_STEP]->(s)",
                    {"wf": wf_name, "step": full_step},
                )
                step_count += 1

        counts["workflows"] = len(workflows)
        counts["steps"] = step_count

        # -- Link systems to workflows --
        system_workflows = [
            ("Water",      ["dft", "training", "md", "analysis"]),
            ("Interface",  ["training", "md", "analysis"]),
            ("IrO2_Water", ["analysis"]),
        ]
        for sys_name, wfs in system_workflows:
            for wf in wfs:
                self._safe_exec(
                    "MATCH (s:System {name: $sys}), (w:Workflow {name: $wf}) "
                    "MERGE (s)-[:USES_WORKFLOW]->(w)",
                    {"sys": sys_name, "wf": wf},
                )

        # -- Link materials to systems --
        material_systems = [
            ("H", "Water"), ("O", "Water"),
            ("H", "Interface"), ("O", "Interface"), ("C", "Interface"),
            ("Ir", "IrO2_Water"), ("O", "IrO2_Water"), ("H", "IrO2_Water"),
        ]
        for mat, sys_name in material_systems:
            self._safe_exec(
                "MATCH (m:Material {symbol: $mat}), (s:System {name: $sys}) "
                "MERGE (m)-[:PART_OF]->(s)",
                {"mat": mat, "sys": sys_name},
            )

        return counts

    # ------------------------------------------------------------------ #
    #  Querying
    # ------------------------------------------------------------------ #

    def query(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        result = self._conn.execute(cypher, parameters=params or {})
        rows: list[dict] = []
        while result.has_next():
            row_values = result.get_next()
            col_names = result.get_column_names()
            rows.append(dict(zip(col_names, row_values)))
        return rows

    def get_pseudopotential(self, element_symbol: str) -> Optional[str]:
        """Look up the pseudopotential filename for an element."""
        rows = self.query(
            "MATCH (m:Material {symbol: $sym})-[:HAS_PSEUDO]->(p:Pseudopotential) RETURN p.filename AS filename",
            {"sym": element_symbol},
        )
        return rows[0]["filename"] if rows else None

    def get_system_info(self, system_name: str) -> Optional[dict]:
        """Get full system details including type_map, temperature, style."""
        rows = self.query(
            "MATCH (s:System {name: $name}) "
            "RETURN s.name AS name, s.type_map AS type_map, "
            "s.temperature AS temperature, s.lammps_style AS lammps_style",
            {"name": system_name},
        )
        return rows[0] if rows else None

    def get_workflow_steps(self, workflow_name: str) -> list[dict]:
        """Get all steps for a workflow in order."""
        return self.query(
            "MATCH (w:Workflow {name: $wf})-[:HAS_STEP]->(s:Step) "
            "RETURN s.name AS name, s.script_template AS description "
            "ORDER BY s.name",
            {"wf": workflow_name},
        )

    def get_system_materials(self, system_name: str) -> list[str]:
        """Get all element symbols that are part of a system."""
        rows = self.query(
            "MATCH (m:Material)-[:PART_OF]->(s:System {name: $sys}) RETURN m.symbol AS symbol",
            {"sys": system_name},
        )
        return [r["symbol"] for r in rows]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _safe_exec(self, cypher: str, params: Optional[dict[str, Any]] = None) -> None:
        """Execute a Cypher statement, ignoring 'already exists' errors."""
        try:
            self._conn.execute(cypher, parameters=params or {})
        except RuntimeError as exc:
            if "already exists" in str(exc).lower():
                pass
            else:
                raise

    def close(self) -> None:
        """Close the database connection."""
        # Kùzu Python API handles cleanup on GC, but explicit close is polite
        self._conn = None  # type: ignore[assignment]
        self._db = None  # type: ignore[assignment]
