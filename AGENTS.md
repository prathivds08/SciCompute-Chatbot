# Workspace Rules for AI Models and Agents

## Dataset Access & Execution Policy

### 1. Direct Data Access Restriction
- The AI model/agent is **strictly prohibited from directly reading, inspecting, displaying, or regurgitating the raw contents** of `backend/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` (or any `.extxyz` file).
- Do not dump atomic coordinates, forces, lattice vectors, or trajectory frames directly into prompt context or chat responses.
- The model must never ingest this file into vector stores or retrieval databases.

### 2. Docker Container Execution Allowed
- The AI model **CAN write and execute Python scripts in Docker that call this dataset**.
- In the Docker execution environment (`deepmd-analysis-sandbox`), the dataset is mounted read-only at `/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` (accessible via `DATA_DIR / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"`).
- Containerized scripts are permitted to load the structure via scientific libraries (e.g., `ase.io.read`), compute derived metrics (e.g., interfacial mass density profiles, radial distribution functions, DeePMD raw set conversions), and save figures/summary reports to the `/workspace` output directory.
- Only aggregate results, key numerical metrics, and generated graphical representations (plots) are returned to the user.
