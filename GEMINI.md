# Gemini Model Guidelines

## Dataset Access & Docker Execution Policy

1. **No Direct Data Access**:
   The model must NOT have direct access to, or display the raw data of:
   `backend/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`
   Do not read, parse, or output raw atomic coordinate arrays, forces, or xyz blocks in responses.

2. **Docker Calling Policy**:
   The model CAN write Python code that runs inside the Docker container to call and process this dataset:
   - File path in Docker: `/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`
   - Volume mount: `backend/data` is mounted to `/data` (read-only)
   - Permitted operations inside Docker: Reading via `ase.io.read`, computing water density profiles, radial distribution functions (RDF), DeePMD dataset conversions, and rendering matplotlib figures.
   - Returned outputs: Standard output summaries, metrics, and generated image artifacts (`.png`).
