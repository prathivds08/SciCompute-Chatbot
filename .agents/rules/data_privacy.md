---
name: data_privacy_and_docker_execution
description: Prohibits direct model access to raw extxyz trajectory datasets while permitting Docker-based code execution.
---

# Dataset Privacy & Container Execution Rules

- **Direct Data Access**: Strictly prohibited. The model must not inspect, parse, or output raw contents of `backend/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`.
- **RAG & Vector Storage**: The file is blacklisted from vector stores and cannot be embedded.
- **Docker Execution**: Allowed. The model can write Python code that runs inside Docker, referencing `/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` (or `DATA_DIR / 'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'`).
- **Allowed Outputs**: Plots, summary statistics, and derived scientific metrics.
