# MD_Simulation_Script.ipynb

Generates a full chain of chunked LAMMPS production-MD job folders (`in.lammps` + SLURM submission scripts) for DeepMD-driven runs, covering both the **Water** and **Interface** (graphene-water) systems in one pass. Designed to run long trajectories (e.g. 5 ns) as a series of restart-linked batches (e.g. 500 ps each) so a crash/power failure only loses the in-progress batch.

Intended to be run from inside `TransferLearning_Corrected_Myself/` (see [[transfer-learning-potentials]] directory layout).

## What it does
1. Prompts for run parameters (see table below).
2. Validates that total run length is an exact multiple of the batch length; computes `num_batches` and `batch_steps` (`batch_time_ps / DT`, with `DT = 0.0005` ps fixed).
3. For each model in `MODELS` (currently just `DPMD`) × each system in `SYSTEMS` (`Interface`, `Water`):
   - Resolves the frozen-graph paths: `Frozen_Graphs/<model>_<system>/<i>/graph<i>.pb` for `i = 1..graph_num`.
   - Picks the starting configuration file and LAMMPS script "style":
     - **Water** → `Long_Run/Starting_Configurations/Water_DPMD_Validation.lmp`, **`simple` style** (mirrors a minimal reference water script: single `fix nvt` at `tau=0.1`, no equilibration staging, DCD dump, literal parameter values).
     - **Interface** → `Long_Run/Starting_Configurations/Graphene_Water_DPMD_Validation_file.lmp`, **`complex` style** (LAMMPS variables, momentum-zeroing fix, two-stage equilibration + production, custom `thermo_output.txt` logging via `fix print`, per-atom dump).
   - Writes one folder per batch: `Long_Run/<model>_<system>/batch_<n>/` containing `in.lammps`, `lmp.sub`, and `run_job.sh`.
     - Batch 1 does `read_data` from the starting-configuration file and does `velocity create` at the system's temperature.
     - Every later batch does `read_restart` from `batch_<n-1>/lammps.restart` instead, and (for `complex` style) appends to `thermo_output.txt`/dump rather than overwriting.
     - Every batch ends with `write_restart lammps.restart` and a `timer timeout <HH:MM:SS> every 100` safety cutoff.
   - Writes `Long_Run/<model>_<system>/submit_from.sh <start_batch>` — chains all remaining batches via `sbatch --dependency=afterok:<prev_jobid>`, so the whole run (or a resume from any batch) is one command.

## Per-system fixed parameters (hardcoded in-script)
| | Water | Interface |
|---|---|---|
| Temperature | 300 K | 330 K |
| Atom types (mass) | 1=H (2.014), 2=O (15.999) | 1=O (15.999), 2=H (1.00794), 3=C (12.0107) |
| LAMMPS style | `simple` | `complex` |
| Thermostat | `fix nvt` tau=0.1 (metal units, literal) | `fix nvt` tau=`100*dt` (variable-driven) |
| Dump | DCD, every 100 steps | custom (id type x y z), every `DUMP_FREQ` |

> Note: `TEMP_BY_SYSTEM` in-code has a comment "water lowered to 300 per request" — the [[transfer-learning-potentials]] notes elsewhere say runs use 330 K, so double-check which temperature is current before submitting.

## Inputs (interactive prompts)
| Prompt | Example | Meaning |
|---|---|---|
| Username | `ch24d003` | For SLURM `--mail-user=<user>@smail.iitm.ac.in` |
| Total run length (ps) | `5000` | Must be a multiple of the batch length |
| Batch chunk length (ps) | `500` | e.g. 10 batches for a 5000 ps total |
| Timer timeout per batch | `23:55:00` | Also used as the SLURM `--time` |
| Number of graphs/models | `3` | How many `graph<i>.pb` frozen graphs to load (>1 enables `out_file md.out out_freq 100` on `pair_style deepmd`) |
| CPU or GPU | `CPU` / `GPU` | Selects `lmp.sub` template |

## Output layout
```
Long_Run/<model>_<system>/
├── batch_1/
│   ├── in.lammps
│   ├── lmp.sub
│   └── run_job.sh
├── batch_2/
│   └── ...
└── submit_from.sh
```

## SLURM templates generated
- **CPU**: 2 nodes × 48 tasks/node, `standard` partition, `OMP_NUM_THREADS=2`, `mpirun -np 96 .../deepmd/bin/lmp`.
- **GPU**: 1 node, 1 GPU, `gpu` partition, `srun .../deepmd_gpu/bin/lmp`.

## Usage
```bash
cd TransferLearning_Corrected_Myself/
jupyter nbconvert --to script --execute MD_Simulation_Script.ipynb   # or run interactively
# then, per combo:
bash Long_Run/DPMD_Water/submit_from.sh 1
bash Long_Run/DPMD_Interface/submit_from.sh 1
# resume after a failure at batch N:
bash Long_Run/DPMD_Water/submit_from.sh N
```

## Known limitations
- `MODELS = ["DPMD"]` and `SYSTEMS = ["Interface", "Water"]` are hardcoded lists at the top of the cell — extending to BPNN/NequIP graphs means adding entries here (the graph-path logic and `pair_style deepmd` already generalize to any model name, since all three model types load through the same DeepMD LAMMPS plugin).
- All paths are relative to `ROOT = "."`, i.e. wherever the notebook is launched from — must be run from `TransferLearning_Corrected_Myself/`.
- No validation that the frozen-graph files or starting-configuration `.lmp` files actually exist before writing job scripts that reference them.
