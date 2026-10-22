# EnergyCutoffandKpointsScript.ipynb

Generates and analyzes Quantum ESPRESSO (`pw.x`) SCF convergence tests for `ecutwfc` (plane-wave energy cutoff) or k-point mesh density, starting from a single CIF structure file. Two cells: one **generates** the job directories/inputs, the other **parses** the finished runs and plots convergence.

## What it does

### Cell 1 — Job generator
1. Reads a structure from a CIF file via ASE (`ase.io.read`).
2. Asks whether you're testing `energycutoff` (fixed k-mesh, varying `ecutwfc`) or `kpoints` (fixed `ecutwfc`, varying k-mesh).
3. For every value in the sweep, creates a subfolder (`<cif_dir>/<mode>/<value>/`) containing:
   - `pw.in` — QE SCF input (`ibrav=0`, PBE, Marzari-Vanderbilt smearing, `degauss=0.02`, `ecutrho = 4×ecutwfc`), written via `ase.calculators.espresso`/`ase.io.write`.
   - `qe.sub` — SLURM batch script (2 nodes × 24 tasks, `standard` partition, 1 hr walltime) that runs `pw.x`.
4. Writes a top-level `submit.sh` in `<cif_dir>/<mode>/` that loops over every subfolder and `sbatch`es its `qe.sub`.

### Cell 2 — Convergence analysis / plotting
1. Re-prompts for the CIF path and mode (independent of cell 1 — can be run later/separately).
2. Scans `<cif_dir>/<mode>/*/pw.out`, reads total energy from each finished job with `ase.io.read(..., format='espresso-out')`.
3. Prints a table of value / total energy (eV) / ΔE vs. the previous point.
4. Plots energy vs. cutoff (or k-mesh) and saves a publication-style PNG: `Convergence_<mode>_<cif_basename>.png`.

## Prerequisites
- Python packages: `ase`, `numpy`, `matplotlib`.
- A pseudopotential file matching the element(s) in the CIF, present at `../../pseudo/` relative to the output directory (path is hardcoded as `pseudo_dir = "./../../pseudo/"`).
- QE 6.7 installed at `$HOME/qe-6.7/bin/pw.x` on the HPC cluster (hardcoded in `qe.sub`).

## Inputs (interactive prompts)
| Prompt | Example | Notes |
|---|---|---|
| CIF path | `./Pt.cif` | Relative to notebook's working directory |
| Mode | `energycutoff` or `kpoints` | |
| ecutwfc value(s) *(energycutoff mode)* | `60 80 100` | Space-separated, Ry |
| Fixed k-mesh *(energycutoff mode)* | `4 4 4` | |
| k-mesh value(s) *(kpoints mode)* | `4 6 8` | Builds `n×n×n` for each `n` |
| Fixed ecutwfc *(kpoints mode)* | `100` | |

## Output layout
```
<cif_dir>/<mode>/
├── <value_1>/
│   ├── pw.in
│   └── qe.sub
├── <value_2>/
│   └── ...
└── submit.sh
```

## Known limitations / things to check before reuse
- **Pseudopotential dictionary is hardcoded** to `{'Pt': 'Pt_ONCV_PBE-1.2.upf'}` — must be edited in-cell for any element other than Pt (the comment referencing Si/Mg/O is stale and doesn't match the dict).
- `nbnd: 44` is hardcoded in the QE `system` namelist — appropriate for Pt's electron count at the tested cell size, not general-purpose.
- The k-points plot parser assumes folder names like `k_4`, `k_6`; it strips the `k_` prefix to get the x-axis value.
- Cell 2 silently skips any `pw.out` it can't parse (bare `except: continue`), so an incomplete/crashed job just disappears from the plot rather than raising a visible error — check the printed table row count against the number of submitted jobs.
