# Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb

Post-processes DeepMD/LAMMPS `.extxyz` trajectories of a metal-oxide/water interface (template: IrO₂, but the metal symbol and stoichiometry are user inputs) to characterize interfacial water structure and surface hydroxylation kinetics, averaged across multiple DP (Deep Potential) model trajectories. Organized as independently runnable blocks: two analysis+save pipelines, their matching plot cells, and a combined 4-panel summary figure + text report.

## Structure
| Cell | Role |
|---|---|
| 0 | Imports (ASE, `pwtools`, matplotlib, `NeighborList`) |
| 1 | Comment-only overview of the density/RDF analysis |
| 2 | **Density + RDF analysis** → saves `simulation_results.npz` |
| 3 | **Density + RDF plots** (separate figures) — loads `simulation_results.npz` |
| 4 | Comment-only overview of the survival/dissociation analysis |
| 5 | **Survival probability + hydroxyl fraction analysis** → saves `survival_fraction_results.npz` |
| 6 | **Survival + fraction plots** (separate figures) — loads `survival_fraction_results.npz` |
| 7 | **Combined 2×2 summary figure** (all four analyses) + `summary_report.txt` — loads both `.npz` files |
| 8 | Empty |

Each analysis cell (2, 5) and its plotting cell (3, 6) can be re-run independently as long as the corresponding `.npz` exists — you don't need to re-run the (slow) trajectory processing just to change plot styling.

## What it computes

### 1. Site identification (cells 2 & 5, done once per model from frame 0)
- **Surface metal (M_sur):** metal atoms within 1.5 Å (in *z*) of the topmost metal layer.
- **Bulk oxygen:** the `s × N_metal` lowest-*z* oxygens (`s` = O stoichiometry, e.g. 2 for IrO₂), i.e. lattice oxygen accounted for by the ideal formula.
- **Water oxygen (O_w):** any oxygen not classified as bulk.
- **CUS metal / adsorbed oxygen (O_adb):** surface metal atoms with a water oxygen within 2.5 Å are "coordinatively unsaturated" (Cus) sites; that water oxygen becomes O_adb.
- **Surface/bridge oxygen (O_sur / O_br):** the lattice oxygens just below the bulk cutoff, count matched to the number of Cus sites found.

### 2. Water density profile + RDFs (cell 2)
- Bins O_w and all H positions along *z* (bin width 0.10 Å) every 10th frame, converts counts to mass density (g/cm³) via `ρ = (N/N_A × 18.015) / (bin volume in cm³)`.
- Temporarily relabels atoms (`O_w→W`, `M_cus→Si`, `O_sur→S`) so `pwtools.crys.rpdf` can compute three pair RDFs: `M_cus–O_w`, `O_br–H_w`, `O_br–O_w` (`dr=0.05` Å).
- Averages per-frame density/RDF within each model, then stores one curve per DP model in `simulation_results.npz` (keys: `m_symbol, ow_y, hw_y, bins_z, rd, g_mow, g_osh, g_osow`) — cross-model mean ± std is computed at plot time.

### 3. Hydroxyl fraction + survival probability (cell 5)
- Auto-detects the trajectory frame spacing from `.info['time']` (falls back to 0.05 ps/frame = 0.0005 ps timestep × 100-step dump interval if absent), combined with `analysis_stride=10` to get ns/frame.
- Per frame, uses `ase.neighborlist.neighbor_list` (O–H cutoff 1.2 Å) to classify each O_adb site as **OH** (1 H neighbor, dissociated) or **H₂O** (2 H neighbors, intact), and tracks the % of adsorbed sites in each state over time.
- **Survival probability P(t):** for 101 log-spaced-in-index time lags τ, computes the fraction of a given state's index set at frame *j* that remains in that same set for the entire interval `[j, j+τ]` (via successive set intersections), averaged over all start frames *j*.
- Saves `survival_fraction_results.npz` (keys: `m_symbol, times_surv, surv_oh, surv_h2o, frac_oh, frac_h2o, dt_ns`).

### 4. Plotting (cells 3, 6, 7)
- Cells 3 & 6 make separate per-analysis figures with shaded ±1 std bands across DP models (this is the "separate plot capability" in the notebook's filename).
- Cell 7 loads **both** `.npz` files and lays out a 2×2 summary figure (density / RDF / survival / dissociation fraction), saved as `full_analysis_presentation.png`, plus a `summary_report.txt` with residence times (via trapezoidal integration of P(t)) and average dissociated/molecular fractions ± std across models.

## Inputs (interactive prompts, asked separately in cells 2 and 5 — must match)
| Prompt | Example | Notes |
|---|---|---|
| Metal chemical symbol | `Ir` | |
| O stoichiometry | `2` (IrO₂) | Used to split bulk vs. surface/water oxygen |
| Number of DP models | `4` | Trajectories at `<base_dir>/<1..dp>/iro2_water_concat.extxyz` |
| Base directory path | `/scratch/.../IrO2_Runs/` | **Must end with a trailing slash** — it's string-concatenated directly (`base_dir + str(mod_idx) + '/...'`) |

## Expected directory layout
```
<base_dir>/
├── 1/iro2_water_concat.extxyz
├── 2/iro2_water_concat.extxyz
├── ...
```
Output files (`simulation_results.npz`, `survival_fraction_results.npz`) are written to `base_dir`, but cell 7 loads them from the **current working directory** (no `base_dir` prefix) — copy/run from `base_dir`, or adjust the paths in cell 7, or the combined figure will raise `FileNotFoundError`.

## Prerequisites
- Python packages: `ase`, `numpy`, `matplotlib`, `pwtools` (for `rpdf`/`atoms2struct`/`Trajectory`).
- Trajectories in `.extxyz` format with per-frame `time` info (optional but enables accurate ns scaling; otherwise falls back to a hardcoded 0.05 ps/frame).

## Known limitations / things to check before reuse
- **Labels say OD/D₂O, math says H/H₂O.** Cells 5–7 print/plot labels `OD` and `D₂O`, and `dp/model` docstrings mention deuterium, but the actual O–H neighbor cutoff, molar mass (18.015 g/mol, i.e. H₂O not D₂O), and `neighbor_list('ijd', frame, {('O','H'): 1.2})` call are all light-hydrogen physics. If these are genuinely D₂O trajectories, the mass-density formula and bond cutoff should be adjusted for deuterium; if they're H₂O, the OD/D₂O labels in cells 5–7 are just mislabeled and should read OH/H₂O (matching cells 1–3, which correctly use O_w/H_w).
- Cell 2's `overall_start_time = time.perf_counter()` uses a `time` module imported both at top-level (cell 0) and again inside cell 2 — harmless but redundant.
- Site identification (Cus/bulk/surface) is computed once from **frame 0 of model 1 only** (cell 2) or **frame 0 of each model independently** (cell 5) — a slightly different, per-model site list in cell 5 vs. a single shared list in cell 2 is a real inconsistency between the two halves of the notebook if the surface reconstructs differently across models.
- `o_bul_cutoff` in cell 5 uses a single-oxygen z-value cutoff (`<=`) rather than the index-based bulk list cell 2 builds — a subtly different (simpler, single-threshold) definition of "bulk" than cell 2's exact-match index lookup; fine for well-separated slabs, worth checking for edge cases where two oxygens sit at the same z-cutoff altitude.
- Both `rpdf` (cell 2) and `neighbor_list` (cell 5) calls assume the relabeled/original chemical symbols are unique enough for correct masking — if the real system already contains an element sharing a placeholder symbol (`Si`, `W`, `S`), the RDF masking in cell 2 would misassign atoms.
- No check that `dp` matches the number of trajectory folders that actually exist; a missing folder in cell 2 will raise `FileNotFoundError` and abort (cell 5 catches this per-model and just skips it, so the two cells behave differently on missing data).
