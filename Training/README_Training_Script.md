# Training_Script.ipynb

Converts raw DeePMD-kit training data (`box.raw`, `coord.raw`, `energy.raw`, ...) into `set.*` binary training sets, carves out a held-out test split, and generates a batch of DeepMD `input.json` + SLURM training jobs (one per random seed) for every `*_Water` system found under `Raw_Files/`.

## What it does
1. Prompts for a SLURM username and the number of DP models (seeds) to train per system.
2. Scans `Raw_Files/` for directories ending in `_Water`.
3. For each such system directory that has `box.raw` + `coord.raw`:
   - Reads `coord.raw`, `energy.raw`, `force.raw`, `box.raw`, `virial.raw` (whichever exist and are non-empty) line by line.
   - Randomly samples up to 100 configurations (all of them if fewer than 100 exist) as a **held-out test set**, written to a new `Test/` subfolder, and copies over `type.raw`/`type_map.raw` unchanged.
   - Writes a `raw_to_set.sh` script into both the training directory and `Test/`. This script:
     - Splits each `*.raw` file into chunks of 2000 lines (configurable via `$1`) using `split`.
     - Converts each chunk into a `set.NNN/` folder of `.npy` arrays (via an embedded `python3` heredoc using `numpy.loadtxt`/`np.save`), matching the `set.*` layout DeepMD-kit expects.
   - Runs `raw_to_set.sh` automatically (via `subprocess.run`) in both the train and test directories for every processed system.
4. For each system, generates `graph_num` DeepMD training jobs under `DP_Models/<system>/<1..graph_num>/`, each containing:
   - `input.json` — a `se_e2_a` descriptor DeepMD-kit v1-style config (see table below), pointing `training_data`/`validation_data` at the relative paths of the train/`Test` set directories.
   - `train.sub` — SLURM script (1 node, 1 GPU, `gpu` partition, 24 hr) running `python -m deepmd train input.json`.

## DeepMD `input.json` key settings (fixed per model)
| Section | Setting | Value |
|---|---|---|
| descriptor | type / rcut_smth / rcut / neuron / axis_neuron | `se_e2_a` / 3.0 / 6.0 Å / [120,120,120] / 12 |
| descriptor | sel | [120, 64] (max H, O neighbors) |
| fitting_net | neuron / resnet_dt | [120,120,120] / True |
| learning_rate | type / start_lr / decay_steps / decay_rate | exp / 0.005 / 5000 / 0.95 |
| loss | pref_e (start→limit) / pref_f (start→limit) / pref_v | 0.02→8 / 1000→1 / 0 (unused) |
| training | stop_batch / batch_size / numb_test | 1,000,000 / 1 / 10 |

`type_map` is fixed to `["H", "O"]` (pure water only — no carbon type, so this script is water-system-specific, unlike the graphene-water interface handled elsewhere in the project).

Seeds are hardcoded: `[1, 6534653463541, 652465462461]`. If `graph_num` exceeds 3, every extra model reuses the **last** seed in the list (`seed_values[-1]`) rather than generating a new one — worth checking before requesting more than 3 models.

## Inputs (interactive prompts)
| Prompt | Example | Meaning |
|---|---|---|
| Username | `ch24d003` | For SLURM `--mail-user=<user>@smail.iitm.ac.in` |
| Number of DP_models per Water system | `3` | How many seeded `input.json`/`train.sub` pairs to generate per system (≤3 gets distinct hardcoded seeds) |

## Required directory layout before running
```
<working_dir>/
├── Raw_Files/
│   └── <SystemName>_Water/
│       ├── box.raw
│       ├── coord.raw
│       ├── energy.raw
│       ├── force.raw
│       └── virial.raw          (optional but recommended)
└── DP_Models/                   # must already exist (script exits if missing)
```

## Output layout
```
Raw_Files/<SystemName>_Water/
├── set.000/, set.001/, ...      # training sets (.npy)
└── Test/
    ├── (100 sampled raw lines per file, before conversion)
    └── set.000/, ...            # test sets (.npy)

DP_Models/<SystemName>_Water/
├── 1/
│   ├── input.json
│   └── train.sub
├── 2/
│   └── ...
```

## Usage
```bash
cd <project_dir>   # containing Raw_Files/ and DP_Models/
jupyter nbconvert --to script --execute Training_Script.ipynb
# then submit:
for d in DP_Models/*_Water/*/; do (cd "$d" && sbatch train.sub); done
```

## Known limitations
- Only processes systems whose directory name ends in `_Water` — a differently-named system (or the graphene-water interface data) won't be picked up by this script.
- `nsym_list = ["H", "O"]` (the DeepMD `type_map`) is hardcoded, so it silently mislabels any system with different chemistry rather than erroring.
- Test-set sampling is done on the **raw text files before** the `set.*` conversion, and training-set conversion is done on the **full original raw files** (not with the 100 sampled configs excluded) — the same configurations that land in `Test/` may still be present in the training `set.*` folders. Worth confirming this is the intended train/test policy, or exclude the sampled indices from the training raws if strict separation is needed.
- `raw_to_set.sh`'s inner `python3` heredoc silently swallows any conversion error (bare `except: pass`) — a malformed `.raw` file produces a missing `.npy` with no visible warning at generation time.
