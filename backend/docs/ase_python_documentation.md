# ASE (Atomic Simulation Environment) Python API Reference

This document serves as the canonical reference for generating correct, crash-free Python code using the Atomic Simulation Environment (`ase`).

---

## 1. Reading Structures & Trajectories (`ase.io.read`)

The `ase.io.read()` function has fundamentally different return types depending on the `index` parameter:

| `index` Parameter | Return Type | Description & Usage |
| :--- | :--- | :--- |
| `index=0` (or integer, e.g. `-1`) | **`ase.Atoms`** | Returns a **single frame**. Use this for static analysis, single structure inspection, or single-point calculations. |
| `index=":"` (slice string) | **`list[ase.Atoms]`** | Returns a **list of frames** (trajectory). You **CANNOT** call `Atoms` methods directly on this list! |

### Mandatory Pattern: Single Structure Inspection & Property Analysis
To load a structure and extract positions, cell, energy, forces, or boundary atoms, **ALWAYS use `index=0`**:
```python
import numpy as np
from ase.io import read

# [V] MANDATORY PATTERN for single structure / frame:
atoms = read("structure.extxyz", index=0)  # Returns single Atoms object

# Now all Atoms methods work directly:
positions = atoms.get_positions()            # Shape (N, 3) in Angstrom
symbols = np.array(atoms.get_chemical_symbols())
cell = atoms.cell                            # 3x3 Lattice Cell
scaled_positions = atoms.get_scaled_positions(wrap=True) # Fractional [0, 1)
atom_count = len(atoms)                      # Always len(atoms)
```

> **WARNING (Fatal Error to Avoid)**:
> Never use `index=":"` when inspecting a structure! Calling `read(..., index=":")` returns a Python `list[Atoms]`. Calling `atoms.get_positions()` on a list immediately crashes with `AttributeError: 'list' object has no attribute 'get_positions'`. Only use `index=":"` when iterating across all frames in a trajectory (`for frame in frames:`).

---

## 2. Canonical `ase.Atoms` Methods & Properties

Always use the official ASE methods on a single `Atoms` object:

### System Geometry & Dimensions
- **Atom Count**:
  - `len(atoms)`: Total number of atoms.
  - *Anti-patterns (DO NOT EXIST)*: `atoms.get_count()`, `atoms.count()`, `atoms.num_atoms()`.
- **Cell & Lattice**:
  - `atoms.cell`: An `ase.cell.Cell` object representing the 3x3 lattice vectors.
  - `atoms.get_cell()`: Returns the `Cell` object.
  - `atoms.cell.lengths()`: 1D array of cell vector lengths `[a, b, c]` in Ångström.
  - `atoms.cell.angles()`: 1D array of cell angles `[alpha, beta, gamma]` in degrees.
  - `atoms.get_volume()`: Volume of the unit cell in Å³.
  - `atoms.pbc`: Periodic boundary condition flags `[pbc_x, pbc_y, pbc_z]` (e.g. `[True, True, True]`).
- **Positions**:
  - `atoms.get_positions()`: Cartesian coordinates array of shape `(N, 3)` in Ångström.
  - `atoms.get_scaled_positions(wrap=True)`: Fractional coordinates in the range `[0, 1)`. **Always pass `wrap=True`** when working with periodic boundary conditions!
- **Chemical Elements**:
  - `atoms.get_chemical_symbols()`: List of element strings `['Ti', 'O', 'H', ...]`.
  - `atoms.get_atomic_numbers()`: NumPy array of atomic numbers (e.g. `[22, 8, 1, ...]`).

### Energy & Forces
- **Potential Energy**:
  - `atoms.get_potential_energy()`: Potential energy in electronvolts (eV).
  - *Anti-patterns*: `atoms.get_energy()`, `atoms.energy` (DO NOT EXIST).
- **Forces**:
  - `atoms.get_forces()`: Force array of shape `(N, 3)` in eV/Å.

---

## 3. Pairwise Distances and Neighbor Finding

### Method A: All Pairwise Distances with PBC
```python
import numpy as np

# Compute all pairwise distances with Minimum Image Convention (mic=True)
dists = atoms.get_all_distances(mic=True)  # Shape (N, N)

# Exclude self-interaction (diagonal is 0.0)
np.fill_diagonal(dists, np.inf)

# Find minimum distance and check overlap
min_dist = np.min(dists)
has_overlaps = bool(np.any(dists < 1e-5))  # Vectorized check
```

### Method B: Distance Between Specific Atom Subsets
```python
# Distances between indices_i and indices_j
vectors, distances = atoms.get_distances(indices_i, indices_j, mic=True)
```

### Method C: Memory-Efficient Neighbor List
```python
from ase.neighborlist import neighbor_list

# 'i': atom index, 'j': neighbor index, 'd': distance
i_idx, j_idx, dist_ij = neighbor_list('ijd', atoms, cutoff=3.0)

# Vectorized filtering for bonded atoms:
bonded_mask = dist_ij < 1.2
bonded_i = i_idx[bonded_mask]
bonded_j = j_idx[bonded_mask]
```

---

## 4. Standard Unit Conversions (`ase.units`)

Always import conversion constants from `ase.units`:
```python
from ase.units import Hartree, Bohr, Ang, eV, fs, ps

# Convert potential energy from eV to Hartree:
energy_Hartree = atoms.get_potential_energy() / Hartree

# Convert Bohrs to Angstroms:
length_Ang = length_Bohr * Bohr
```
