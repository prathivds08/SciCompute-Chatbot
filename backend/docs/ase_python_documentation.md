# ASE (Atomic Simulation Environment) Python API Reference & Documentation

This document is the canonical reference for generating correct, robust, crash-free scientific Python code with the Atomic Simulation Environment (`ase`). It details correct function signatures, return types, common pitfalls, and verified code patterns.

---

## 1. Structure & Trajectory I/O (`ase.io.read`, `ase.io.write`)

### Reading Structures (`from ase.io import read`)

`ase.io.read(filename, index=None, format=None, **kwargs)` has fundamentally different return types depending on `index`:

| `index` Parameter | Return Type | Intended Use Case |
| :--- | :--- | :--- |
| `index=0` (or integer, e.g. `-1`) | **`ase.Atoms`** | **Single structure / frame**. Use for static analysis, single-point inspection, geometry checks, density profiles, and surface site identification. |
| `index=":"` (slice string) | **`list[ase.Atoms]`** | **Multi-frame trajectory**. Use ONLY when iterating frame-by-frame (`for frame in frames:`). |

```python
from ase.io import read
import numpy as np

# [V] CANONICAL SINGLE-FRAME LOADING PATTERN:
atoms = read("structure.extxyz", index=0)  # Returns ONE ase.Atoms object

# All Atoms attributes and methods are directly accessible:
positions = atoms.get_positions()            # Shape (N, 3), float64, in Angstrom
symbols = np.array(atoms.get_chemical_symbols()) # 1D array of strings ['Ti', 'O', 'H', ...]
cell = atoms.cell                            # ase.cell.Cell object (3x3 lattice vectors)
cell_lengths = atoms.cell.lengths()          # [a, b, c] in Angstrom
scaled = atoms.get_scaled_positions(wrap=True) # Fractional coordinates in [0, 1)
atom_count = len(atoms)                      # ALWAYS len(atoms), NEVER atoms.get_count()
energy_eV = atoms.get_potential_energy()     # Scalar potential energy in eV
forces = atoms.get_forces()                  # Shape (N, 3) in eV/Angstrom
```

> **FATAL ERROR TO AVOID**:
> Never call `read(..., index=":")` when you only need a single structure! Calling `read(path, index=":")` returns a Python `list[Atoms]`. Calling `atoms.get_positions()` on a list immediately crashes with:
> `AttributeError: 'list' object has no attribute 'get_positions'`.

---

### Writing Structures (`from ase.io import write`)

`ase.io.write(filename, images, format=None, **kwargs)` writes an `Atoms` object or a list of `Atoms` objects to disk (supports `.extxyz`, `.xyz`, `.cif`, `.pdb`, `POSCAR`, LAMMPS data, etc.).

```python
from ase import Atoms
from ase.io import write

# [V] CREATING AND SAVING A NEW ATOMS OBJECT:
# 1. Always construct a new Atoms object from the class `ase.Atoms`:
# Note: `atoms` returned by read() is an INSTANCE. Do NOT call `atoms(...)`!
new_structure = Atoms(
    symbols=boundary_symbols,        # e.g. ['Ti', 'Ti', 'O', ...]
    positions=boundary_positions,    # Shape (M, 3) in Angstrom
    cell=atoms.cell,                 # Inherit original 3x3 lattice cell
    pbc=atoms.pbc,                   # Inherit periodic boundary flags [True, True, True]
)

# 2. Write to extxyz or other supported formats:
write("boundary_atoms.extxyz", new_structure)
```

---

## 2. Distance Calculations (`from ase.geometry import get_distances`)

### Function Signature & Crucial Rules

```python
from ase.geometry import get_distances

# Function signature:
# get_distances(p1, p2=None, cell=None, pbc=None)
```

- **Arguments**:
  - `p1`: Cartesian coordinates array of shape `(N, 3)` (or single point `(1, 3)`).
  - `p2`: Optional Cartesian coordinates array of shape `(M, 3)`. If `None`, pairwise distances between all points in `p1` are computed.
  - `cell`: 3x3 lattice vectors array or `atoms.cell`.
  - `pbc`: Periodic boundary condition flags, e.g. `atoms.pbc` or `True`.
- **CRITICAL: DO NOT PASS `mic=True`**:
  `ase.geometry.get_distances` does **NOT** accept a keyword argument named `mic`. Supplying `cell` and `pbc` automatically activates the Minimum Image Convention.
  Passing `mic=True` raises:
  `TypeError: get_distances() got an unexpected keyword argument 'mic'`.
- **CRITICAL: RETURN VALUE IS A TUPLE `(displacement_vectors, distances)`**:
  `get_distances()` returns a 2-element tuple:
  1. `D` (displacement vectors): shape `(N, M, 3)` containing vectors $\vec{r}_j - \vec{r}_i$ with PBC applied.
  2. `d` (distances): shape `(N, M)` containing scalar Euclidean distances.
  To extract distances, unpack the tuple or take index `[1]`.

### Canonical Patterns with `get_distances`

```python
from ase.geometry import get_distances

# [V] Distance matrix between subset 1 (Ti) and subset 2 (Water O):
ti_pos = positions[ti_indices]          # Shape (N_ti, 3)
water_o_pos = positions[water_o_indices] # Shape (N_ow, 3)

# Pass cell and pbc directly (NO mic=True parameter!):
disp_vectors, dist_matrix = get_distances(
    ti_pos,
    water_o_pos,
    cell=atoms.cell,
    pbc=atoms.pbc,
)
# dist_matrix has shape (N_ti, N_ow)

# Find Ti atoms within 2.50 A of ANY water oxygen:
is_near_water = np.any(dist_matrix < 2.50, axis=1) # Boolean mask for ti_indices
boundary_ti = ti_indices[is_near_water]
```

### Contrast with `atoms.get_all_distances()` and `atoms.get_distances()`
- `atoms.get_all_distances(mic=True)`: Called on an `Atoms` instance. Takes `mic=True` and returns a single `(N, N)` distance matrix directly:
  ```python
  dists = atoms.get_all_distances(mic=True)
  np.fill_diagonal(dists, np.inf)
  has_overlap = bool(np.any(dists < 1e-5))
  ```
- `atoms.get_distances(a, indices, mic=True)`: Called on an `Atoms` instance. Takes index `a`, an array of neighbor `indices`, and `mic=True`, returning a 1D array of scalar distances.

---

## 3. High-Performance Neighbor Finding (`from ase.neighborlist import neighbor_list`)

### Function Signature & Return Values

```python
from ase.neighborlist import neighbor_list

# Function signature:
# neighbor_list(quantities, a, cutoff, self_interaction=False, max_nbins=1000000.0)
```

- **Arguments**:
  - `quantities`: String of desired return values. Common choice: `'ijd'` (or `'ij'`).
    - `'i'`: 1D int array of source atom indices.
    - `'j'`: 1D int array of target neighbor atom indices.
    - `'d'`: 1D float array of scalar distances $|\vec{r}_j - \vec{r}_i|$.
    - `'D'`: 2D float array of Cartesian displacement vectors $\vec{r}_j - \vec{r}_i$ with PBC.
    - `'S'`: 2D int array of periodic unit cell shift vectors.
  - `a`: The `ase.Atoms` object.
  - `cutoff`:
    - **Uniform scalar float**: e.g. `cutoff=2.8` (checks all atom pairs within 2.8 Å).
    - **Pairwise cutoff dictionary**: e.g. `{('O', 'H'): 1.20, ('H', 'O'): 1.20}`.
      > **CRITICAL**: When using a dictionary, always provide BOTH permutations `('A', 'B')` and `('B', 'A')`. If you only provide `('O', 'H')`, queries checking `i` as Hydrogen and `j` as Oxygen will be ignored.
    - **Per-atom radius list**: `natural_cutoffs(atoms, mult=1.0)`.

### Canonical Recipes

#### 1. Distinguishing Water Oxygen ($O_w$) from Lattice Oxygen ($O_{\text{lat}}$)
In interfacial metal-oxide/water simulations, water molecules contain covalent O–H bonds ($d \approx 0.98\text{--}1.05\text{ \AA}$):

```python
from ase.neighborlist import neighbor_list
import numpy as np

symbols = np.array(atoms.get_chemical_symbols())
o_all = np.where(symbols == "O")[0]

# Intramolecular O-H cutoff: 1.20 A (captures covalent bonds, excludes H-bonds > 1.7 A)
i_oh, _, _ = neighbor_list("ijd", atoms, {("O", "H"): 1.20, ("H", "O"): 1.20})
water_o_idx = np.unique(i_oh[symbols[i_oh] == "O"])
lattice_o_idx = np.setdiff1d(o_all, water_o_idx)

print(f"Total O: {len(o_all)} | Water O (Ow): {len(water_o_idx)} | Lattice O: {len(lattice_o_idx)}")
```

#### 2. Identifying Interfacial Surface Sites Coordinating with Water
Find surface metal atoms (e.g. Ti) coordinating directly with water oxygen within 2.50 Å:

```python
# Bidirectional pair cutoffs:
i_pairs, j_pairs, _ = neighbor_list("ijd", atoms, {("Ti", "O"): 2.50, ("O", "Ti"): 2.50})

# Select pairs where source is Ti and neighbor is a water oxygen:
is_ti = symbols[i_pairs] == "Ti"
is_ow = np.isin(j_pairs, water_o_idx)
coordinating_ti_idx = np.unique(i_pairs[is_ti & is_ow])
```

#### 3. Computing Coordination Numbers
```python
# Coordination number of each atom within 3.0 A:
i_idx, _, _ = neighbor_list("ij", atoms, cutoff=3.0)
coord_numbers = np.bincount(i_idx, minlength=len(atoms))
```

---

## 4. Physical Units & Conversions (`from ase.units import Hartree`)

Constants in `ase.units` are defined in **electronvolts (eV)**, **Ångströms (Å)**, and atomic mass units.

| Constant | Physical Meaning | Exact ASE Value / Dimension |
| :--- | :--- | :--- |
| `Hartree` | 1 Hartree atomic unit of energy | `27.211386... eV` |
| `Bohr` | 1 Bohr radius | `0.529177... Angstrom` |
| `fs` | 1 femtosecond time unit | In ASE internal time units (`~0.098227 ps`) |
| `ps` | 1 picosecond time unit | `1000.0 * fs` |
| `kB` | Boltzmann constant | `8.617333...e-5 eV / K` |
| `mol` | Avogadro's number scaling | `6.02214076e23` |
| `kJ` | 1 kilojoule | `1000.0 / 1.602176634e-19 eV` |

### Canonical Conversion Formulas

```python
from ase.units import Hartree, Bohr, kB

# 1. Convert potential energy from eV to Hartree:
energy_eV = atoms.get_potential_energy()
energy_Hartree = energy_eV / Hartree

# 2. Convert forces from eV / Angstrom to Hartree / Bohr:
forces_eV_A = atoms.get_forces()
forces_Ha_Bohr = forces_eV_A / (Hartree / Bohr)

# 3. Thermal energy k_B * T in eV at T = 300 K:
T = 300.0
kBT_eV = kB * T  # ~0.02585 eV
```

> **FATAL IMPORT ERROR TO AVOID**:
> Never write `from ase.units import Avogadro`! `Avogadro` does **not** exist in `ase.units`.
> Instead, define the constant directly: `NA = 6.02214076e23` or use `from scipy.constants import Avogadro`.

---

## 5. Summary Cheat-Sheet: Function API Matrix

| Goal | Correct API Call | What Breaks / Fails |
| :--- | :--- | :--- |
| **Load single frame** | `read(path, index=0)` | `read(path, index=":")` returns `list[Atoms]`, causing `AttributeError`. |
| **Write structure** | `write("file.extxyz", Atoms(...))` | Calling `atoms(...)` on the instance crashes with `TypeError`. |
| **Point distances** | `vecs, dists = get_distances(p1, p2, cell=cell, pbc=pbc)` | `get_distances(..., mic=True)` raises `TypeError: unexpected keyword 'mic'`. |
| **All pair distances** | `atoms.get_all_distances(mic=True)` | Omitting `mic=True` ignores periodic box wrapping. |
| **Neighbor list** | `i, j, d = neighbor_list('ijd', atoms, {('A','B'): r, ('B','A'): r})` | Asymmetric dict `{('A','B'): r}` misses reverse direction pairs. |
| **Energy to Hartree** | `atoms.get_potential_energy() / Hartree` | `from ase.constants import Hartree` raises `ModuleNotFoundError`. |
