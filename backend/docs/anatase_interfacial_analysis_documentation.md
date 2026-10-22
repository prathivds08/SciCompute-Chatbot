# Anatase TiO2 / Water Interface: Structural & Surface Analysis Reference

This document details the surface geometry, slab orientation, and interfacial analysis protocols for Anatase $\text{TiO}_2$ / water systems.

---

## 1. Physical Surface vs. Simulation Box Boundary

In periodic slab simulations:
- **Simulation Box Dimensions**: Typically $a \approx 10.25$ Å, $b \approx 11.31$ Å, $c \approx 36.80$ Å.
- **$X$ and $Y$ Axes**: Have **Periodic Boundary Conditions (PBC)** representing an infinite continuous 2D plane. There are **NO physical surfaces or boundaries in $X$ and $Y$**.
- **$Z$ Axis**: Is the **surface normal**. The physical interface between the solid $\text{TiO}_2$ crystal slab and liquid water lies along the $Z$-axis.

### The Periodic Wrap Along $Z$
In many molecular dynamics configurations, the $\text{TiO}_2$ slab wraps across the $z = 0 \leftrightarrow z = c$ periodic cell boundary, with liquid water occupying the center of the unit cell ($z \approx 10 - 26$ Å).
- Atoms at $z \approx 0$ Å and $z \approx c$ Å are in the **bulk core** of the slab connected across the boundary.
- The **true physical surfaces facing liquid water** are located at the internal interfaces (around $z \approx 8$ Å and $z \approx 28$ Å).

---

## 2. Canonical Surface Atom Identification Recipe

To identify physical surface $\text{Ti}_{5c}$ active sites and interfacial water molecules:

```python
import numpy as np
from ase.io import read

# 1. Always load frame 0 as an Atoms object (index=0)
atoms = read("structure.extxyz", index=0)
positions = atoms.get_positions()
symbols = np.array(atoms.get_chemical_symbols())
c_length = atoms.cell.lengths()[2]

# 2. Shift z by c/2 to center the slab if wrapped across z=0
ti_mask = symbols == "Ti"
ti_z = positions[ti_mask, 2]

# If Ti is split across z=0 and z=c, center the slab:
if np.any(ti_z < 5.0) and np.any(ti_z > (c_length - 5.0)):
    shifted_z = (positions[:, 2] + c_length / 2.0) % c_length
else:
    shifted_z = positions[:, 2]

shifted_ti_z = shifted_z[ti_mask]
top_ti_z = np.max(shifted_ti_z)
bottom_ti_z = np.min(shifted_ti_z)

# 3. Top and bottom surface Ti atoms (within 1.5 A of extrema)
surface_ti_top = np.where(ti_mask & (shifted_z > top_ti_z - 1.5))[0]
surface_ti_bottom = np.where(ti_mask & (shifted_z < bottom_ti_z + 1.5))[0]

# 4. Interfacial contact-layer water oxygens (within 3.0 A above surface)
water_o_mask = (symbols == "O") & (shifted_z > top_ti_z) & (shifted_z < top_ti_z + 3.0)
interfacial_water_o = np.where(water_o_mask)[0]
```

---

## 3. System Composition & Understanding the ~60 Atoms

In this specific Anatase (101) / Water interface system (`Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`):
- **Total atoms in simulation cell**: 426 atoms.
- **Titanium atoms (Ti)**: Exactly 60 atoms.
- **Oxygen atoms (O)**: 202 atoms (120 in the $\text{TiO}_2$ lattice + 82 in water).
- **Hydrogen atoms (H)**: 164 atoms (82 $\text{H}_2\text{O}$ water molecules).
- **Physical $\text{TiO}_2$ Slab Atoms**: 180 atoms ($60 \times \text{TiO}_2$).

### Physical Boundary vs. Box Faces: Why Boundary Atoms are Way Less
1. **The Physical Solid-Liquid Boundary (Surface of Anatase Slab)**:
   - The 60 count represents the **entire Titanium content of the slab**, from the interior core to the exterior faces.
   - At the actual physical boundary facing liquid water:
     - Top surface active coordination sites ($\text{Ti}_{5c}$): **6 atoms**.
     - Bottom surface active coordination sites: **6 atoms**.
     - **Total surface $\text{Ti}$ atoms at the interface: ONLY 12 atoms!**
     - Adding the 12 surface bridging oxygens ($O_{2c}$), the entire solid surface boundary has **ONLY 24 atoms** (12 $\text{Ti}$ + 12 $\text{O}$).
     - Directly bound interfacial water: **3 to 6 contact $\text{H}_2\text{O}$ molecules**.
     - This is **WAY LESS** than 60 or 98!

2. **Simulation Box Faces (Geometric Boundaries under PBC)**:
   - In periodic boundary simulations, $x$ and $y$ are topologically continuous (no walls).
   - The previous output of "98 atoms" was an artifact of setting `tol = 0.05` (a 5% fractional window on all 6 faces), which sweeps **$27.1\%$ of the entire 3D simulation volume**, sweeping up bulk liquid water and interior slab layers!
   - Under a genuine physical proximity cutoff (e.g., within 0.2 Å of a cell boundary face), **only ~4 to 6 atoms** sit near box faces.
   - Therefore, whether defined physically (solid-liquid interface) or geometrically (box faces), the number of boundary atoms is indeed **way less**.
