import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list
from ase.units import Hartree

def main() -> None:
    """Calculate and report the potential energy of anatase TiO2/water interface configuration."""
    
    # Path handling with container fallback
    CONTAINER_PATH = Path("/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
    LOCAL_PATH = Path(__file__).resolve().parent / "data" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    
    DATA_PATH = CONTAINER_PATH if CONTAINER_PATH.exists() else LOCAL_PATH
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Structure file not found: {DATA_PATH}")
    
    # Load single structure (index=0 for static configuration)
    atoms = read(str(DATA_PATH), index=0)
    
    n_atoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    
    # Get potential energy in Hartree and convert to eV
    energy_Ha = atoms.get_potential_energy() / Hartree
    
    # Basic structural metadata for verification
    cell_lengths = atoms.cell.lengths()
    c_len = float(cell_lengths[2])  # Z-dimension (slab normal)
    
    # Identify element counts for stoichiometry check
    ti_mask = symbols == "Ti"
    o_mask = symbols == "O"
    h_mask = symbols == "H"
    
    n_ti = int(np.sum(ti_mask))
    n_o = int(np.sum(o_mask))
    n_h = int(np.sum(h_mask))
    
    # Check for any overlapping atoms (should be false)
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    has_overlap = bool(np.any(dists < 1e-5))
    
    # Print summary to stdout
    print("=" * 60)
    print("ANATASE TiO2/WATER INTERFACE ENERGY ANALYSIS")
    print("=" * 60)
    print(f"Total atoms: {n_atoms}")
    print(f"  - Ti atoms: {n_ti}")
    print(f"  - O atoms: {n_o}")
    print(f"  - H atoms: {n_h}")
    print("-" * 40)
    print(f"Potential Energy (Hartree): {energy_Ha:.6f} Ha")
    print(f"Potential Energy (eV):      {energy_Ha * 27.211386245948: .2f} eV")
    print("-" * 40)
    print(f"Cell dimensions (Angstrom): ({cell_lengths[0]:.2f}, {cell_lengths[1]:.2f}, {c_len:.2f})")
    print(f"PBC overlap check (<1e-5 A): {has_overlap}")
    print("=" * 60)
    
    # Generate a simple density profile plot for visualization
    dz = 0.15
    bins = np.arange(0.0, c_len + dz, dz)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist_o, _ = np.histogram(atoms.get_positions()[o_mask, 2], bins=bins)
    
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(bin_centers, hist_o, lw=2.0, label="Oxygen Density Profile", color="#377eb8")
    plt.xlabel("z (Angstrom)")
    plt.ylabel("Number of O atoms per bin")
    plt.title(f"Anatase TiO2/Water Interface - Oxygen Distribution\nTotal Energy: {energy_Ha:.4f} Ha")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to output file
    plt.savefig("figure_output.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: figure_output.png")
    plt.close("all")

if __name__ == "__main__":
    main()