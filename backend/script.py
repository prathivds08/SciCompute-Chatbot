from pathlib import Path
from ase.io import read
import numpy as np


def main():
    # Define the path to the .extxyz file
    CONTAINER_PATH = Path(r"data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
    LOCAL_PATH = Path(__file__).resolve().parent / "data" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    DATA_PATH = CONTAINER_PATH if CONTAINER_PATH.exists() else LOCAL_PATH




# Check if the file exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"The file {DATA_PATH} does not exist. Please provide the correct path to the .extxyz file.")




# Read the .extxyz file
    atoms = read(str(DATA_PATH), index=0)  # Read the first frame of the trajectory




# Get the lattice vectors
    cell = atoms.get_cell()
    print(f"Lattice vectors: {cell}")




# Get the positions of the atoms
    positions = atoms.get_positions()
    print(f"Atom positions: {positions}")




# Get the boundary atoms
    boundary_atoms = []
    tolerance = 0.05  # Fractional tolerance




    scaled_positions = atoms.get_scaled_positions(wrap=False)
    for i, pos in enumerate(scaled_positions):
        if np.isclose(pos[0], 0.0, atol=tolerance) or np.isclose(pos[0], 1.0, atol=tolerance):
            boundary_atoms.append(i)
        if np.isclose(pos[1], 0.0, atol=tolerance) or np.isclose(pos[1], 1.0, atol=tolerance):
            boundary_atoms.append(i)
        if np.isclose(pos[2], 0.0, atol=tolerance) or np.isclose(pos[2], 1.0, atol=tolerance):
            boundary_atoms.append(i)




# Output the boundary atoms and their positions
    print("Boundary atoms and their positions:")
    for atom_index in boundary_atoms:
        print(f"Atom {atom_index}: {positions[atom_index]}")


if __name__ == "__main__":
    main()