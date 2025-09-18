# -*- coding: utf-8 -*-
"""
Interactive Phonon Dispersion + Eigenvector Visualizer
Author: Han-Hsuan Wu
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # for 3D lattice plotting
from ipywidgets import interact, IntSlider
from ase.io import read  # for POSCAR parsing (uses ASE)


# ===============================
# 1. File Parsers
# ===============================

def parse_band_yaml(filename="band.yaml"):

    """
    Optimized parsing of band.yaml from phonopy.
    
    Performance improvements:
    - Pre-allocate numpy arrays with known dimensions
    - Minimize Python loops and list operations
    - Use vectorized numpy operations where possible
    - Efficient complex number construction
    """

    
    start_time = time.time()
    
    # Load YAML file
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    
    yaml_load_time = time.time()
    print(f"YAML loading took: {yaml_load_time - start_time:.2f} seconds")
    
    phonon_data = data['phonon']
    nqpoints = len(phonon_data)
    nbands = len(phonon_data[0]['band'])
    natoms = len(phonon_data[0]['band'][0]['eigenvector'])
    
    print(f"Data dimensions: {nqpoints} q-points, {nbands} bands, {natoms} atoms")
    
    # Pre-allocate arrays - this is crucial for performance
    qpoints = np.empty((nqpoints, 3), dtype=np.float64)
    frequencies = np.empty((nqpoints, nbands), dtype=np.float64)
    eigenvectors = np.empty((nqpoints, nbands, natoms, 3), dtype=np.complex128)
    
    # Optimized parsing loop
    for i, qpoint_data in enumerate(phonon_data):
        qpoints[i] = qpoint_data['q-position']
        
        bands = qpoint_data['band']
        
        # Extract frequencies using list comprehension (faster than loop)
        frequencies[i] = [band['frequency'] for band in bands]
        
        # Optimized eigenvector extraction
        for j, band in enumerate(bands):
            eigvec_data = band['eigenvector']
            
            # Convert to numpy arrays directly, avoiding nested loops
            real_imag_data = np.array(eigvec_data)  # Shape: (natoms, 3, 2)
            eigenvectors[i, j] = real_imag_data[:, :, 0] + 1j * real_imag_data[:, :, 1]
    
    end_time = time.time()
    print(f"Total parsing took: {end_time - start_time:.2f} seconds")
    print(f"Data processing took: {end_time - yaml_load_time:.2f} seconds")
    
    return qpoints, frequencies, eigenvectors


def parse_poscar(filename="POSCAR"):
    """Parse POSCAR using ASE."""
    atoms = read(filename)
    return atoms


# ===============================
# 2. Plotting
# ===============================

def plot_dispersion(qpoints, frequencies):
    """Plot phonon dispersion curve."""
    plt.figure(figsize=(6,4))
    for band in range(frequencies.shape[1]):
        plt.plot(range(len(qpoints)), frequencies[:, band], 'k-')
    plt.xlabel("Q-point index")
    plt.ylabel("Frequency (THz)")
    plt.title("Phonon Dispersion")
    plt.show()


def plot_eigenvectors(atoms, eigenvecs, scale=1.0):
    """Overlay phonon eigenvectors on the lattice."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # plot atoms
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        ax.scatter(*pos, s=100, label=sym if i==0 else "")

    # plot eigenvectors as arrows
    for i, pos in enumerate(positions):
        vec = np.real(eigenvecs[i]) * scale
        ax.quiver(pos[0], pos[1], pos[2], vec[0], vec[1], vec[2], color='r')

    ax.set_title("Phonon Mode Eigenvectors")
    plt.show()


# ===============================
# 3. Interactive Controls
# ===============================

def interactive_viewer(qpoints, frequencies, eigenvectors, atoms):
    """Interactive widget to explore phonon modes."""

    def view_mode(q_index=0, band_index=0):
        plt.figure(figsize=(6,4))
        for band in range(frequencies.shape[1]):
            plt.plot(range(len(qpoints)), frequencies[:, band], 'k-', alpha=0.3)
        plt.plot(q_index, frequencies[q_index, band_index], 'ro', markersize=10)
        plt.xlabel("Q-point index")
        plt.ylabel("Frequency (THz)")
        plt.title("Phonon Dispersion (selected mode highlighted)")
        plt.show()

        # plot eigenvectors
        eigvecs = eigenvectors[q_index, band_index]
        plot_eigenvectors(atoms, eigvecs, scale=2.0)

    interact(view_mode,
             q_index=IntSlider(min=0, max=len(qpoints)-1, step=1, value=0),
             band_index=IntSlider(min=0, max=frequencies.shape[1]-1, step=1, value=0))


# ===============================
# 4. Main Execution
# ===============================

if __name__ == "__main__":
    import os
    path = r"C:\Users\hanhsuan\Desktop\New folder"
    print("Load band.yaml and POSCAR")
    qpoints, freqs, eigvecs = parse_band_yaml(os.path.join(path, "band-twin-4.yaml")) #band-twin-4.yaml
    atoms = parse_poscar(os.path.join(path, "POSCAR"))
    # Step 1: quick dispersion plot
    plot_dispersion(qpoints, freqs)

    # Step 2: start interactive viewer (in Jupyter)
    # interactive_viewer(qpoints, freqs, eigvecs, atoms)
