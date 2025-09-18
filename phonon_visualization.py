# -*- coding: utf-8 -*-
"""
Interactive Phonon Dispersion + Eigenvector Visualizer
Author: Han-Hsuan Wu
"""

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
    
    import re

    start_time = time.time()

    with open(filename, "r") as f:
        text = f.read()

    # Find all q-point headers and their positions
    qpos_iter = list(re.finditer(r'(?m)^\s*-\s*q-position:\s*\[\s*([^\]]+)\]', text))
    if not qpos_iter:
        raise ValueError('Could not find any "q-position" entries in band.yaml')

    nqpoints = len(qpos_iter)

    # Extract q-point coordinates
    qpoints = np.empty((nqpoints, 3), dtype=np.float64)
    for qi, m in enumerate(qpos_iter):
        vals = [float(x) for x in m.group(1).split(",")]
        if len(vals) != 3:
            raise ValueError("q-position does not have 3 components")
        qpoints[qi] = vals

    # Build spans for each q-point section
    spans = []
    for i, m in enumerate(qpos_iter):
        start = m.end()
        end = qpos_iter[i + 1].start() if (i + 1) < nqpoints else len(text)
        spans.append((start, end))

    # Determine nbands and natoms from the first q-point section
    first_section = text[spans[0][0] : spans[0][1]]

    # Count bands in first section by counting frequency entries
    band_matches_first = list(re.finditer(r'(?m)^\s*frequency:\s*([-\d.+Ee]+)', first_section))
    nbands = len(band_matches_first)
    if nbands == 0:
        raise ValueError('Could not determine "nbands" from frequency entries')

    # Extract first eigenvector block and count atoms
    first_ev_block_match = re.search(
        r'(?ms)eigenvector:\s*(.*?)(?=^\s*frequency:|\Z)', first_section
    )
    if not first_ev_block_match:
        raise ValueError('Could not find "eigenvector" block in first q-point section')

    first_ev_block = first_ev_block_match.group(1)
    natoms = len(re.findall(r'(?m)^\s*-\s*#\s*atom', first_ev_block))
    if natoms == 0:
        # Fallback: count complex pairs, divide by 3 components per atom
        pairs_first = re.findall(r'\[\s*([-\d.+Ee]+)\s*,\s*([-\d.+Ee]+)\s*\]', first_ev_block)
        if len(pairs_first) % 3 != 0:
            raise ValueError("Could not infer natoms from eigenvector pairs")
        natoms = len(pairs_first) // 3

    # Pre-allocate arrays
    print(f"Data dimensions: {nqpoints} q-points, {nbands} bands, {natoms} atoms")
    qpoints_array = qpoints
    frequencies = np.empty((nqpoints, nbands), dtype=np.float64)
    eigenvectors = np.empty((nqpoints, nbands, natoms, 3), dtype=np.complex128)

    # Helper regex for band blocks within a q-point section:
    # capture frequency and the following eigenvector block
    band_block_re = re.compile(
        r'(?ms)^\s*frequency:\s*([-\d.+Ee]+)\s*.*?\beigenvector:\s*(.*?)(?=^\s*frequency:|\Z)'
    )

    # Parse all sections
    for qi, (s, e) in enumerate(spans):
        section = text[s:e]
        band_index = 0
        for bm in band_block_re.finditer(section):
            freq_str = bm.group(1)
            frequencies[qi, band_index] = float(freq_str) * 4.14 #THz to meV
            ev_block = bm.group(2)

            # Extract all [real, imag] pairs in order
            pairs = re.findall(r'\[\s*([-\d.+Ee]+)\s*,\s*([-\d.+Ee]+)\s*\]', ev_block)
            if len(pairs) < natoms * 3:
                raise ValueError(
                    f"Not enough eigenvector components at q={qi}, band={band_index}"
                )

            # Fill eigenvectors in the order: atoms x (x,y,z)
            for idx in range(natoms * 3):
                a = idx // 3
                c = idx % 3
                re_part = float(pairs[idx][0])
                im_part = float(pairs[idx][1])
                eigenvectors[qi, band_index, a, c] = re_part + 1j * im_part

            band_index += 1

        if band_index != nbands:
            raise ValueError(f"Expected {nbands} bands, found {band_index} at q-index {qi}")

    yaml_load_time = time.time()
    print(f"Text parsing took: {yaml_load_time - start_time:.2f} seconds")

    return qpoints_array, frequencies, eigenvectors


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
