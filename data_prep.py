import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds
from sklearn.preprocessing import StandardScaler
import re

# Atomic numbers mapping
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Sr': 38, 'Ba': 56
}

def parse_xyz(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0])
    atoms = []
    coords = []
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        if not parts: continue
        symbol = parts[0]
        symbol = re.sub(r'[^a-zA-Z]', '', symbol)
        atoms.append(ATOMIC_NUMBERS[symbol])
        coords.append([float(x) for x in parts[1:4]])

    atoms = np.array(atoms)
    coords = np.array(coords)
    # Center at origin
    coords = coords - np.mean(coords, axis=0)
    return atoms, coords

def calculate_maths_features(A, B, C, D):
    diffs = [B - A, C - B, D - C]

    def safe_div(n, d):
        if abs(d) < 1e-12: return 0.0
        res = n / d
        return np.clip(res, -1e3, 1e3)

    rel_steps = [safe_div(diffs[1], diffs[0]), safe_div(diffs[2], diffs[1])]
    log_comp = [np.log(abs(d) + 1e-12) for d in diffs]

    # Scale A, B, C to avoid huge tan values and also keep them in a reasonable range for sin/cos
    # Energies are e.g. -100. sin(-100) is fine. tan(-100) is fine.
    # But tan(x) is periodic.
    t_smooth = [np.sin(A), np.cos(B), np.clip(np.tan(C), -1e3, 1e3)]

    second_diff = diffs[1] - diffs[0]

    features = diffs + rel_steps + log_comp + t_smooth + [second_diff]

    # Final check for NaNs or Inf
    features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
    return [float(f) for f in features]

def get_rdkit_descriptors(atoms, coords, use_rdkit=True):
    if not use_rdkit:
        return []

    # Create an RDKit molecule from 3D coordinates
    symbols = {v: k for k, v in ATOMIC_NUMBERS.items()}
    xyz_str = f"{len(atoms)}\n\n"
    for a, c in zip(atoms, coords):
        # Use fixed-point notation to avoid scientific notation which RDKit XYZ parser might fail on
        xyz_str += f"{symbols[a]} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"

    mol = Chem.MolFromXYZBlock(xyz_str)
    if mol is None:
        return [0.0] * 5

    try:
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)
    except:
        pass

    # Ensure properties and rings are initialized
    mol.UpdatePropertyCache()
    Chem.FastFindRings(mol)

    # Selected descriptors: MolWt, MolMR, TPSA, MaxPartialCharge, MinPartialCharge
    # (Gasteiger charges require computation)
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        max_q = max([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())])
        min_q = min([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())])
    except:
        max_q, min_q = 0.0, 0.0

    try:
        wt = Descriptors.MolWt(mol)
    except: wt = 0.0
    try:
        mr = Descriptors.MolMR(mol)
    except: mr = 0.0
    try:
        tpsa = Descriptors.TPSA(mol)
    except: tpsa = 0.0

    desc_vals = [wt, mr, tpsa, max_q, min_q]

    # Masking for diatomic or undefined
    if len(atoms) <= 2:
        # Requirement: "Все неопределенные/ошибочные дескрипторы RDKit для двухатомных молекул заполняются нейтральным константным значением (-1 или средним по датасету)."
        # Treat 0.0 as potentially undefined for TPSA or charges in simple dimers
        desc_vals = [v if v != 0.0 else -1.0 for v in desc_vals]

    return desc_vals

def build_graph(atoms, coords):
    # Edges: r < 1.6 A
    adj = []
    edge_attr = []
    num_atoms = len(atoms)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 1.6:
                adj.append([i, j])
                adj.append([j, i])
                edge_attr.append([dist])
                edge_attr.append([dist])

    x = []
    for i in range(num_atoms):
        # Node features: [Z, r_i, theta_i] - but it says "Z_i, coordinates".
        # Let's use [Z_i, x_i, y_i, z_i] or just Z_i and position?
        # Requirement: "Vertices (V_i): Вектор признаков каждого атома: [Z_i, r_i, theta_i]"
        # For polyatomic, r_i and theta_i are less clear.
        # For diatomic, r_i could be distance to center.
        # Let's use [Z_i, coords_x, coords_y, coords_z] instead or just Z and distance to origin?
        # Re-reading: "Вектор признаков каждого атома v_i: [Z_i, r_i, theta_i]"
        # In polar coordinates for diatomic?
        # Let's assume spherical coordinates (r, theta, phi) but only r and theta are requested?
        # Or maybe it's meant for diatomic only?
        # I'll use [Z_i, x_i, y_i] if 2D or [Z_i, np.linalg.norm(coords[i]), 0]
        # I will use [Z_i, np.linalg.norm(coords[i]), np.arctan2(coords[i][1], coords[i][0])]
        r = np.linalg.norm(coords[i])
        theta = np.arccos(coords[i][2]/r) if r > 0 else 0
        x.append([atoms[i], r, theta])

    x = torch.tensor(x, dtype=torch.float)
    if adj:
        edge_index = torch.tensor(adj, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def prepare_data(train_csv, val_csv, rdkit_enabled=True):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    all_data = []

    # Process Train (Diatomics)
    for idx, row in train_df.iterrows():
        name = row['name']
        n_elec = row['total electrons']
        r1 = row['R1']
        # For atom names like H2, He2, Li2, FH, HNa
        # We need atomic numbers.
        if r1 == 0: # It's a single atom
            atom_symbols = re.findall('[A-Z][a-z]?', name)
            z_list = [ATOMIC_NUMBERS[s] for s in atom_symbols]
            coords = np.array([[0.0, 0.0, 0.0]])
            atoms = np.array(z_list)
        else:
            # Diatomic
            atom_symbols = re.findall('[A-Z][a-z]?', name)
            if len(atom_symbols) == 1: # e.g. H2, N2
                atom_symbols = [atom_symbols[0], atom_symbols[0]]
            z_list = [ATOMIC_NUMBERS[s] for s in atom_symbols]
            atoms = np.array(z_list)
            # Center diatomic at origin: -r1/2 and +r1/2
            coords = np.array([[0.0, 0.0, -r1/2.0], [0.0, 0.0, r1/2.0]])

        math_feats = calculate_maths_features(row['RHF Energy (Hartree)'],
                                               row['MP2 Energy'],
                                               row['CCSD Energy'],
                                               row['CCSD(T) Energy'])

        rdkit_feats = get_rdkit_descriptors(atoms, coords, use_rdkit=rdkit_enabled)
        # Padding/Masking for RDKit: if diatomic, some might be 0, we'll keep them as is or fill -1.
        # Requirement: "Все неопределенные/ошибочные дескрипторы RDKit для двухатомных молекул заполняются нейтральным константным значением (-1 или средним по датасету)."

        target = (row['CCSDTQ Energy'] - row['CCSDT Energy']) / n_elec

        graph = build_graph(atoms, coords)

        all_data.append({
            'graph': graph,
            'math_feats': math_feats,
            'n_elec': [n_elec],
            'rdkit_feats': rdkit_feats,
            'target': target,
            'is_train': True,
            'name': name,
            'n_atoms': len(atoms),
            'true_ccsdtq': row['CCSDTQ Energy'],
            'ccsdt': row['CCSDT Energy']
        })

    # Process Val (Polyatomics)
    for idx, row in val_df.iterrows():
        name = row['name']
        xyz_file = name + ".xyz"
        parsed = parse_xyz(xyz_file)
        if parsed is None:
            # Try names like 1-CH2 -> CH2? No, there is 1-CH2.xyz
            continue

        atoms, coords = parsed
        n_elec = row['total electrons']

        math_feats = calculate_maths_features(row['RHF Energy (Hartree)'],
                                               row['MP2 Energy'],
                                               row['CCSD Energy'],
                                               row['CCSD(T) Energy'])

        rdkit_feats = get_rdkit_descriptors(atoms, coords, use_rdkit=rdkit_enabled)

        # Target for validation (some might be NaN if unknown, but here we have them)
        if pd.isna(row['CCSDTQ Energy']):
            target = 0.0 # Placeholder
        else:
            target = (row['CCSDTQ Energy'] - row['CCSDT Energy']) / n_elec

        graph = build_graph(atoms, coords)

        all_data.append({
            'graph': graph,
            'math_feats': math_feats,
            'n_elec': [n_elec],
            'rdkit_feats': rdkit_feats,
            'target': target,
            'is_train': False,
            'name': name,
            'n_atoms': len(atoms),
            'true_ccsdtq': row['CCSDTQ Energy'],
            'ccsdt': row['CCSDT Energy']
        })

    return all_data

if __name__ == "__main__":
    data = prepare_data('training-set_2026summer.csv', 'validation-set_2026summer.csv')
    print(f"Loaded {len(data)} molecules")
    print(f"Sample math features: {data[0]['math_feats']}")
    print(f"Sample RDKit features: {data[0]['rdkit_feats']}")
