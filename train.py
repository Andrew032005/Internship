import os
import argparse
import pandas as pd
import numpy as np
import random
import json
import re
import joblib

# 1. CPU thread and instruction set limitation to prevent "Illegal instruction"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
os.environ["DNNL_MAX_CPU_ISA"] = "SSE42"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import RDKit before Torch to avoid potential clashes in MKL/OpenBLAS initialization
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Setup PyTorch threads and reproducibility
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Model Definition ---
class ModernHybridGNN(nn.Module):
    def __init__(self, node_in_feats=3, gnn_hidden_dim=32, gnn_out_feats=16,
                 math_feats_dim=12, rdkit_feats_dim=5, use_rdkit=False,
                 mlp_hidden_dims=[128, 64, 32], dropout=0.15):
        super(ModernHybridGNN, self).__init__()
        self.use_rdkit = use_rdkit

        # GNN Branch
        self.conv1 = GCNConv(node_in_feats, gnn_hidden_dim)
        self.bn1 = nn.BatchNorm1d(gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_out_feats)
        self.bn2 = nn.BatchNorm1d(gnn_out_feats)

        # MLP Input Dimension
        input_dim = gnn_out_feats + math_feats_dim + 1
        if use_rdkit:
            input_dim += rdkit_feats_dim

        # Modern MLP
        layers = []
        curr_dim = input_dim
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, data, math_feats, n_elec, rdkit_feats=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.silu(self.bn1(self.conv1(x, edge_index)))
        x = F.silu(self.bn2(self.conv2(x, edge_index)))
        gnn_emb = global_mean_pool(x, batch)

        if self.use_rdkit and rdkit_feats is not None:
            combined = torch.cat([gnn_emb, math_feats, n_elec, rdkit_feats], dim=1)
        else:
            combined = torch.cat([gnn_emb, math_feats, n_elec], dim=1)
        return self.mlp(combined)

# --- Helper Functions ---
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_molecule_name(name):
    element_to_z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17}
    if name in element_to_z: return [element_to_z[name]], 0
    if name.endswith('2') and name[:-1] in element_to_z:
        z = element_to_z[name[:-1]]
        return [z, z], 1
    for i in range(1, len(name)):
        p1, p2 = name[:i], name[i:]
        if p1 in element_to_z and p2 in element_to_z: return [element_to_z[p1], element_to_z[p2]], 1
    return [1], 0

def build_graph(name, bond_length):
    z, is_diatomic = parse_molecule_name(name)
    n = len(z)
    feats = np.zeros((n, 3))
    for i in range(n):
        feats[i, 0] = z[i]
        if is_diatomic: feats[i, 1] = z[1-i]
        feats[i, 2] = bond_length
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) if is_diatomic else torch.empty((2, 0), dtype=torch.long)
    return Data(x=torch.tensor(feats, dtype=torch.float), edge_index=edge_index)

def calculate_maths_features(A, B, C, D, epsilon=1e-8):
    F1, F2, F3 = B-A, C-B, D-C
    scale = abs(A) + epsilon
    res = [F1/scale, F2/scale, F3/scale, F2/(abs(F1)+epsilon), F3/(abs(F2)+epsilon),
           np.sign(F1)*np.log(1+abs(F1)), np.sign(F2)*np.log(1+abs(F2)), np.sign(F3)*np.log(1+abs(F3)),
           np.tanh(F2/(F1+epsilon)), np.tanh(F3/(F2+epsilon) if abs(F2)>epsilon else 0),
           np.tanh(F3/(F1+epsilon)), (F3-F2)/(abs(F1)+epsilon)]
    return np.array(res)

def sanitize_rdkit_descriptors(raw_features):
    # If error occurred during extraction or value is 0/NaN, replace with -1.0
    # Unified logic: treat 0.0 as potentially undefined for some features
    return [float(v) if (v != 0.0 and not np.isnan(v)) else -1.0 for v in raw_features]

def get_rdkit_descriptors_dimer(name, bond_length):
    atom_symbols = re.findall('[A-Z][a-z]?', name)
    if not atom_symbols: return [0.0]*5
    if len(atom_symbols) == 1:
        if bond_length > 0: atom_symbols = [atom_symbols[0], atom_symbols[0]]
        else: xyz = f"1\n\n{atom_symbols[0]} 0.0 0.0 0.0\n"
    if len(atom_symbols) > 1:
        xyz = f"2\n\n{atom_symbols[0]} 0.0 0.0 0.0\n{atom_symbols[1]} 0.0 0.0 {bond_length:.10f}\n"

    mol = Chem.MolFromXYZBlock(xyz)
    if not mol: return [0.0]*5
    try:
        if mol.GetNumAtoms() > 1: rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)
        wt = Descriptors.MolWt(mol)
        mr = Descriptors.MolMR(mol)
        tpsa = Descriptors.TPSA(mol)
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
        charges = [c if not np.isnan(c) else 0.0 for c in charges]
        res = [float(wt), float(mr), float(tpsa), float(max(charges)), float(min(charges))]
        return sanitize_rdkit_descriptors(res)
    except: return [0.0]*5

def main():
    parser = argparse.ArgumentParser(description='Train modern hybrid GNN+MLP model for CCSDTQ extrapolation.')
    parser.add_argument('--train_data', type=str, default='training-set_2026summer_dif.csv')
    parser.add_argument('--use_rdkit', action='store_true', help='Enable RDKit descriptors')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    set_random_seeds(42)
    subdir = "with_rdkit" if args.use_rdkit else "without_rdkit"
    out_path = os.path.join(args.output_dir, subdir)
    os.makedirs(out_path, exist_ok=True)

    print(f"Loading data from {args.train_data} (RDKit: {args.use_rdkit})...")
    df = pd.read_csv(args.train_data)
    processed = []
    # Open training.log for writing
    log_f = open('training.log', 'w')
    log_f.write("Molecule,Total_Electrons,Target_Normalized\n")

    for _, row in df.iterrows():
        name, r1, elec = str(row['name']), float(row['R1']), float(row['total electrons'])
        A, B, C, D, L = row['RHF Energy (Hartree)'], row['MP2 Energy'], row['CCSD Energy'], row['CCSD(T) Energy'], row['CCSDTQ Energy']
        CCSDT = row['CCSDT Energy']

        g = build_graph(name, r1)
        mf = calculate_maths_features(A, B, C, D)

        # New target: (CCSDTQ - CCSDT) / N_elec
        target = (L - CCSDT) / elec
        rd = get_rdkit_descriptors_dimer(name, r1) if args.use_rdkit else []
        processed.append({'g': g, 'mf': mf, 'elec': [elec], 'y': [target], 'rd': rd})

        # Write to log
        log_f.write(f"{name},{elec},{target:.10f}\n")

    log_f.close()

    train_data_list, val_data_list = train_test_split(processed, test_size=0.2, random_state=42)
    m_scaler = StandardScaler().fit([d['mf'] for d in train_data_list])
    e_scaler = StandardScaler().fit([d['elec'] for d in train_data_list])
    r_scaler = StandardScaler().fit(np.nan_to_num([d['rd'] for d in train_data_list])) if args.use_rdkit else None

    def prep_pyg(data_list, m_s, e_s, r_s=None):
        out = []
        for d in data_list:
            g = d['g'].clone()
            g.mf = torch.tensor(m_s.transform([d['mf']]), dtype=torch.float)
            g.elec = torch.tensor(e_s.transform([d['elec']]), dtype=torch.float)
            if r_s: g.rd = torch.tensor(r_s.transform(np.nan_to_num([d['rd']])), dtype=torch.float)
            g.y = torch.tensor(d['y'], dtype=torch.float)
            out.append(g)
        return out

    t_loader = DataLoader(prep_pyg(train_data_list, m_scaler, e_scaler, r_scaler), batch_size=args.batch_size, shuffle=True)
    v_loader = DataLoader(prep_pyg(val_data_list, m_scaler, e_scaler, r_scaler), batch_size=args.batch_size)

    model = ModernHybridGNN(use_rdkit=args.use_rdkit)
    # Added weight_decay for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = 1e10
    history = []
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train(); train_loss = 0
        for b in t_loader:
            optimizer.zero_grad()
            p = model(b, b.mf, b.elec, b.rd if args.use_rdkit else None)
            l = criterion(p, b.y.view(-1, 1))
            if not torch.isnan(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += l.item() * b.num_graphs

        model.eval(); val_loss, val_mae = 0, 0
        with torch.no_grad():
            for b in v_loader:
                p = model(b, b.mf, b.elec, b.rd if args.use_rdkit else None)
                v_l = criterion(p, b.y.view(-1, 1))
                val_loss += v_l.item() * b.num_graphs
                val_mae += torch.abs(p - b.y.view(-1, 1)).sum().item()

        v_l_avg, v_m_avg = val_loss/len(val_data_list), val_mae/len(val_data_list)
        history.append({'train_loss': train_loss/len(train_data_list), 'val_loss': v_l_avg, 'val_mae': v_m_avg})

        if v_l_avg < best_val_loss:
            best_val_loss = v_l_avg
            torch.save(model.state_dict(), os.path.join(out_path, 'best_model.pth'))

        if (epoch+1)%50==0: print(f"Epoch {epoch+1}: Val MSE {v_l_avg:.9f}, Val MAE {v_m_avg:.9f}")

    # --- Save Outputs ---
    pd.DataFrame(history).to_csv(os.path.join(out_path, 'training_metrics.csv'), index=False)
    df_feat = df.copy()
    df_feat['Target_normalized'] = (df['CCSDTQ Energy'] - df['CCSDT Energy']) / df['total electrons']
    df_feat.to_csv(os.path.join(out_path, 'training-set_features_normalized.csv'), index=False)

    torch.save(model.state_dict(), os.path.join(out_path, 'Final_model_Maths_GNN_features_normalized.pth'))
    torch.save(model.state_dict(), os.path.join(out_path, 'gnn_model_weights.pth'))
    joblib.dump({'m': m_scaler, 'e': e_scaler, 'r': r_scaler}, os.path.join(out_path, 'scalers.pkl'))
    joblib.dump(m_scaler, os.path.join(out_path, 'maths_scaler.pkl'))
    joblib.dump(e_scaler, os.path.join(out_path, 'electron_scaler.pkl'))
    if args.use_rdkit: joblib.dump(r_scaler, os.path.join(out_path, 'rdkit_scaler.pkl'))
    joblib.dump(StandardScaler(), os.path.join(out_path, 'gnn_scaler.pkl'))

    with open(os.path.join(out_path, 'random_seed_info.json'), 'w') as f:
        json.dump({'random_seed': 42, 'numpy_seed': 42, 'torch_seed': 42, 'tensorflow_seed': 42}, f)
    print(f"Training complete. Outputs saved to {out_path}")

if __name__ == "__main__":
    import sys
    # Create an output log file to capture errors and logs
    with open('train.out', 'w') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            main()
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
