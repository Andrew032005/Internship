import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re

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

# Import RDKit before Torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Setup PyTorch
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Model Definition (Must match train.py) ---
class ModernHybridGNN(nn.Module):
    def __init__(self, node_in_feats=3, gnn_hidden_dim=32, gnn_out_feats=16,
                 math_feats_dim=12, rdkit_feats_dim=5, use_rdkit=False,
                 mlp_hidden_dims=[128, 64, 32], dropout=0.15):
        super(ModernHybridGNN, self).__init__()
        self.use_rdkit = use_rdkit
        self.conv1 = GCNConv(node_in_feats, gnn_hidden_dim)
        self.bn1 = nn.BatchNorm1d(gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_out_feats)
        self.bn2 = nn.BatchNorm1d(gnn_out_feats)
        input_dim = gnn_out_feats + math_feats_dim + 1
        if use_rdkit: input_dim += rdkit_feats_dim
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
def parse_xyz(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f: lines = f.readlines()
    num = int(lines[0])
    atoms, coords = [], []
    for line in lines[2:2+num]:
        parts = line.split()
        if not parts: continue
        atoms.append(re.sub(r'[^a-zA-Z]', '', parts[0]))
        coords.append([float(x) for x in parts[1:4]])
    element_to_z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ca': 20, 'Sr': 38, 'Ba': 56, 'Ti': 22, 'V': 23}
    z_list = [element_to_z.get(a, 1) for a in atoms]
    coords = np.array(coords)
    coords = coords - np.mean(coords, axis=0) # Centering
    return z_list, coords, atoms

def build_poly_graph(z, coords, els):
    radii = {'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Be': 1.05, 'Mg': 1.50, 'Ca': 1.80, 'Sr': 2.00, 'Ba': 2.15, 'Ti': 2.10, 'V': 2.30}
    adj, lens = [], []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < (radii.get(els[i], 0.7) + radii.get(els[j], 0.7)) * 1.2:
                adj.extend([[i, j], [j, i]]); lens.append(dist)
    avg_l = np.mean(lens) if lens else 0.0
    feats = np.zeros((len(z), 3))
    for i in range(len(z)):
        feats[i, 0] = z[i]
        neighs = [z[edge[1]] for edge in np.array(adj).reshape(-1, 2) if edge[0] == i] if adj else []
        if neighs: feats[i, 1] = neighs[0]
        feats[i, 2] = avg_l
    return Data(x=torch.tensor(feats, dtype=torch.float), edge_index=torch.tensor(adj, dtype=torch.long).t() if adj else torch.empty((2, 0), dtype=torch.long))

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
    return [float(v) if (v != 0.0 and not np.isnan(v)) else -1.0 for v in raw_features]

def get_rdkit_descriptors_poly(els, coords):
    xyz = f"{len(els)}\n\n"
    for el, c in zip(els, coords): xyz += f"{el} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
    mol = Chem.MolFromXYZBlock(xyz)
    if not mol: return [0.0]*5
    try:
        if mol.GetNumAtoms() > 1: rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)
        wt = Descriptors.MolWt(mol); mr = Descriptors.MolMR(mol); tpsa = Descriptors.TPSA(mol)
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
        charges = [c if not np.isnan(c) else 0.0 for c in charges]
        res = [float(wt), float(mr), float(tpsa), float(max(charges)), float(min(charges))]
        return sanitize_rdkit_descriptors(res)
    except: return [0.0]*5

def parse_molecule_name_dimer(name):
    element_to_z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17}
    if name in element_to_z: return [element_to_z[name]], 0
    if name.endswith('2') and name[:-1] in element_to_z:
        z = element_to_z[name[:-1]]
        return [z, z], 1
    for i in range(1, len(name)):
        p1, p2 = name[:i], name[i:]
        if p1 in element_to_z and p2 in element_to_z: return [element_to_z[p1], element_to_z[p2]], 1
    return [1], 0

def build_graph_dimer(name, bond_length):
    z, is_diatomic = parse_molecule_name_dimer(name)
    n = len(z)
    feats = np.zeros((n, 3))
    for i in range(n):
        feats[i, 0] = z[i]
        if is_diatomic: feats[i, 1] = z[1-i]
        feats[i, 2] = bond_length
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) if is_diatomic else torch.empty((2, 0), dtype=torch.long)
    return Data(x=torch.tensor(feats, dtype=torch.float), edge_index=edge_index)

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
        wt = Descriptors.MolWt(mol); mr = Descriptors.MolMR(mol); tpsa = Descriptors.TPSA(mol)
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
        charges = [c if not np.isnan(c) else 0.0 for c in charges]
        res = [float(wt), float(mr), float(tpsa), float(max(charges)), float(min(charges))]
        return sanitize_rdkit_descriptors(res)
    except: return [0.0]*5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data', type=str, default='validation-set_2026summer_dif.csv')
    parser.add_argument('--use_rdkit', action='store_true')
    parser.add_argument('--model_dir', type=str, default='outputs')
    args = parser.parse_args()

    subdir = "with_rdkit" if args.use_rdkit else "without_rdkit"
    path = os.path.join(args.model_dir, subdir)
    if not os.path.exists(os.path.join(path, 'best_model.pth')):
        print(f"Error: Model not found in {path}. Run train.py first.")
        return

    model = ModernHybridGNN(use_rdkit=args.use_rdkit)
    model.load_state_dict(torch.load(os.path.join(path, 'best_model.pth')))
    model.eval()
    scalers = joblib.load(os.path.join(path, 'scalers.pkl'))
    m_scaler, e_scaler, r_scaler = scalers['m'], scalers['e'], scalers.get('r')

    df = pd.read_csv(args.val_data)
    results = []

    log_f = open('validation.log', 'w')
    log_f.write("Molecule,Actual_L,Predicted_L,Error_kcal\n")

    for _, row in df.iterrows():
        name, elec = str(row['name']), float(row['total electrons'])
        p = parse_xyz(f"{name}.xyz")

        if p:
            z, coords, els = p
            g = build_poly_graph(z, coords, els)
            rd = get_rdkit_descriptors_poly(els, coords) if args.use_rdkit else []
        else:
            # Diatomic or single atom fallback (e.g. Ba, H2)
            r1 = float(row['R1']) if 'R1' in row and not pd.isna(row['R1']) else 0.0
            g = build_graph_dimer(name, r1)
            rd = get_rdkit_descriptors_dimer(name, r1) if args.use_rdkit else []

        # New base for prediction is CCSDT (column index 8 or row['CCSDT Energy'])
        CCSDT = row['CCSDT Energy']
        CCSDTQ = row['CCSDTQ Energy']

        # FIX: Use CCSD(T) as 4th argument D, as in training
        mf = calculate_maths_features(row['RHF Energy (Hartree)'], row['MP2 Energy'], row['CCSD Energy'], row['CCSD(T) Energy'])
        m_s = torch.tensor(m_scaler.transform([mf]), dtype=torch.float)
        e_s = torch.tensor(e_scaler.transform([[elec]]), dtype=torch.float)
        r_s = torch.tensor(r_scaler.transform([rd]), dtype=torch.float) if args.use_rdkit else None

        with torch.no_grad():
            pred_norm = model(Batch.from_data_list([g]), m_s, e_s, r_s).item()

        # Restore CCSDTQ: Predicted_L = CCSDT + pred_norm * elec
        # Model outputs normalized correction per electron
        pred_corr_absolute = pred_norm * elec
        pred_L = CCSDT + pred_corr_absolute

        # Calculate Error strictly as difference in physical units
        actual_corr_absolute = CCSDTQ - CCSDT
        error_kcal = (pred_corr_absolute - actual_corr_absolute) * 627.509

        # Add MAE and RMSE per sample (though RMSE for a single point is just |error|)
        results.append({
            'molecule': name, 'actual_L': CCSDTQ, 'predicted_L': pred_L,
            'actual_corr': actual_corr_absolute, 'pred_corr': pred_corr_absolute,
            'error_kcal': error_kcal,
            'MAE': abs(pred_L - CCSDTQ),
            'RMSE': np.sqrt((pred_L - CCSDTQ)**2)
        })

        log_f.write(f"{name},{CCSDTQ:.10f},{pred_L:.10f},{error_kcal:.10f}\n")

    log_f.close()
    res_df = pd.DataFrame(results)

    # Global metrics
    clean = res_df.dropna()

    # Bias-Variance decomposition (Corrected approach based on residuals in kcal/mol)
    errors_kcal = (clean['predicted_L'] - clean['actual_L']) * 627.509
    bias_kcal = errors_kcal.mean()
    bias_sq_kcal = bias_kcal ** 2
    variance_kcal = errors_kcal.var()

    # Save the file in ROOT for supervisor compatibility
    res_df.to_csv('validation_results_maths_gnn-DTQ_normalized.csv', index=False)
    # Also save in output path
    res_df.to_csv(os.path.join(path, 'validation_results_maths_gnn-DTQ_normalized.csv'), index=False)

    # Save logs to specific folder as well
    import shutil
    if os.path.exists('training.log'): shutil.copy('training.log', os.path.join(path, 'training.log'))
    if os.path.exists('validation.log'): shutil.copy('validation.log', os.path.join(path, 'validation.log'))

    # Calculate final metrics in kcal/mol for physical interpretation
    mae_kcal = mean_absolute_error(clean['actual_L'], clean['predicted_L']) * 627.509
    rmse_kcal = np.sqrt(mean_squared_error(clean['actual_L'], clean['predicted_L'])) * 627.509

    print(f"Final Validation Metrics:")
    print(f"  MAE: {mae_kcal:.4f} kcal/mol")
    print(f"  RMSE: {rmse_kcal:.4f} kcal/mol")

    # --- Visualizations ---
    plots = os.path.join(path, 'plots'); os.makedirs(plots, exist_ok=True)
    mp = os.path.join(path, 'training_metrics.csv')
    if os.path.exists(mp):
        h = pd.read_csv(mp)
        plt.figure(figsize=(8,6)); plt.plot(h['train_loss'], label='Train MSE'); plt.plot(h['val_loss'], label='Val MSE'); plt.yscale('log'); plt.legend(); plt.title('Learning Curves'); plt.savefig(os.path.join(plots, 'learning_curves.png')); plt.close()

    plt.figure(figsize=(8,6)); plt.scatter(clean['actual_L'], clean['predicted_L'], alpha=0.6); lims = [clean['actual_L'].min(), clean['actual_L'].max()]; plt.plot(lims, lims, 'r--'); plt.xlabel('Actual Energy'); plt.ylabel('Predicted'); plt.savefig(os.path.join(plots, 'scatter_energy.png')); plt.close()
    plt.figure(figsize=(8,6)); plt.scatter(clean['actual_corr'], clean['pred_corr'], alpha=0.6, color='orange'); lims = [clean['actual_corr'].min(), clean['actual_corr'].max()]; plt.plot(lims, lims, 'r--'); plt.xlabel('Actual Correction'); plt.ylabel('Predicted Correction'); plt.savefig(os.path.join(plots, 'scatter_correction.png')); plt.close()
    plt.figure(figsize=(8,6)); plt.hist(clean['error_kcal'], bins=30); plt.xlabel('Error (kcal/mol)'); plt.savefig(os.path.join(plots, 'error_histogram.png')); plt.close()

    clean['n_atoms'] = clean['molecule'].apply(lambda x: len(parse_xyz(f"{x}.xyz")[0]) if os.path.exists(f"{x}.xyz") else 2)
    s_err = clean.groupby('n_atoms')['error_kcal'].apply(lambda x: np.mean(np.abs(x)))
    plt.figure(figsize=(8,6)); plt.plot(s_err.index, s_err.values, marker='o'); plt.xlabel('Atoms'); plt.ylabel('MAE (kcal/mol)'); plt.title('Error vs Molecule Size'); plt.savefig(os.path.join(plots, 'error_vs_size.png')); plt.close()

    # 5. Bias-Variance Decomposition Histogram (Corrected)
    plt.figure(figsize=(8,6))
    plt.bar(['Bias^2', 'Variance'], [bias_sq_kcal, variance_kcal], color=['salmon', 'lightblue'])
    plt.ylabel('Value (kcal/mol)^2')
    plt.title('Corrected Bias-Variance Decomposition of Prediction Error')
    plt.savefig(os.path.join(plots, 'bias_variance_decomp.png'))
    plt.close()

    # 6. Predicted vs Residuals
    residuals = clean['predicted_L'] - clean['actual_L']
    plt.figure(figsize=(8,6))
    plt.scatter(clean['predicted_L'], residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Energy (Hartree)')
    plt.ylabel('Residuals (Hartree)')
    plt.title('Predicted vs Residuals')
    plt.savefig(os.path.join(plots, 'predicted_vs_residuals.png'))
    plt.close()

    # 7. Saliency (Corrected for GNN)
    model.eval(); sm = df.iloc[0]; name_s = sm['name']; p = parse_xyz(f"{name_s}.xyz")
    if p or True: # Use any molecule from sample
        if p:
            z, co, el = p; g = build_poly_graph(z, co, el)
            rd = get_rdkit_descriptors_poly(el, co) if args.use_rdkit else []
        else:
            r1 = float(sm['R1']) if 'R1' in sm and not pd.isna(sm['R1']) else 0.0
            g = build_graph_dimer(name_s, r1)
            rd = get_rdkit_descriptors_dimer(name_s, r1) if args.use_rdkit else []

        mf = calculate_maths_features(sm['RHF Energy (Hartree)'], sm['MP2 Energy'], sm['CCSD Energy'], sm['CCSD(T) Energy'])

        m_s = torch.tensor(m_scaler.transform([mf]), dtype=torch.float, requires_grad=True)
        e_s = torch.tensor(e_scaler.transform([[sm['total electrons']]]), dtype=torch.float, requires_grad=True)

        r_s = None
        if args.use_rdkit:
            r_s = torch.tensor(r_scaler.transform([rd]), dtype=torch.float, requires_grad=True)

        g_batch = Batch.from_data_list([g])
        g_batch.x.requires_grad = True

        out = model(g_batch, m_s, e_s, r_s)
        out.backward()

        gnn_importance = g_batch.x.grad.abs().mean().item() if g_batch.x.grad is not None else 0.0

        imps = [m_s.grad.abs().mean().item(), gnn_importance, e_s.grad.abs().mean().item()]
        if args.use_rdkit: imps.append(r_s.grad.abs().mean().item())

        labels = ['Maths', 'GNN', 'N_elec']
        if args.use_rdkit: labels.append('RDKit')
        plt.figure(figsize=(8,6))
        plt.bar(labels, imps, color=['blue', 'green', 'red', 'purple'])
        plt.title('Corrected Feature Domain Importance (Saliency)')
        plt.savefig(os.path.join(plots, 'feature_importance.png'))
        plt.close()

    print(f"Validation complete. Plots in {plots}")

if __name__ == "__main__":
    import sys
    # Create an output log file to capture errors and logs
    with open('validate.out', 'w') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            main()
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
