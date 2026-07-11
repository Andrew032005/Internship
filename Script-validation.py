import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
import json

class ExtendedMolecularGraphGenerator:
    """Extended Molecular Graph Generator for multi-atomic molecules"""

    def __init__(self):
        self.element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ca': 10, 'Sr': 10, 'Ba': 10, 'Ti': 22, 'V': 23
        }

        # Covalent radii for bond detection (in Angstroms)
        self.covalent_radii = {
            'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.57,
            'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Be': 1.05, 'Mg': 1.50, 'Ca': 1.80, 'Sr': 2.00, 'Ba': 2.15, 'Ti': 2.10, 'V': 2.30
        }

    def parse_xyz_file(self, xyz_file_path):
        """Parse XYZ file to extract elements and coordinates"""
        with open(xyz_file_path, 'r') as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        elements = []
        coordinates = []

        for i in range(2, 2 + num_atoms):
            parts = lines[i].strip().split()
            element = parts[0]
            coords = [float(x) for x in parts[1:4]]
            elements.append(element)
            coordinates.append(coords)

        return elements, np.array(coordinates), lines

    def calculate_bond_length(self, coord1, coord2):
        """Calculate distance between two atoms"""
        return np.linalg.norm(coord1 - coord2)

    def detect_bonds(self, elements, coordinates, threshold=1.2):
        """Detect bonds based on covalent radii and distances"""
        num_atoms = len(elements)
        bonds = []
        bond_lengths = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                r1 = self.covalent_radii.get(elements[i], 0.7)
                r2 = self.covalent_radii.get(elements[j], 0.7)
                distance = self.calculate_bond_length(coordinates[i], coordinates[j])

                if distance < (r1 + r2) * threshold:
                    bonds.append((i, j))
                    bond_lengths.append(distance)

        return bonds, bond_lengths

    def build_graph(self, elements, coordinates):
        """Build graph representation for multi-atomic molecules - compatible with training format"""
        bonds, bond_lengths = self.detect_bonds(elements, coordinates)
        atomic_numbers = [self.element_to_z[element] for element in elements]
        num_atoms = len(atomic_numbers)

        # Create node features - using same 3-feature format as training
        node_features = np.zeros((num_atoms, 3))
        avg_bond_length = np.mean(bond_lengths) if bond_lengths else 0.0

        for i in range(num_atoms):
            node_features[i, 0] = atomic_numbers[i]  # Current atom's atomic number

            # For atoms with neighbors, include neighbor info
            neighbor_indices = [j for bond in bonds if i in bond for j in bond if j != i]
            if neighbor_indices:
                # Use first neighbor's atomic number (similar to training approach)
                node_features[i, 1] = atomic_numbers[neighbor_indices[0]]

            node_features[i, 2] = avg_bond_length  # Average bond length from XYZ structure

        # Create edge index
        edge_list = []
        for bond in bonds:
            edge_list.append([bond[0], bond[1]])
            edge_list.append([bond[1], bond[0]])  # Undirected graph

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Return as PyG Data object - same format as training
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            num_atoms=num_atoms,
            num_nodes=num_atoms
        )

class GNNModel(torch.nn.Module):
    """GNN Model with global pooling - same architecture as training"""
    def __init__(self, num_node_features, hidden_dim=16, output_dim=1):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Global mean pooling to get one output per graph (molecule)
        x = global_mean_pool(x, batch)
        return x

def global_mean_pool(x, batch):
    """
    Global mean pooling implementation - same as training
    x: node features [num_nodes, feature_dim]
    batch: batch indices [num_nodes]
    returns: graph-level features [num_graphs, feature_dim]
    """
    if batch is None:
        return torch.mean(x, dim=0, keepdim=True)

    unique_batches = torch.unique(batch)
    pooled = []
    for batch_idx in unique_batches:
        mask = (batch == batch_idx)
        pooled.append(torch.mean(x[mask], dim=0))
    return torch.stack(pooled)

def calculate_maths_features(A, B, C, D, epsilon=1e-8):
    """Calculate normalized and robust mathematical features (same as training)"""
    # Original differences
    F1 = B - A
    F2 = C - B
    F3 = D - C

    # Normalized by initial value magnitude
    scale = abs(A) + epsilon
    F1_norm = F1 / scale
    F2_norm = F2 / scale
    F3_norm = F3 / scale

    # Relative features (scale-invariant)
    F4 = (C - B) / (abs(F1) + epsilon)  # Next step relative to first
    F5 = F3 / (abs(F2) + epsilon)        # Last step relative to total span

    # Log-transformed features (compress extreme values)
    F6 = np.sign(F1) * np.log(1 + abs(F1))
    F7 = np.sign(F2) * np.log(1 + abs(F2))
    F8 = np.sign(F3) * np.log(1 + abs(F3))

    # Convergence rate features
    r1 = (C - B) / (F1 + epsilon)
    r2 = F3 / (C - B + epsilon) if abs(C - B) > epsilon else 0

    # Bounded ratios using tanh
    F9 = np.tanh(r1)
    F10 = np.tanh(r2)

    # Richardson-based feature
    F11 = np.tanh(F3 / (F1 + epsilon))

    # Second-order differences
    F12 = ((D - C) - (C - B)) / (abs(B - A) + epsilon)

    return np.array([F1_norm, F2_norm, F3_norm, F4, F5, F6, F7, F8, F9, F10, F11, F12])

def validate_on_multi_atomic_molecules():
    """
    Complete validation script for multi-atomic molecules
    Uses the same GNN architecture and feature processing as training
    """
    print("=" * 80)
    print("VALIDATION ON MULTI-ATOMIC MOLECULES (WITH NORMALIZED FEATURES)")
    print("=" * 80)

    # Load trained models and scalers
    print("Loading trained models and feature scalers...")
    try:
        # Try to load final model with normalized features
        try:
            keras_model = load_model('Final_model_Maths_GNN_features_normalized.keras')
            print("✓ Loaded Final_model_Maths_GNN_features_normalized.keras")
        except:
            # Fallback to non-normalized model if normalized version doesn't exist
            try:
                keras_model = load_model('Final_model_Maths_GNN_features.keras')
                print("✓ Loaded Final_model_Maths_GNN_features.keras (non-normalized)")
            except:
                keras_model = load_model('best_model_Maths_GNN_features.keras')
                print("✓ Loaded best_model_Maths_GNN_features.keras (non-normalized)")

        # Load all feature scalers
        try:
            maths_scaler = joblib.load('maths_scaler.pkl')
            print("✓ Loaded maths_scaler.pkl")
        except:
            print("⚠  maths_scaler.pkl not found, will use non-normalized Maths features")
            maths_scaler = None

        try:
            gnn_scaler = joblib.load('gnn_scaler.pkl')
            print("✓ Loaded gnn_scaler.pkl")
        except:
            print("⚠  gnn_scaler.pkl not found, will use non-normalized GNN features")
            gnn_scaler = None

        try:
            electron_scaler = joblib.load('electron_scaler.pkl')
            print("✓ Loaded electron_scaler.pkl")
        except:
            print("✗ electron_scaler.pkl not found")
            return

        # Load GNN model with same architecture
        gnn_checkpoint = torch.load('gnn_model_weights.pth')

        # Check saved format - compatible with training script format
        if 'model_state_dict' in gnn_checkpoint:
            # New format: contains model state dict, random seed and config
            model_config = gnn_checkpoint.get('model_config', {
                'num_node_features': 3,
                'hidden_dim': 16,
                'output_dim': 1
            })
            gnn_model = GNNModel(
                num_node_features=model_config['num_node_features'],
                hidden_dim=model_config['hidden_dim'],
                output_dim=model_config['output_dim']
            )
            gnn_model.load_state_dict(gnn_checkpoint['model_state_dict'])
            print("✓ Loaded GNN model from new format")
        else:
            # Old format: directly state dictionary
            gnn_model = GNNModel(num_node_features=3, hidden_dim=16, output_dim=1)
            gnn_model.load_state_dict(gnn_checkpoint)
            print("✓ Loaded GNN model from old format")

        gnn_model.eval()

        print("✓ All models and scalers loaded successfully")

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("Troubleshooting tips:")
        print("1. Make sure all required files exist:")
        print("   - Final_model_Maths_GNN_features_normalized.keras or other model files")
        print("   - gnn_model_weights.pth")
        print("   - maths_scaler.pkl, gnn_scaler.pkl, electron_scaler.pkl")
        print("2. Check if the GNN model format matches the expected format")
        return

    # Initialize graph generator
    graph_generator = ExtendedMolecularGraphGenerator()

    # Read multi-atomic molecules data from CSV
    print("Reading multi-atomic molecules data from polyatomic_output_final.csv...")
    try:
        data = pd.read_csv('validation-set_2026summer.csv')

        # Extract data using same column indices as training script
        sample_names = data.iloc[:, 0].values
        total_electrons = data.iloc[:, 1].values
        # Skip column 2 (bond_lengths) - using XYZ files for structure instead
        A = data.iloc[:, 3].values
        B = data.iloc[:, 6].values  # Added B column for Maths features
        C = data.iloc[:, 8].values
        D = data.iloc[:, 9].values
        L = data.iloc[:, 10].values

        print(f"✓ Successfully loaded {len(sample_names)} molecules from CSV")
        print(f"  Columns used: A (col 3), B (col 5), C (col 6), D (col 7), L (col 10)")

    except Exception as e:
        print(f"✗ Error loading validation-set: {e}")
        return

    # Collect data for all molecules
    graph_data_list = []
    molecule_info = []

    for i in range(len(sample_names)):
        name = sample_names[i]
        xyz_file = f"{name}.xyz"

        print(f"\nProcessing {name}:")
        print(f"  Looking for XYZ file: {xyz_file}")

        try:
            # Check if XYZ file exists
            if not os.path.exists(xyz_file):
                print(f"  ✗ XYZ file not found: {xyz_file}")
                continue

            elements, coordinates, file_content = graph_generator.parse_xyz_file(xyz_file)

            # Print the actual content of the XYZ file
            print(f"  ✓ XYZ file content:")
            for line in file_content:
                print(f"    {line.strip()}")

            graph_data = graph_generator.build_graph(elements, coordinates)

            print(f"  Atoms: {len(elements)} ({elements})")
            print(f"  Node features shape: {graph_data.x.shape}")
            print(f"  Number of edges: {graph_data.edge_index.shape[1]}")

            graph_data_list.append(graph_data)

            molecule_info.append({
                'name': name,
                'total_electrons': total_electrons[i],
                'A': A[i],
                'B': B[i],  # Added B value
                'C': C[i],
                'D': D[i],
                'L': L[i],
                'num_atoms': len(elements)
            })

        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
            continue

    if not molecule_info:
        print("No molecules processed successfully.")
        return

    # Batch process GNN outputs
    print(f"\nBatch processing GNN outputs for {len(molecule_info)} molecules...")

    try:
        # Create batch from all molecules
        batch_data = Batch.from_data_list(graph_data_list)

        # Run GNN inference
        with torch.no_grad():
            gnn_output = gnn_model(batch_data)

        gnn_output_np = gnn_output.numpy().flatten()
        print(f"Raw GNN outputs: {gnn_output_np}")

        # Normalize GNN outputs if scaler is available
        if gnn_scaler is not None:
            gnn_output_normalized = gnn_scaler.transform(gnn_output_np.reshape(-1, 1)).flatten()
            print(f"Normalized GNN outputs: {gnn_output_normalized}")
        else:
            gnn_output_normalized = gnn_output_np
            print("⚠  Using non-normalized GNN outputs")

    except Exception as e:
        print(f"✗ Error in GNN processing: {e}")
        return

    # Make predictions for each molecule
    print(f"\nMaking predictions with normalized features...")
    results = []

    for i, mol_data in enumerate(molecule_info):
        try:
            # Calculate Maths features (12 features)
            maths_features = calculate_maths_features(
                mol_data['A'], mol_data['B'], mol_data['C'], mol_data['D']
            )

            # Normalize Maths features if scaler is available
            if maths_scaler is not None:
                maths_features_normalized = maths_scaler.transform(maths_features.reshape(1, -1)).flatten()
            else:
                maths_features_normalized = maths_features
                print("⚠  Using non-normalized Maths features")

            # Normalize electrons
            electrons_normalized = electron_scaler.transform([[mol_data['total_electrons']]])[0, 0]

            # Combine normalized features - same order as training: 12 Maths + 1 GNN + 1 Electron = 14 features
            combined_features = np.hstack([
                maths_features_normalized,  # 12 normalized Maths features
                gnn_output_normalized[i],   # 1 normalized GNN feature
                electrons_normalized        # 1 normalized electron feature
            ]).reshape(1, -1)

            print(f"\n{mol_data['name']} feature summary:")
            print(f"  Maths features (normalized): {maths_features_normalized[:3]}...")  # Show first 3
            print(f"  GNN feature (normalized): {gnn_output_normalized[i]:.6f}")
            print(f"  Electron feature (normalized): {electrons_normalized:.6f}")

            # Make prediction - model predicts (L-D)/total_electrons
            prediction = keras_model.predict(combined_features, verbose=0)
            predicted_normalized_target = prediction[0, 0]

            # Calculate actual values
            actual_L_minus_D = mol_data['L'] - mol_data['D']
            actual_normalized_target = actual_L_minus_D / mol_data['total_electrons']

            # Convert predicted normalized target back to L-D
            predicted_L_minus_D = predicted_normalized_target * mol_data['total_electrons']

            # Calculate predicted L value
            predicted_L = predicted_L_minus_D + mol_data['D']

            # Calculate errors with signs (not absolute values)
            error_normalized_target = predicted_normalized_target - actual_normalized_target
            error_L_minus_D = predicted_L_minus_D - actual_L_minus_D
            error_L = predicted_L - mol_data['L']

            # Hartree to kcal/mol conversion (with sign)
            hartree_to_kcal_mol = 627.509
            error_L_minus_D_kcal_mol = error_L_minus_D * hartree_to_kcal_mol
            error_L_kcal_mol = error_L * hartree_to_kcal_mol

            print(f"{mol_data['name']} ({mol_data['num_atoms']} atoms):")
            print(f"  Actual (L-D)/total_electrons: {actual_normalized_target:.6f}")
            print(f"  Predicted (L-D)/total_electrons: {predicted_normalized_target:.6f}")
            print(f"  Error (L-D)/total_electrons: {error_normalized_target:+.6f}")

            print(f"  Actual L-D: {actual_L_minus_D:.6f}")
            print(f"  Predicted L-D: {predicted_L_minus_D:.6f}")
            print(f"  Error L-D: {error_L_minus_D:+.6f} Hartree ({error_L_minus_D_kcal_mol:+.2f} kcal/mol)")

            print(f"  Actual L: {mol_data['L']:.6f}")
            print(f"  Predicted L: {predicted_L:.6f}")
            print(f"  Error L: {error_L:+.6f} Hartree ({error_L_kcal_mol:+.2f} kcal/mol)")

            results.append({
                'molecule': mol_data['name'],
                'atoms': mol_data['num_atoms'],
                'actual_normalized_target': actual_normalized_target,
                'predicted_normalized_target': predicted_normalized_target,
                'error_normalized_target': error_normalized_target,
                'actual_L_minus_D': actual_L_minus_D,
                'predicted_L_minus_D': predicted_L_minus_D,
                'error_L_minus_D': error_L_minus_D,
                'error_L_minus_D_kcal_mol': error_L_minus_D_kcal_mol,
                'actual_L': mol_data['L'],
                'predicted_L': predicted_L,
                'error_L': error_L,
                'error_L_kcal_mol': error_L_kcal_mol
            })

        except Exception as e:
            print(f"✗ Error predicting {mol_data['name']}: {e}")
            continue

    # Print summary
    if results:
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY (WITH NORMALIZED FEATURES)")
        print("=" * 80)

        df_results = pd.DataFrame(results)

        # Display results with signed errors
        print("\nDetailed Results:")
        display_columns = [
            'molecule', 'atoms',
            'actual_normalized_target', 'predicted_normalized_target', 'error_normalized_target',
            'actual_L_minus_D', 'predicted_L_minus_D', 'error_L_minus_D', 'error_L_minus_D_kcal_mol',
            'actual_L', 'predicted_L', 'error_L', 'error_L_kcal_mol'
        ]
        print(df_results[display_columns].to_string(index=False, float_format='%+.6f'))

        # Performance analysis by atom count
        print(f"\nPerformance Analysis by Atom Count:")
        for atoms in sorted(df_results['atoms'].unique()):
            subset = df_results[df_results['atoms'] == atoms]
            avg_error_normalized = subset['error_normalized_target'].mean()
            avg_error_L_minus_D = subset['error_L_minus_D'].mean()
            avg_error_L_minus_D_kcal = subset['error_L_minus_D_kcal_mol'].mean()
            avg_error_L = subset['error_L'].mean()
            avg_error_L_kcal = subset['error_L_kcal_mol'].mean()

            print(f"  {atoms} atoms: {len(subset)} molecules")
            print(f"    Average Error (L-D)/total_electrons: {avg_error_normalized:+.6f}")
            print(f"    Average Error L-D: {avg_error_L_minus_D:+.6f} Hartree ({avg_error_L_minus_D_kcal:+.2f} kcal/mol)")
            print(f"    Average Error L: {avg_error_L:+.6f} Hartree ({avg_error_L_kcal:+.2f} kcal/mol)")

        overall_avg_error_normalized = df_results['error_normalized_target'].mean()
        overall_avg_error_L_minus_D = df_results['error_L_minus_D'].mean()
        overall_avg_error_L_minus_D_kcal = df_results['error_L_minus_D_kcal_mol'].mean()
        overall_avg_error_L = df_results['error_L'].mean()
        overall_avg_error_L_kcal = df_results['error_L_kcal_mol'].mean()

        print(f"\nOverall Performance:")
        print(f"  Average Error (L-D)/total_electrons: {overall_avg_error_normalized:+.6f}")
        print(f"  Average Error L-D: {overall_avg_error_L_minus_D:+.6f} Hartree ({overall_avg_error_L_minus_D_kcal:+.2f} kcal/mol)")
        print(f"  Average Error L: {overall_avg_error_L:+.6f} Hartree ({overall_avg_error_L_kcal:+.2f} kcal/mol)")

        # Performance interpretation based on L-D error
        print(f"\nPerformance Interpretation (based on L-D error):")
        avg_abs_error_L_minus_D_kcal = df_results['error_L_minus_D_kcal_mol'].abs().mean()
        if avg_abs_error_L_minus_D_kcal < 1.0:
            print("  ✓ EXCELLENT: Model generalizes well to larger molecules")
        elif avg_abs_error_L_minus_D_kcal < 3.0:
            print("  ✓ GOOD: Reasonable performance on larger molecules")
        elif avg_abs_error_L_minus_D_kcal < 5.0:
            print("  ⚠️  FAIR: Some generalization capability")
        else:
            print("  ✗ POOR: Limited generalization to larger molecules")

        # Feature information
        print(f"\nFeature Information:")
        print(f"  - Using 14 NORMALIZED features: 12 Maths features + 1 GNN feature + 1 Electron feature")
        print(f"  - All features are standardized using StandardScaler")
        print(f"  - Target variable: (L-D)/total_electrons")
        print(f"  - Model predicts normalized target, then converted back to L-D and L")

        # Model and normalization status
        print(f"\nModel and Normalization Status:")
        model_name = "Normalized" if 'normalized' in keras_model.__class__.__name__.lower() else "Non-normalized"
        print(f"  - Model type: {model_name}")
        print(f"  - Maths features normalized: {maths_scaler is not None}")
        print(f"  - GNN features normalized: {gnn_scaler is not None}")
        print(f"  - Electron features normalized: {electron_scaler is not None}")

        # Error bias analysis
        print(f"\nError Bias Analysis:")
        positive_errors = (df_results['error_L'] > 0).sum()
        negative_errors = (df_results['error_L'] < 0).sum()
        zero_errors = (df_results['error_L'] == 0).sum()
        print(f"  - Positive errors (overestimation): {positive_errors} molecules")
        print(f"  - Negative errors (underestimation): {negative_errors} molecules")
        print(f"  - Zero errors: {zero_errors} molecules")

        if overall_avg_error_L > 0:
            print(f"  - Overall bias: Overestimation ({overall_avg_error_L:+.6f} Hartree)")
        elif overall_avg_error_L < 0:
            print(f"  - Overall bias: Underestimation ({overall_avg_error_L:+.6f} Hartree)")
        else:
            print(f"  - Overall bias: No bias")

        # Save results to CSV
        output_file = "validation_results_maths_gnn-DTQ_normalized.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n✓ Validation results saved to {output_file}")

    else:
        print("\nNo successful validations completed.")

if __name__ == "__main__":
    validate_on_multi_atomic_molecules()
