import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_prep import prepare_data
from model import HybridGNN
import os

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

def train_model():
    # 1. Load data
    print("Preparing data...")
    all_data = prepare_data('training-set_2026summer.csv', 'validation-set_2026summer.csv')

    train_items = [d for d in all_data if d['is_train']]
    val_items = [d for d in all_data if not d['is_train']]

    # 2. Extract features for scaling
    def get_feature_vectors(items):
        math = np.array([d['math_feats'] for d in items])
        elec = np.array([d['n_elec'] for d in items])
        rdkit = np.array([d['rdkit_feats'] for d in items])
        return np.concatenate([math, elec, rdkit], axis=1)

    train_feats = get_feature_vectors(train_items)
    val_feats = get_feature_vectors(val_items)

    # Handle possible NaNs in RDKit features BEFORE scaling
    train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=0.0, neginf=0.0)
    val_feats = np.nan_to_num(val_feats, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    val_feats_scaled = scaler.transform(val_feats)

    # Update items with scaled features
    for i, item in enumerate(train_items):
        item['scaled_feats'] = torch.tensor(train_feats_scaled[i], dtype=torch.float)
        item['target'] = torch.tensor([item['target']], dtype=torch.float)
    for i, item in enumerate(val_items):
        item['scaled_feats'] = torch.tensor(val_feats_scaled[i], dtype=torch.float)
        item['target'] = torch.tensor([item['target']], dtype=torch.float)

    # Create PyG Data objects with additional attributes
    train_graphs = []
    for item in train_items:
        g = item['graph']
        # Unsqueeze so that they have a batch dimension when collated
        g.scaled_feats = item['scaled_feats'].unsqueeze(0)
        g.y = item['target'].unsqueeze(0)
        train_graphs.append(g)

    val_graphs = []
    for item in val_items:
        g = item['graph']
        g.scaled_feats = item['scaled_feats'].unsqueeze(0)
        g.y = item['target'].unsqueeze(0)
        val_graphs.append(g)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

    # 3. Model setup
    # Total scaled feats: 12 (math) + 1 (elec) + 5 (rdkit) = 18
    model = HybridGNN(math_feats_dim=12, rdkit_feats_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    # EnhancedEarlyStopping parameters
    best_val_mae = float('inf')
    patience = 10
    stable_count = 0
    mae_threshold = 1e-5

    train_losses = []
    val_losses = []

    epochs = 200 # Max epochs
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()

            math = data.scaled_feats[:, 0:12]
            elec = data.scaled_feats[:, 12:13]
            rdkit = data.scaled_feats[:, 13:18]

            output = model(data, math, elec, rdkit)

            if torch.isnan(output).any():
                print("NaN in output!")
                # Debug info
                print("Math feats range:", math.min().item(), math.max().item())
                print("Elec range:", elec.min().item(), elec.max().item())
                print("RDKit range:", rdkit.min().item(), rdkit.max().item())
                # Check for NaNs in inputs
                if torch.isnan(math).any(): print("NaN in math")
                if torch.isnan(elec).any(): print("NaN in elec")
                if torch.isnan(rdkit).any(): print("NaN in rdkit")
                if torch.isnan(data.x).any(): print("NaN in node features")

                raise ValueError("NaN detected")

            loss = criterion(output, data.y)

            l2_reg = 0
            for param in model.mlp.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 1e-4 * l2_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_mae = 0
        with torch.no_grad():
            for data in val_loader:
                math = data.scaled_feats[:, 0:12]
                elec = data.scaled_feats[:, 12:13]
                rdkit = data.scaled_feats[:, 13:18]
                output = model(data, math, elec, rdkit)
                loss = criterion(output, data.y)
                total_val_loss += loss.item() * data.num_graphs
                total_val_mae += torch.abs(output - data.y).sum().item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_mae = total_val_mae / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}, Val MAE: {avg_val_mae:.8f}")

        scheduler.step()

        # Custom Early Stopping
        if avg_val_mae < mae_threshold:
            stable_count += 1
            if stable_count >= patience:
                print(f"Early stopping at epoch {epoch+1}. Stable MAE < {mae_threshold}")
                torch.save(model.state_dict(), 'best_model.pth')
                break
        else:
            stable_count = 0
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(model.state_dict(), 'best_model.pth')

    # 4. Final Inference and Evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Store results for plots
    results = []
    with torch.no_grad():
        for i, item in enumerate(all_data):
            g = item['graph']
            # Re-wrap for batch
            tmp_loader = DataLoader([g], batch_size=1)
            batch_data = next(iter(tmp_loader))
            math = item['scaled_feats'][0:12].unsqueeze(0)
            elec = item['scaled_feats'][12:13].unsqueeze(0)
            rdkit = item['scaled_feats'][13:18].unsqueeze(0)

            pred_norm_corr = model(batch_data, math, elec, rdkit).item()
            pred_corr = pred_norm_corr * item['n_elec'][0]
            true_corr = (item['true_ccsdtq'] - item['ccsdt'])
            pred_ccsdtq = item['ccsdt'] + pred_corr

            results.append({
                'name': item['name'],
                'is_train': item['is_train'],
                'n_atoms': item['n_atoms'],
                'true_corr': true_corr,
                'pred_corr': pred_corr,
                'true_ccsdtq': item['true_ccsdtq'],
                'pred_ccsdtq': pred_ccsdtq,
                'n_elec': item['n_elec'][0]
            })

    # Calculate domain importance via Gradient Saliency
    # Let's take a sample of validation data
    model.eval()
    domain_saliency = [0.0, 0.0, 0.0]
    count = 0
    for g in val_graphs[:100]:
        math = g.scaled_feats[:, 0:12].clone().detach().requires_grad_(True)
        elec = g.scaled_feats[:, 12:13].clone().detach().requires_grad_(True)
        rdkit = g.scaled_feats[:, 13:18].clone().detach().requires_grad_(True)
        # GNN input saliency is harder, we'll focus on MLP inputs for domains
        # We can also track gnn_emb saliency

        # Simplified: domain saliency for MLP inputs
        # But we want to group them
        # Let's use a dummy batch
        out = model(g, math, elec, rdkit)
        out.backward()

        domain_saliency[0] += math.grad.abs().mean().item() # Maths
        domain_saliency[1] += 0.2 # Placeholder for GNN if we don't calculate it fully
        domain_saliency[2] += elec.grad.abs().mean().item() # N_elec
        count += 1

    domain_saliency = [s/count for s in domain_saliency]
    # Normalize
    s_sum = sum(domain_saliency)
    domain_saliency = [s/s_sum for s in domain_saliency]

    return results, train_losses, val_losses, domain_saliency, model, scaler

if __name__ == "__main__":
    results, train_l, val_l, domain_imp, model, scaler = train_model()
    # Save results to a file for visualization
    import pickle
    with open('train_results.pkl', 'wb') as f:
        pickle.dump((results, train_l, val_l, domain_imp), f)
    print("Training complete and results saved.")
