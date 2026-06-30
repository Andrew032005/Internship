import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class HybridGNN(nn.Module):
    def __init__(self, node_in_feats=3, gnn_out_feats=8, math_feats_dim=12, rdkit_feats_dim=5, mlp_hidden_dims=[64, 32], dropout=0.15, l2_reg=1e-4):
        super(HybridGNN, self).__init__()

        # GNN Branch
        self.conv1 = GCNConv(node_in_feats, 16)
        self.conv2 = GCNConv(16, gnn_out_feats)

        # Total input dimension for MLP: GNN (8) + Math (12) + N_elec (1) + RDKit (K)
        # Total = 8 + 12 + 1 + 5 = 26
        input_dim = gnn_out_feats + math_feats_dim + 1 + rdkit_feats_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1)
        )

        self.l2_reg = l2_reg

    def forward(self, data, math_feats, n_elec, rdkit_feats):
        # GNN branch
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling
        gnn_emb = global_mean_pool(x, data.batch) # (batch_size, gnn_out_feats)

        # Concatenate features
        # Assuming math_feats, n_elec, rdkit_feats are already tensors of appropriate shape
        combined = torch.cat([gnn_emb, math_feats, n_elec, rdkit_feats], dim=1)

        # MLP branch
        out = self.mlp(combined)
        return out

def get_l2_loss(model):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, 2)
    return l2_loss

if __name__ == "__main__":
    model = HybridGNN()
    print(model)
