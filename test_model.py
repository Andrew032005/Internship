import unittest
import torch
from model import HybridGNN
from torch_geometric.data import Data, Batch

class TestModel(unittest.TestCase):
    def test_forward_pass(self):
        model = HybridGNN(node_in_feats=3, gnn_out_feats=8, math_feats_dim=12, rdkit_feats_dim=5)

        # Dummy graph data
        x = torch.randn((4, 3))
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        # Create a Batch object
        data = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([data])

        math_feats = torch.randn((1, 12))
        n_elec = torch.randn((1, 1))
        rdkit_feats = torch.randn((1, 5))

        output = model(batch, math_feats, n_elec, rdkit_feats)
        self.assertEqual(output.shape, (1, 1))
        self.assertFalse(torch.isnan(output).any())

    def test_gradients(self):
        model = HybridGNN(node_in_feats=3, gnn_out_feats=8, math_feats_dim=12, rdkit_feats_dim=5)

        x = torch.randn((4, 3))
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([data])

        math_feats = torch.randn((1, 12))
        n_elec = torch.randn((1, 1))
        rdkit_feats = torch.randn((1, 5))

        output = model(batch, math_feats, n_elec, rdkit_feats)
        loss = output.sum()
        loss.backward()

        # Check gradients in GNN branch
        self.assertIsNotNone(model.conv1.lin.weight.grad)
        # Check gradients in MLP branch
        self.assertIsNotNone(model.mlp[0].weight.grad)

if __name__ == '__main__':
    unittest.main()
