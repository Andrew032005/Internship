import unittest
import numpy as np
import torch
import os
from data_prep import parse_xyz, calculate_maths_features, get_rdkit_descriptors, build_graph

class TestDataPrep(unittest.TestCase):
    def test_parse_xyz(self):
        # Create a dummy xyz file
        with open('test.xyz', 'w') as f:
            f.write("2\nTest molecule\nH 0 0 0\nH 0 0 1\n")
        atoms, coords = parse_xyz('test.xyz')
        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms[0], 1)
        # Check centering: mean should be approx 0
        self.assertTrue(np.allclose(np.mean(coords, axis=0), 0))
        os.remove('test.xyz')

    def test_calculate_maths_features(self):
        features = calculate_maths_features(-100, -100.1, -100.2, -100.3)
        self.assertEqual(len(features), 12)
        for f in features:
            self.assertFalse(np.isnan(f))
            self.assertFalse(np.isinf(f))

    def test_get_rdkit_descriptors(self):
        atoms = np.array([1, 1])
        coords = np.array([[0, 0, 0], [0, 0, 0.74]])
        desc = get_rdkit_descriptors(atoms, coords)
        self.assertEqual(len(desc), 5)
        # Our masking logic: if diatomic and TPSA=0, it should become -1
        # H2 TPSA is 0.
        self.assertIn(-1.0, desc)

    def test_build_graph(self):
        atoms = np.array([1, 1])
        coords = np.array([[0, 0, 0], [0, 0, 1.0]])
        graph = build_graph(atoms, coords)
        self.assertEqual(graph.x.shape, (2, 3))
        # Edge exists because 1.0 < 1.6
        self.assertEqual(graph.edge_index.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
