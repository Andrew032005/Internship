import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch_geometric.loader import DataLoader

# Conversion factor: 1 Hartree = 627.509 kcal/mol
H_TO_KCAL = 627.509

def visualize():
    if not os.path.exists('train_results.pkl'):
        print("Results file not found.")
        return

    with open('train_results.pkl', 'rb') as f:
        results, train_losses, val_losses, domain_imp = pickle.load(f)

    # 1. Learning Loss Curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Learning Loss Curves')
    plt.yscale('log')
    plt.legend()
    plt.savefig('plot1_loss_curves.png')
    plt.close()

    # Data for other plots
    train_results = [r for d in [results] for r in d if r['is_train']]
    val_results = [r for d in [results] for r in d if not r['is_train']]

    # 2. Scatter Plot for Correction (Normalized or physical?)
    # Technical assignment: "Scatter Plot Delta_E: Axis X: True diff, Axis Y: Pred correction (after denormalization)"
    plt.figure(figsize=(8, 6))
    plt.scatter([r['true_corr'] for r in train_results], [r['pred_corr'] for r in train_results],
                alpha=0.5, label='Train (Dimers)', color='blue')
    plt.scatter([r['true_corr'] for r in val_results], [r['pred_corr'] for r in val_results],
                alpha=0.5, label='Val (Polyatomics)', color='orange')

    all_corr = [r['true_corr'] for r in results]
    min_c, max_c = min(all_corr), max(all_corr)
    plt.plot([min_c, max_c], [min_c, max_c], 'r--', label='Ideal')
    plt.xlabel('True Difference (Hartree)')
    plt.ylabel('Predicted Correction (Hartree)')
    plt.title('Scatter Plot: Correction')
    plt.legend()
    plt.savefig('plot2_scatter_correction.png')
    plt.close()

    # 3. Scatter Plot for full restored energy
    plt.figure(figsize=(8, 6))
    plt.scatter([r['true_ccsdtq'] for r in train_results], [r['pred_ccsdtq'] for r in train_results],
                alpha=0.5, label='Train (Dimers)', color='blue')
    plt.scatter([r['true_ccsdtq'] for r in val_results], [r['pred_ccsdtq'] for r in val_results],
                alpha=0.5, label='Val (Polyatomics)', color='orange')

    all_e = [r['true_ccsdtq'] for r in results if not np.isnan(r['true_ccsdtq'])]
    if all_e:
        min_e, max_e = min(all_e), max(all_e)
        plt.plot([min_e, max_e], [min_e, max_e], 'r--', label='Ideal')
    plt.xlabel('True CCSDTQ Energy (Hartree)')
    plt.ylabel('Restored CCSDTQ Energy (Hartree)')
    plt.title('Scatter Plot: Full Energy')
    plt.legend()
    plt.savefig('plot3_scatter_ccsdtq.png')
    plt.close()

    # 4. Error Distribution Histogram (kcal/mol)
    train_errors = [abs(r['pred_ccsdtq'] - r['true_ccsdtq']) * H_TO_KCAL for r in train_results if not np.isnan(r['true_ccsdtq'])]
    val_errors = [abs(r['pred_ccsdtq'] - r['true_ccsdtq']) * H_TO_KCAL for r in val_results if not np.isnan(r['true_ccsdtq'])]

    plt.figure(figsize=(8, 6))
    plt.hist(train_errors, bins=50, alpha=0.5, label='Train (Dimers)', density=True)
    plt.hist(val_errors, bins=50, alpha=0.5, label='Val (Polyatomics)', density=True)
    plt.xlabel('Absolute Error (kcal/mol)')
    plt.ylabel('Density')
    plt.title('Error Distribution Histogram')
    plt.legend()
    plt.savefig('plot4_error_hist.png')
    plt.close()

    # 5. Feature Importance (Domain-based importance)
    plt.figure(figsize=(8, 6))

    domain_labels = ['Maths', 'GNN', 'N_elec']
    plt.bar(domain_labels, domain_imp, color=['blue', 'green', 'red'])
    plt.ylabel('Gradient Saliency (normalized)')
    plt.title('Feature Importance by Domain')
    plt.savefig('plot5_feature_importance.png')
    plt.close()

    # 6. Dependence of error on system size (N atoms)
    all_valid_results = [r for r in results if not np.isnan(r['true_ccsdtq'])]
    sizes = sorted(list(set([r['n_atoms'] for r in all_valid_results])))
    avg_errors = []
    for s in sizes:
        errs = [abs(r['pred_ccsdtq'] - r['true_ccsdtq']) * H_TO_KCAL for r in all_valid_results if r['n_atoms'] == s]
        avg_errors.append(np.mean(errs))

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, avg_errors, marker='o', linestyle='-')
    plt.xlabel('Number of Atoms (N)')
    plt.ylabel('Mean Absolute Error (kcal/mol)')
    plt.title('Error vs Molecular Size')
    plt.grid(True)
    plt.savefig('plot6_error_vs_size.png')
    plt.close()

    print("All 6 plots saved.")

if __name__ == "__main__":
    visualize()
