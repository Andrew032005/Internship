import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, Callback
import joblib
import random

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# Set random seeds for reproducibility across all libraries
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set random seeds at the beginning
set_random_seeds(42)

class MolecularGraphGenerator:
    """Molecular Graph Generator - Builds molecular structure from molecular names and bond lengths"""

    def __init__(self):
        self.element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17
        }

    def parse_molecule_name(self, name):
        """Parse molecular names to return atomic types and counts"""
        if name in self.element_to_z:
            return [self.element_to_z[name]], [name], 0  # Single atom
        if name.endswith('2') and name[:-1] in self.element_to_z:
            element = name[:-1]
            z = self.element_to_z[element]
            return [z, z], [element, element], 1  # Homonuclear diatomic
        for i in range(1, len(name)):
            part1, part2 = name[:i], name[i:]
            if part1 in self.element_to_z and part2 in self.element_to_z:
                z1 = self.element_to_z[part1]
                z2 = self.element_to_z[part2]
                return [z1, z2], [part1, part2], 1  # Heteronuclear diatomic

        print(f"Warning: Unable to parse molecule name '{name}', defaulting to single atom")
        return [1], ['H'], 0  # Default to Hydrogen atom

    def build_graph(self, name, bond_length):
        """Build graph representation for the molecule - now returns complete Data object"""
        atomic_numbers, elements, is_diatomic = self.parse_molecule_name(name)

        # Initialize features with zeros for padding to size 3
        num_atoms = len(atomic_numbers)
        node_features = np.zeros((num_atoms, 3))

        for i in range(num_atoms):
            node_features[i, 0] = atomic_numbers[i]  # Atomic number of current atom

            # For atoms with neighbors, include neighbor info
            if is_diatomic and i == 0:  # First atom gets second atom's info
                node_features[i, 1] = atomic_numbers[1]
            elif is_diatomic and i == 1:  # Second atom gets first atom's info
                node_features[i, 1] = atomic_numbers[0]

            node_features[i, 2] = bond_length  # Bond length

        # Create edge index
        if is_diatomic:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Return as PyG Data object with num_atoms information
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            num_atoms=num_atoms,
            num_nodes=num_atoms
        )

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=16, output_dim=1):
        super(GNNModel, self).__init__()
        # Initialize with specific random seed for reproducibility
        torch.manual_seed(42)
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
    Simple global mean pooling implementation
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
    """Calculate normalized and robust mathematical features"""
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

    return np.array([F1_norm, F2_norm, F3_norm, F4, F5, F6, F7, F8, F9, F10, F11, F12]).T  # Transpose to have features in rows

# Enhanced early stopping and logging callback
class EnhancedEarlyStopping(Callback):
    def __init__(self, mae_thresh, mse_thresh, consecutive_epochs=5):
        super().__init__()
        self.mae_thresh = mae_thresh
        self.mse_thresh = mse_thresh
        self.consecutive_epochs = consecutive_epochs
        self.consecutive_success = 0
        self.best_weights = None
        self.best_mse = float('inf')
        self.best_mae = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_mae = logs.get('val_mae')
        val_mse = logs.get('val_loss')
        train_mae = logs.get('mae')
        train_mse = logs.get('loss')

        if val_mae is None or val_mse is None:
            print("Warning: Validation metrics not available")
            return

        # Update best weights
        if val_mse < self.best_mse:
            self.best_mse = val_mse
            self.best_mae = val_mae
            self.best_weights = self.model.get_weights()

        # Detailed output for each epoch's metrics
        print(f"\nEpoch {epoch+1}:")
        print(f"  Training - MSE: {train_mse:.9f}, MAE: {train_mae:.9f}")
        print(f"  Validation - MSE: {val_mse:.9f}, MAE: {val_mae:.9f}")
        print(f"  Target - MSE < {self.mse_thresh}, MAE < {self.mae_thresh}")

        # Check stopping condition
        if val_mae < self.mae_thresh and val_mse < self.mse_thresh:
            self.consecutive_success += 1
            print(f"  Consecutive success: {self.consecutive_success}/{self.consecutive_epochs}")

            if self.consecutive_success >= self.consecutive_epochs:
                print(f"\n*** Early stopping triggered ***")
                print(f"MAE={val_mae:.9f} < {self.mae_thresh} and MSE={val_mse:.9f} < {self.mse_thresh}")
                print(f"for {self.consecutive_epochs} consecutive epochs")
                self.model.stop_training = True
                # Restore best weights
                self.model.set_weights(self.best_weights)
        else:
            self.consecutive_success = 0
            print(f"  Conditions not met, resetting counter")

# Feature correlation analysis function
def analyze_feature_correlations(features, feature_names, target):
    """Analyze correlations between features and target variable"""
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS (Before Training)")
    print("="*80)

    # Create DataFrame with all features and target
    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df['Target_(L-D)/total_electrons'] = target

    # Calculate correlation matrix
    correlation_matrix = feature_df.corr()

    # Print correlations with target variable
    target_correlations = correlation_matrix['Target_(L-D)/total_electrons'].drop('Target_(L-D)/total_electrons')
    print("Correlation with Target ((L-D)/total_electrons):")
    for feature, corr in target_correlations.items():
        print(f"  {feature}: {corr:.6f}")

    # Print feature-feature correlations
    print("\nFeature-Feature Correlations (top 5 highest):")
    feature_corrs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = correlation_matrix.iloc[i, j]
            feature_corrs.append((feature_names[i], feature_names[j], corr))

    # Sort by absolute value and take top 5
    feature_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    for feat1, feat2, corr in feature_corrs[:5]:
        print(f"  {feat1} vs {feat2}: {corr:.6f}")

    return correlation_matrix

# Feature importance analysis based on correlation
def analyze_feature_importance_correlation(features, feature_names, target):
    """Feature importance analysis based on correlation"""
    print("\n" + "="*80)
    print("CORRELATION-BASED FEATURE IMPORTANCE (Before Training)")
    print("="*80)

    # Calculate absolute correlation with target for each feature
    correlations = []
    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(features[:, i], target)[0, 1]
        correlations.append((feature_name, abs(corr)))

    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    # Calculate relative importance percentage
    total_correlation = sum(corr for _, corr in correlations)

    print("Feature Importance (based on absolute correlation with target):")
    for feature_name, corr in correlations:
        relative_importance = (corr / total_correlation) * 100
        print(f"  {feature_name}: {corr:.6f} ({relative_importance:.1f}%)")

    return correlations

# Improved feature importance analysis
def improved_feature_importance_analysis(model, feature_names):
    """Improved feature importance analysis (based on weights and correlation)"""
    print("\n" + "="*80)
    print("IMPROVED FEATURE IMPORTANCE ANALYSIS (After Training)")
    print("="*80)

    # Get model weights
    weights_layer1 = model.layers[0].get_weights()[0]  # First layer weights
    weights_layer2 = model.layers[1].get_weights()[0]  # Second layer weights

    # Calculate total impact for each feature
    importance_scores = []
    for feature_idx in range(weights_layer1.shape[0]):
        # Feature impact to all neurons in first layer
        impact_to_layer1 = np.sum(np.abs(weights_layer1[feature_idx, :]))

        # Consider propagation to second layer (simplified calculation)
        layer1_to_output = np.mean(np.abs(weights_layer2))
        total_impact = impact_to_layer1 * layer1_to_output

        importance_scores.append(total_impact)

    # Normalize
    importance_scores = np.array(importance_scores)
    total = np.sum(importance_scores)
    relative_importance = (importance_scores / total) * 100

    print("Improved Weight-based Feature Importance:")
    results = []
    for name, score, rel in zip(feature_names, importance_scores, relative_importance):
        results.append((name, score, rel))
        print(f"  {name}: {score:.4f} ({rel:.1f}%)")

    return results

# Read data from CSV
print("Reading training data...")
data = pd.read_csv('validation-set_2026summer.csv')

# Extract sample names and other columns
sample_names = data.iloc[:, 0].values
total_electrons = data.iloc[:, 1].values
bond_lengths = data.iloc[:, 2].values
A = data.iloc[:, 3].values
B = data.iloc[:, 6].values  # Added B column for Maths features
C = data.iloc[:, 8].values
D = data.iloc[:, 9].values
L = data.iloc[:, 10].values

# Initialize molecular graph generator and create graph data list
graph_generator = MolecularGraphGenerator()
graph_data_list = []

# Prepare input for GNN - now creating separate Data objects for each molecule
print("Building molecular graphs...")
for i in range(len(data)):
    name = sample_names[i]
    bond_length = bond_lengths[i] if bond_lengths[i] > 0 else 0.0
    graph_data = graph_generator.build_graph(name, bond_length)

    if graph_data is not None:
        graph_data_list.append(graph_data)
    else:
        print(f"Warning: Unable to build graph for molecule '{name}' at index {i}")

# Create a batch from all graphs for processing
batch_data = Batch.from_data_list(graph_data_list)

# Check the processed graph data
print(f"Total molecular graphs: {len(graph_data_list)}")
print(f"Total nodes in batch: {batch_data.num_nodes}")
print(f"Total edges in batch: {batch_data.edge_index.shape[1]}")
print(f"Batch object: {batch_data}")

# Calculate Maths features (replacing CF features)
print("Calculating Maths features...")
X_maths = []
for i in range(len(data)):
    features = calculate_maths_features(A[i], B[i], C[i], D[i])
    X_maths.append(features)

X_maths = np.array(X_maths)

# Prepare target variable - MODIFIED: (L-D)/total_electrons
y = (L - D) / total_electrons

# ============================================================================
# 新增：特征归一化
# ============================================================================
print("\n" + "="*80)
print("FEATURE NORMALIZATION")
print("="*80)

# 创建归一化器
maths_scaler = StandardScaler()
gnn_scaler = StandardScaler()
electron_scaler = StandardScaler()

# 数据拆分 - 首先拆分索引以确保对齐
print("Splitting data into training and test sets...")
indices = np.arange(len(data))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# 拆分所有数据使用相同的索引
X_maths_train = X_maths[train_indices]
X_maths_test = X_maths[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

total_electrons_train = total_electrons[train_indices]
total_electrons_test = total_electrons[test_indices]

# 归一化特征
print("Normalizing features...")
# Maths特征归一化
X_maths_train_normalized = maths_scaler.fit_transform(X_maths_train)
X_maths_test_normalized = maths_scaler.transform(X_maths_test)

# 总电子数归一化
total_electrons_train_normalized = electron_scaler.fit_transform(total_electrons_train.reshape(-1, 1))
total_electrons_test_normalized = electron_scaler.transform(total_electrons_test.reshape(-1, 1))

# 创建GNN模型并进行全局池化
gnn_model = GNNModel(num_node_features=3, hidden_dim=16, output_dim=1)

# 获取GNN输出 - 现在无论原子数量如何，每个分子返回一个输出
print("Generating GNN features...")
gnn_model.eval()
with torch.no_grad():
    gnn_output = gnn_model(batch_data)

# 将gnn_output转换为numpy数组
gnn_output_np = gnn_output.numpy()

# GNN输出归一化
gnn_output_train = gnn_output_np[train_indices]
gnn_output_test = gnn_output_np[test_indices]

gnn_output_train_normalized = gnn_scaler.fit_transform(gnn_output_train)
gnn_output_test_normalized = gnn_scaler.transform(gnn_output_test)

# 打印归一化前后的统计信息
print("\nNormalization Statistics:")
print("Maths Features - Before normalization:")
print(f"  Train: mean={np.mean(X_maths_train):.4f}, std={np.std(X_maths_train):.4f}")
print(f"  After normalization: mean={np.mean(X_maths_train_normalized):.4f}, std={np.std(X_maths_train_normalized):.4f}")

print("GNN Output - Before normalization:")
print(f"  Train: mean={np.mean(gnn_output_train):.4f}, std={np.std(gnn_output_train):.4f}")
print(f"  After normalization: mean={np.mean(gnn_output_train_normalized):.4f}, std={np.std(gnn_output_train_normalized):.4f}")

print("Total Electrons - Before normalization:")
print(f"  Train: mean={np.mean(total_electrons_train):.4f}, std={np.std(total_electrons_train):.4f}")
print(f"  After normalization: mean={np.mean(total_electrons_train_normalized):.4f}, std={np.std(total_electrons_train_normalized):.4f}")

# 组合所有归一化后的特征：Maths特征(12) + GNN特征(1) + 总电子数(1) = 14个特征
print("Combining all normalized features...")
combined_train_features = np.hstack((
    X_maths_train_normalized,
    gnn_output_train_normalized,
    total_electrons_train_normalized
))

combined_test_features = np.hstack((
    X_maths_test_normalized,
    gnn_output_test_normalized,
    total_electrons_test_normalized
))

# ============================================================================
# 新功能1：将特征保存到新的CSV文件（包含归一化后的特征）
# ============================================================================
print("\n" + "="*80)
print("SAVING FEATURES TO NEW CSV FILE")
print("="*80)

# 创建包含所有特征的DataFrame
features_data = data.copy()

# 添加14个原始特征列
features_data['Maths_F1_norm'] = X_maths[:, 0]
features_data['Maths_F2_norm'] = X_maths[:, 1]
features_data['Maths_F3_norm'] = X_maths[:, 2]
features_data['Maths_F4'] = X_maths[:, 3]
features_data['Maths_F5'] = X_maths[:, 4]
features_data['Maths_F6'] = X_maths[:, 5]
features_data['Maths_F7'] = X_maths[:, 6]
features_data['Maths_F8'] = X_maths[:, 7]
features_data['Maths_F9'] = X_maths[:, 8]
features_data['Maths_F10'] = X_maths[:, 9]
features_data['Maths_F11'] = X_maths[:, 10]
features_data['Maths_F12'] = X_maths[:, 11]
features_data['GNN_Output'] = gnn_output_np.flatten()
features_data['Total_Electrons'] = total_electrons

# 添加归一化后的特征列
X_maths_normalized_all = maths_scaler.transform(X_maths)
gnn_output_normalized_all = gnn_scaler.transform(gnn_output_np)
total_electrons_normalized_all = electron_scaler.transform(total_electrons.reshape(-1, 1))

features_data['Maths_F1_norm_normalized'] = X_maths_normalized_all[:, 0]
features_data['Maths_F2_norm_normalized'] = X_maths_normalized_all[:, 1]
features_data['Maths_F3_norm_normalized'] = X_maths_normalized_all[:, 2]
features_data['Maths_F4_normalized'] = X_maths_normalized_all[:, 3]
features_data['Maths_F5_normalized'] = X_maths_normalized_all[:, 4]
features_data['Maths_F6_normalized'] = X_maths_normalized_all[:, 5]
features_data['Maths_F7_normalized'] = X_maths_normalized_all[:, 6]
features_data['Maths_F8_normalized'] = X_maths_normalized_all[:, 7]
features_data['Maths_F9_normalized'] = X_maths_normalized_all[:, 8]
features_data['Maths_F10_normalized'] = X_maths_normalized_all[:, 9]
features_data['Maths_F11_normalized'] = X_maths_normalized_all[:, 10]
features_data['Maths_F12_normalized'] = X_maths_normalized_all[:, 11]
features_data['GNN_Output_normalized'] = gnn_output_normalized_all.flatten()
features_data['Total_Electrons_normalized'] = total_electrons_normalized_all.flatten()

# 添加新的目标变量
features_data['Target_(L-D)/total_electrons'] = y

# 保存到新的CSV文件
features_data.to_csv('training-set_features_normalized.csv', index=False)
print("Features saved to 'training-set_features_normalized.csv'")
print(f"Original data shape: {data.shape}")
print(f"New data with features shape: {features_data.shape}")
print("New columns added: 12 Maths features + GNN_Output + Total_Electrons (both original and normalized) + Target_(L-D)/total_electrons")

# ============================================================================
# 新功能2：训练前特征分析（使用归一化后的特征）
# ============================================================================
# 准备完整的特征矩阵用于分析
all_features_normalized = np.hstack((X_maths_normalized_all, gnn_output_normalized_all, total_electrons_normalized_all))
feature_names_normalized = [
    'Maths_F1_norm_normalized', 'Maths_F2_norm_normalized', 'Maths_F3_norm_normalized',
    'Maths_F4_normalized', 'Maths_F5_normalized', 'Maths_F6_normalized',
    'Maths_F7_normalized', 'Maths_F8_normalized', 'Maths_F9_normalized',
    'Maths_F10_normalized', 'Maths_F11_normalized', 'Maths_F12_normalized',
    'GNN_Output_normalized', 'Total_Electrons_normalized'
]

# 运行相关性分析
correlation_matrix = analyze_feature_correlations(all_features_normalized, feature_names_normalized, y)

# 运行基于相关性的特征重要性分析
feature_importance_corr = analyze_feature_importance_correlation(all_features_normalized, feature_names_normalized, y)

# 继续原始的训练过程...
print(f"\nFeature dimension verification:")
print(f"Normalized Maths features dimension: {X_maths_train_normalized.shape}")
print(f"Normalized GNN features dimension: {gnn_output_train_normalized.shape}")
print(f"Normalized Total electrons dimension: {total_electrons_train_normalized.shape}")
print(f"Combined normalized features dimension: {combined_train_features.shape}")

# 显示前3个样本的特征值
print(f"\nNormalized feature values for first 3 samples:")
for i in range(min(3, len(combined_train_features))):
    print(f"Sample {i}: F1_norm={combined_train_features[i,0]:.6f}, "
          f"F2_norm={combined_train_features[i,1]:.6f}, "
          f"F3_norm={combined_train_features[i,2]:.6f}, "
          f"F4={combined_train_features[i,3]:.6f}, "
          f"F5={combined_train_features[i,4]:.6f}, "
          f"F6={combined_train_features[i,5]:.6f}, "
          f"F7={combined_train_features[i,6]:.6f}, "
          f"F8={combined_train_features[i,7]:.6f}, "
          f"F9={combined_train_features[i,8]:.6f}, "
          f"F10={combined_train_features[i,9]:.6f}, "
          f"F11={combined_train_features[i,10]:.6f}, "
          f"F12={combined_train_features[i,11]:.6f}, "
          f"GNN={combined_train_features[i,12]:.6f}, "
          f"Electrons={combined_train_features[i,13]:.6f}")

# 定义训练常量
# 注意：由于目标变量尺度变化，可能需要调整阈值
MAE_THRESHOLD = 0.00002
MSE_THRESHOLD = 0.00000005
CONSECUTIVE_EPOCHS = 5
MAX_EPOCHS = 1000000

print(f"\nTraining configuration:")
print(f"  Target variable: (L-D)/total_electrons")
print(f"  MAE threshold: {MAE_THRESHOLD}")
print(f"  MSE threshold: {MSE_THRESHOLD}")
print(f"  Consecutive epochs: {CONSECUTIVE_EPOCHS}")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Training samples: {len(combined_train_features)}")
print(f"  Test samples: {len(combined_test_features)}")
print(f"  Feature dimension: {combined_train_features.shape[1]}")

# 设置模型检查点
checkpoint = ModelCheckpoint('best_model_Maths_GNN_features_normalized.keras',
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

# 创建Keras模型 - 更新输入维度为14
model = Sequential([
    Dense(64, input_dim=14, activation='relu', kernel_initializer='glorot_uniform'),  # 14个输入特征
    Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
    Dense(1, kernel_initializer='glorot_uniform')
])

# 模型编译
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 创建增强的早停回调
early_stopping = EnhancedEarlyStopping(
    mae_thresh=MAE_THRESHOLD,
    mse_thresh=MSE_THRESHOLD,
    consecutive_epochs=CONSECUTIVE_EPOCHS
)

# 开始训练，使用适当的回调配置
print("\nStarting training with normalized features...")
print("=" * 80)

# 设置TensorFlow随机种子以确保可重复性
import tensorflow as tf
tf.random.set_seed(42)

history = model.fit(
    combined_train_features, y_train,
    epochs=MAX_EPOCHS,
    batch_size=128,
    validation_data=(combined_test_features, y_test),
    callbacks=[checkpoint, early_stopping],
    verbose=0  # 设置为0，使用自定义回调进行输出
)

# 最终评估
print("\n" + "=" * 80)
print("Final Evaluation:")
y_pred = model.predict(combined_test_features)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Test set MSE: {mse:.9f}')
print(f'Test set MAE: {mae:.9f}')

# 保存最终模型用于生产
model.save('Final_model_Maths_GNN_features_normalized.keras')
print("Model saved as 'Final_model_Maths_GNN_features_normalized.keras'")

# 保存GNN模型用于验证
torch.save({
    'model_state_dict': gnn_model.state_dict(),
    'random_seed': 42,
    'model_config': {
        'num_node_features': 3,
        'hidden_dim': 16,
        'output_dim': 1
    }
}, 'gnn_model_weights.pth')
print("GNN model weights saved as 'gnn_model_weights.pth'")

# 保存所有归一化器用于生产
joblib.dump(maths_scaler, 'maths_scaler.pkl')
joblib.dump(gnn_scaler, 'gnn_scaler.pkl')
joblib.dump(electron_scaler, 'electron_scaler.pkl')
print("All scalers saved: 'maths_scaler.pkl', 'gnn_scaler.pkl', 'electron_scaler.pkl'")

# 保存随机种子信息用于可重复性
seed_info = {
    'random_seed': 42,
    'numpy_seed': 42,
    'torch_seed': 42,
    'tensorflow_seed': 42
}
import json
with open('random_seed_info.json', 'w') as f:
    json.dump(seed_info, f)
print("Random seed information saved as 'random_seed_info.json'")

# 输出训练历史摘要
if len(history.history['loss']) > 0:
    final_epoch = len(history.history['loss'])
    final_train_mse = history.history['loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mse = history.history['val_loss'][-1]
    final_val_mae = history.history['val_mae'][-1]

    print(f"\nTraining Summary:")
    print(f"  Total epochs trained: {final_epoch}")
    print(f"  Final training MSE: {final_train_mse:.9f}")
    print(f"  Final training MAE: {final_train_mae:.9f}")
    print(f"  Final validation MSE: {final_val_mse:.9f}")
    print(f"  Final validation MAE: {final_val_mae:.9f}")

# 特征重要性分析（基于第一层权重的简化版本）
print(f"\nFeature Importance Analysis:")
final_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(final_weights), axis=1)

total_importance = np.sum(feature_importance)
relative_importance = (feature_importance / total_importance) * 100

print("Feature importance (based on first layer weights):")
for name, importance, rel_imp in zip(feature_names_normalized, feature_importance, relative_importance):
    print(f"  {name}: {importance:.4f} ({rel_imp:.1f}%)")

# 运行改进的特征重要性分析
improved_importance = improved_feature_importance_analysis(model, feature_names_normalized)

print("\n" + "="*80)
print("COMPARISON: Correlation-based vs Weight-based Importance")
print("="*80)
print("Note: Correlation-based importance shows linear relationship with target")
print("      Weight-based importance shows model's learned feature usage")

print("\n" + "="*80)
print("MODEL SAVING COMPLETE")
print("="*80)
print("Saved files for validation:")
print("  - Final_model_Maths_GNN_features_normalized.keras (Keras model)")
print("  - gnn_model_weights.pth (GNN model weights)")
print("  - maths_scaler.pkl, gnn_scaler.pkl, electron_scaler.pkl (Feature scalers)")
print("  - random_seed_info.json (Random seed information)")
print("\nTraining Configuration:")
print("  - All 14 normalized features participate in joint training")
print("  - Random seeds set for reproducibility")
print("  - GNN model compatible with multi-atomic molecules")
print("  - Features are properly normalized for better training performance")
print("  - Target variable: (L-D)/total_electrons")
