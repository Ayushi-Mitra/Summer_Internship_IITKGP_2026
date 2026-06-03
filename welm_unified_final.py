import os
import pandas as pd
import numpy as np
from scipy.linalg import inv
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import gc

# 1. SETUP, FEATURE ENGINEERING & SCALING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Final_Dataset.xlsx'
full_df = pd.read_excel(filename)
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()

label_map = {'Nominal': 0, 'Replay Attack': 1, 'Covert Attack': 2, 
             'FDI Attack': 3, 'Bias Attack': 4, 'ZD Attack': 5}
full_df['Label_ID'] = full_df['Attack_Type'].map(label_map)
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

print("Engineering features...")
baseline_mask = full_df['Time_Step'] < 20
baselines = full_df[baseline_mask].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baselines.rename(columns={'y_k': 'y_baseline'}, inplace=True)

full_df = pd.merge(full_df, baselines, on=['Run_ID', 'Attack_Type'], how='left')
full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']

full_df['Delta_y'] = full_df['y_deviation'].diff().fillna(0)
full_df['Delta_g'] = full_df['g_k'].diff().fillna(0)
full_df['Slope_10'] = full_df['y_deviation'].diff(10).fillna(0)
full_df['Slope_20'] = full_df['y_deviation'].diff(20).fillna(0)

def fast_cusum(series): return series.abs().cumsum()
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(fast_cusum)

del baselines
gc.collect()

feature_cols = [
    'y_deviation', 'Delta_y', 'Delta_g', 'Slope_10', 'Slope_20', 
    'r_k', 'g_k', 'Mean_g', 'Var_g', 
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

total_runs = int(full_df['Run_ID'].max())
train_end, val_end = int(total_runs * 0.70), int(total_runs * 0.85)

train_df = full_df[full_df['Run_ID'].isin(list(range(1, train_end + 1)))].copy()
val_df = full_df[full_df['Run_ID'].isin(list(range(train_end + 1, val_end + 1)))].copy()
test_df = full_df[full_df['Run_ID'].isin(list(range(val_end + 1, total_runs + 1)))].copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
y_train = train_df['Label_ID'].values

X_val = scaler.transform(val_df[feature_cols])
y_val = val_df['Label_ID'].values

X_test = scaler.transform(test_df[feature_cols])
y_test = test_df['Label_ID'].values

# 2. CUSTOM MEMORY-OPTIMIZED WELM CLASSIFIER
class WELMClassifier:
    def __init__(self, hidden_nodes=100, C=1.0):
        self.hidden_nodes = hidden_nodes
        self.C = C  # Regularization parameter
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.lb = LabelBinarizer()
        
    def relu(self, x):
        return np.maximum(0, x)
        
    def fit(self, X, y):
        # 1. One-hot encode target labels
        Y = self.lb.fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack((1 - Y, Y))
            
        n_samples, n_features = X.shape
        
        # 2. Calculate Class Weights (Inverse Frequency)
        unique_classes, counts = np.unique(y, return_counts=True)
        class_weight_dict = {c: 1.0 / count for c, count in zip(unique_classes, counts)}
        sample_weights = np.array([class_weight_dict[label] for label in y])
        
        # 3. Randomly Initialize Hidden Nodes
        np.random.seed(42)
        self.input_weights = np.random.normal(size=(n_features, self.hidden_nodes))
        self.biases = np.random.normal(size=(self.hidden_nodes,))
        
        # 4. Calculate Hidden Layer Matrix (H)
        H = self.relu(np.dot(X, self.input_weights) + self.biases)
        
        # 5. Calculate Output Weights (Beta) - MEMORY OPTIMIZED
        # Mathematically equivalent to H.T @ W @ H without the N x N matrix crash
        HW = H * sample_weights[:, np.newaxis] 
        HT_W_H = np.dot(H.T, HW)
        I = np.eye(self.hidden_nodes)
        
        HT_W_Y = np.dot(HW.T, Y)
        
        # Beta = (I/C + H^T * W * H)^-1 * H^T * W * Y
        self.output_weights = np.dot(inv(I / self.C + HT_W_H), HT_W_Y)
        
    def predict(self, X):
        H = self.relu(np.dot(X, self.input_weights) + self.biases)
        Y_pred = np.dot(H, self.output_weights)
        return self.lb.classes_[np.argmax(Y_pred, axis=1)]
    

# 3. PAPER METHODOLOGY: WELM TRAINING & TUNING
print("\nTraining WELM (Testing hidden nodes from paper: 50, 100, 200, 300)...")
hidden_nodes_list = [50, 100, 200, 300]
best_nodes = None
best_val_acc = 0
best_model = None

for nodes in hidden_nodes_list:
    print(f"-> Training WELM with hidden_nodes={nodes}...")
    welm = WELMClassifier(hidden_nodes=nodes, C=1.0)
    welm.fit(X_train, y_train)
    
    val_preds = welm.predict(X_val)
    val_acc = np.mean(val_preds == y_val)
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_nodes = nodes
        best_model = welm

print(f"\nBest model selected with hidden_nodes={best_nodes}")

# 4. FINAL EVALUATION
print("\nExecuting final evaluation on Test Set...")
test_preds = best_model.predict(X_test)
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== BASELINE CLASSIFICATION REPORT (WELM) ===")
print(classification_report(y_test, test_preds, target_names=target_names))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, test_preds), index=target_names, columns=target_names))