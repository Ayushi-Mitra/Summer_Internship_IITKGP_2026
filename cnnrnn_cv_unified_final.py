import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
import gc

# 1. SETUP & UNIFIED DATA LOADING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Final_Dataset.xlsx' # Switch to .csv if needed for memory

if not os.path.exists(filename):
    raise FileNotFoundError(f"Missing unified dataset: {filename}")

full_df = pd.read_excel(filename)
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()

label_map = {
    'Nominal': 0, 'Replay Attack': 1, 'Covert Attack': 2, 
    'FDI Attack': 3, 'Bias Attack': 4, 'ZD Attack': 5
}

full_df['Label_ID'] = full_df['Attack_Type'].map(label_map)
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

# 2. MEMORY-OPTIMIZED FEATURE ENGINEERING
print("Engineering features (Memory Optimized Mode)...")

baseline_mask = full_df['Time_Step'] < 20
baselines = full_df[baseline_mask].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baselines.rename(columns={'y_k': 'y_baseline'}, inplace=True)

full_df = pd.merge(full_df, baselines, on=['Run_ID', 'Attack_Type'], how='left')
full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']

print("   -> Calculating vectorized slopes...")
full_df['Delta_y'] = full_df['y_deviation'].diff().fillna(0)
full_df['Delta_g'] = full_df['g_k'].diff().fillna(0)
full_df['Slope_10'] = full_df['y_deviation'].diff(10).fillna(0)
full_df['Slope_20'] = full_df['y_deviation'].diff(20).fillna(0)

print("   -> Calculating localized CUSUM...")
def fast_cusum(series):
    return series.abs().cumsum()

full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(fast_cusum)

del baselines
gc.collect()

feature_cols = [
    'y_deviation', 'Delta_y', 'Delta_g', 'Slope_10', 'Slope_20', 
    'r_k', 'g_k', 'Mean_g', 'Var_g', 
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

# 3. DYNAMIC GRACE PERIOD
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# 4. MODEL ARCHITECTURE
class GlobalConvRNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=6, rnn_type='GRU'):
        super(GlobalConvRNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
            
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        
        rnn_out, _ = self.rnn(x)
        final_state = rnn_out[:, -1, :]
        
        out = self.relu(self.fc1(final_state))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def create_full_sequences(df):
    X, y = [], []
    for _, group in df.groupby(['Run_ID', 'Attack_Type']):
        features = group[feature_cols].values
        labels = group['Label_ID'].values
        final_label = labels[-1] 
        X.append(features)
        y.append(final_label)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 5. 10-FOLD CROSS VALIDATION
print("\nInitiating 10-Fold Cross Validation...")

unique_runs = full_df['Run_ID'].unique()
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Metrics tracking
fold_metrics = {'precision': [], 'recall': [], 'f1': []}
epochs = 30
num_classes = 6
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

for fold, (train_idx, test_idx) in enumerate(kf.split(unique_runs)):
    print(f"\n{'-'*30}")
    print(f"FOLD {fold + 1}/10")
    print(f"{'-'*30}")
    
    # 1. Split Data by Run_ID
    train_runs = unique_runs[train_idx]
    test_runs = unique_runs[test_idx]
    
    train_df = full_df[full_df['Run_ID'].isin(train_runs)].copy()
    test_df = full_df[full_df['Run_ID'].isin(test_runs)].copy()
    
    # 2. Scale Data (Fit ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    
    # 3. Create Sequences
    X_train, y_train = create_full_sequences(train_df)
    X_test, y_test   = create_full_sequences(test_df)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    # 4. Calculate Class Weights for this fold
    class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
    safe_counts = np.where(class_counts == 0, 1, class_counts)
    weights_tensor = torch.tensor(len(y_train) / (num_classes * safe_counts), dtype=torch.float32).to(device)
    
    # 5. Initialize fresh model and optimizer
    model = GlobalConvRNNClassifier(input_dim=len(feature_cols), hidden_dim=128, num_classes=num_classes, rnn_type='GRU').to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # 6. Training Loop for the fold
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        # Quick validation on test set (for early stopping/best model selection within fold)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                
        avg_test_loss = test_loss / len(X_test)
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            
    # 7. Evaluate the best model on this fold's test set
    model.load_state_dict(best_model_state)
    model.eval()
    
    fold_preds, fold_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            fold_preds.extend(predicted.cpu().numpy())
            fold_labels.extend(batch_y.cpu().numpy())
            
    # 8. Calculate and store Macro Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(fold_labels, fold_preds, average='macro', zero_division=0)
    
    fold_metrics['precision'].append(precision)
    fold_metrics['recall'].append(recall)
    fold_metrics['f1'].append(f1)
    
    print(f"Fold {fold+1} Results: Macro Precision: {precision:.4f} | Macro Recall: {recall:.4f} | Macro F1: {f1:.4f}")
    
    # Cleanup memory before next fold
    del train_df, test_df, X_train, y_train, X_test, y_test, model, optimizer
    gc.collect()

# 6. FINAL CROSS-VALIDATION SUMMARY
print("\n" + "="*40)
print("10-FOLD CROSS-VALIDATION RESULTS (MACRO AVERAGE)")
print("="*40)

mean_precision = np.mean(fold_metrics['precision'])
std_precision = np.std(fold_metrics['precision'])

mean_recall = np.mean(fold_metrics['recall'])
std_recall = np.std(fold_metrics['recall'])

mean_f1 = np.mean(fold_metrics['f1'])
std_f1 = np.std(fold_metrics['f1'])

print(f"Macro Precision : {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Macro Recall    : {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Macro F1-Score  : {mean_f1:.4f} ± {std_f1:.4f}")