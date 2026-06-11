import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import gc

# 1. SETUP & UNIFIED DATA LOADING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Dataset_New.csv' # Switch to .csv if needed for memory

if not os.path.exists(filename):
    raise FileNotFoundError(f"Missing unified dataset: {filename}")

full_df = pd.read_csv(filename)
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
    'r_k', 'g_k', 'Mean_g', 'Var_g', 'K_Attack',
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

# 3. DYNAMIC GRACE PERIOD
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# 4. DATA SPLITTING & SCALING
total_runs = int(full_df['Run_ID'].max())
train_end, val_end = int(total_runs * 0.70), int(total_runs * 0.85)

train_runs = list(range(1, train_end + 1))  
val_runs = list(range(train_end + 1, val_end + 1))   
test_runs = list(range(val_end + 1, total_runs + 1))  

train_df = full_df[full_df['Run_ID'].isin(train_runs)].copy()
val_df = full_df[full_df['Run_ID'].isin(val_runs)].copy()
test_df = full_df[full_df['Run_ID'].isin(test_runs)].copy()

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

# 5. FULL TRAJECTORY SEQUENCE GENERATOR
def create_full_sequences(df):
    X, y = [], []
    for _, group in df.groupby(['Run_ID', 'Attack_Type']):
        features = group[feature_cols].values
        labels = group['Label_ID'].values
        final_label = labels[-1] 
        X.append(features)
        y.append(final_label)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

print("Building Full Trajectory Sequences...")
X_train, y_train = create_full_sequences(train_df)
X_val, y_val     = create_full_sequences(val_df)
X_test, y_test   = create_full_sequences(test_df)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

num_classes = 6
class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
safe_counts = np.where(class_counts == 0, 1, class_counts)
weights_tensor = torch.tensor(len(y_train) / (num_classes * safe_counts), dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 6. BASELINE MODEL: CONV1D + LSTM/GRU
class GlobalConvRNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=6, rnn_type='GRU'):
        super(GlobalConvRNNClassifier, self).__init__()
        
        # Spatial Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Temporal Processing
        # You can swap between LSTM and GRU here
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
            
        # Classification Head
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # CNNs require (Batch, Channels, Length)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        
        # RNN Processing
        rnn_out, _ = self.rnn(x)
        
        # To classify the whole trajectory, we grab the final hidden state of the sequence
        # This is where the memory fade happens for long sequences!
        final_state = rnn_out[:, -1, :]
        
        # Final classification
        out = self.relu(self.fc1(final_state))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Initialize with LSTM by default (change to 'GRU' if desired)
model = GlobalConvRNNClassifier(input_dim=len(feature_cols), hidden_dim=128, num_classes=num_classes, rnn_type='GRU').to(device)
weights_tensor = weights_tensor.to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# 7. TRAINING LOOP
epochs = 30 
best_val_loss = float('inf')
best_model_state = None

print(f"\nTraining Baseline Conv1D+GRU on {len(X_train)} full sequences...")

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
        
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    current_val_loss = val_loss / val_total
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {100*train_correct/train_total:.2f}% | "
              f"Val Loss: {current_val_loss:.4f} | Val Acc: {100*val_correct/val_total:.2f}%")
    
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_state = model.state_dict().copy()

# 8. PERFORMANCE VERIFICATION
print("\nExecuting final evaluation on Test Set...")

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("-> Successfully loaded best model weights.")

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== BASELINE CLASSIFICATION REPORT (CNN+GRU) ===")
print(classification_report(all_labels, all_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=target_names, columns=target_names))