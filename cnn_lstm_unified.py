import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP & UNIFIED DATA LOADING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Dataset.xlsx'

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

# 2. FEATURE ENGINEERING (STATE CENTERING & DERIVATIVES)
print("Engineering cumulative, derivative, and centered features...")

# 1. State Centering (Removes the -30 to +30 random initial position variance)
baseline_df = full_df[full_df['Time_Step'] < 20].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baseline_df.rename(columns={'y_k': 'y_baseline'}, inplace=True)
full_df = pd.merge(full_df, baseline_df, on=['Run_ID', 'Attack_Type'], how='left')

# The pure attack drift
full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']

# 2. Derivatives
full_df['Delta_y'] = full_df.groupby(['Run_ID', 'Attack_Type'])['y_k'].diff().fillna(0)
full_df['Delta_g'] = full_df.groupby(['Run_ID', 'Attack_Type'])['g_k'].diff().fillna(0)

# 3. Cumulative Sum
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(lambda x: x.abs().cumsum())

feature_cols = [
    'y_deviation', 'Delta_y', 'r_k', 'g_k', 'Delta_g', 
    'Mean_g', 'Var_g', 
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

# 3. DYNAMIC GRACE PERIOD RELABELING
print("Applying physical grace period logic...")
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# 4. GROUP-BASED SPLITTING & SCALING 
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

# 5. TEMPORAL SLIDING WINDOW SEQUENCE GENERATION
def create_sequences(df, window_size):
    X, y = [], []
    for _, group in df.groupby(['Run_ID', 'Attack_Type']):
        features, labels = group[feature_cols].values, group['Label_ID'].values
        if len(group) >= window_size:
            for i in range(len(group) - window_size + 1):
                X.append(features[i : i + window_size])
                y.append(labels[i + window_size - 1])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

WINDOW_SIZE = 35
X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
X_val, y_val     = create_sequences(val_df, WINDOW_SIZE)
X_test, y_test   = create_sequences(test_df, WINDOW_SIZE)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

num_classes = 6
class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
safe_counts = np.where(class_counts == 0, 1, class_counts)
weights_tensor = torch.tensor(len(y_train) / (num_classes * safe_counts), dtype=torch.float32)

# 6. NEURAL NETWORK SPECIFICATION (DUAL LOSS)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_output):
        attn_scores = self.attention(lstm_output) 
        attn_weights = torch.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_weights * lstm_output, dim=1) 
        return context_vector, attn_weights

class AttentionConvLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AttentionConvLSTMClassifier, self).__init__()
        
        # --- 1. SPATIAL FEATURE EXTRACTION (CNN) ---
        # Reads the 15 features across the time window to find local correlations
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        
        # --- 2. TEMPORAL SEQUENCE MODELING (LSTM) ---
        # Replaces the GRU. Notice we use nn.LSTM here.
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3,
            bidirectional=True
        )
        
        # --- 3. GLOBAL ATTENTION ---
        # Weighs which time steps in the 35-step window are most important
        self.attention = TemporalAttention(hidden_dim*2)
        
        # --- 4. DEEP FEATURE COMPRESSION & CLASSIFICATION ---
        self.fc1 = nn.Linear(hidden_dim*2, 32)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # CNNs in PyTorch expect [Batch, Channels (Features), Length (Time_Steps)]
        x = x.transpose(1, 2)
        
        # 1. Pass through CNN blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # LSTMs expect [Batch, Length (Time_Steps), Features]
        x = x.transpose(1, 2)
        
        # 2. Pass through LSTM
        # LSTM returns the output, plus a tuple of (hidden_state, cell_state)
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        
        # 3. Apply Attention to the LSTM outputs
        context, attn_weights = self.attention(lstm_out)
        
        # 4. Capture the 32-D deep feature embedding (For Center Loss)
        deep_features = self.relu(self.fc1(context))
        deep_features = self.dropout(deep_features)
        
        # 5. Final Classification (For Cross-Entropy)
        out = self.fc2(deep_features)
        
        # Return BOTH to satisfy your Dual-Loss training loop
        return out, deep_features

class CenterLoss(nn.Module):
    def __init__(self, num_classes=6, feat_dim=32):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss

# Initialize Model
model = AttentionConvLSTMClassifier(input_dim=len(feature_cols), hidden_dim=96, num_classes=num_classes)

# Initialize Dual Loss Components
criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_center = CenterLoss(num_classes=num_classes, feat_dim=32)

# Unify optimizers
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion_center.parameters()), 
    lr=0.001, 
    weight_decay=1e-4
)

lambda_c = 0.005

# 7. TRAINING LOOP WITH DUAL LOSS
epochs = 15
best_val_loss = float('inf')
best_model_state = None

print(f"\nTraining Attention-Augmented network (Static lambda_c = {lambda_c}) for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Extract both outputs
        outputs, features = model(batch_X)
        
        # Compute losses
        loss_ce = criterion_ce(outputs, batch_y)
        loss_center = criterion_center(features, batch_y)
        loss = loss_ce + (lambda_c * loss_center)
        
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
            outputs, features = model(batch_X)
            
            loss_ce = criterion_ce(outputs, batch_y)
            loss_center = criterion_center(features, batch_y)
            loss = loss_ce + (lambda_c * loss_center)
            
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    current_val_loss = val_loss / val_total
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {100*train_correct/train_total:.2f}% | "
          f"Val Loss: {current_val_loss:.4f} | Val Acc: {100*val_correct/val_total:.2f}%")
    
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_state = model.state_dict().copy()

# 8. COMPREHENSIVE PERFORMANCE VERIFICATION
print("\nExecuting final evaluation on Test Set...")

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("-> Successfully loaded best model weights based on Validation Loss.")

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs, _ = model(batch_X)  # Deep features not needed for final inference
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=target_names, columns=target_names))