import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SETUP & UNIFIED DATA LOADING
# ==========================================
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

# ==========================================
# 2. FEATURE ENGINEERING (STATE CENTERING & DERIVATIVES)
# ==========================================
print("Engineering cumulative, derivative, and centered features...")

baseline_df = full_df[full_df['Time_Step'] < 20].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baseline_df.rename(columns={'y_k': 'y_baseline'}, inplace=True)
full_df = pd.merge(full_df, baseline_df, on=['Run_ID', 'Attack_Type'], how='left')

full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']
full_df['Delta_y'] = full_df.groupby(['Run_ID', 'Attack_Type'])['y_k'].diff().fillna(0)
full_df['Delta_g'] = full_df.groupby(['Run_ID', 'Attack_Type'])['g_k'].diff().fillna(0)
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(lambda x: x.abs().cumsum())

feature_cols = [
    'y_deviation', 'Delta_y', 'r_k', 'g_k', 'Delta_g', 
    'Mean_g', 'Var_g', 
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

# ==========================================
# 3. DYNAMIC GRACE PERIOD RELABELING
# ==========================================
print("Applying physical grace period logic...")
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# ==========================================
# 4. GROUP-BASED SPLITTING & SCALING 
# ==========================================
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

# ==========================================
# 5. TEMPORAL SLIDING WINDOW SEQUENCE GENERATION
# ==========================================
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

# ==========================================
# 6. NEURAL NETWORK SPECIFICATION: S4D (STATE SPACE MODEL)
# ==========================================
class S4D(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.h = d_model
        self.n = d_state

        # Complex HiPPO-LegS Initialization
        real = torch.ones(self.n) * -0.5
        imag = math.pi * torch.arange(self.n)
        
        self.A = nn.Parameter(torch.complex(real, imag)) 
        self.B = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        self.D = nn.Parameter(torch.randn(self.h))

        # Learnable step size initialization (log scale)
        self.log_dt = nn.Parameter(torch.randn(self.h) - math.log(10))

    def forward(self, x):
        u = x.transpose(1, 2) 
        L = u.size(-1)

        # Continuous to Discrete
        dt = torch.exp(self.log_dt).unsqueeze(-1) # [H, 1]
        A = self.A.unsqueeze(0) # [1, N]
        B = self.B # [H, N]
        C = self.C # [H, N]

        A_dt = A * dt
        B_bar = (torch.exp(A_dt) - 1.0) / A * B

        # Construct Filter Kernel via FFT
        step = torch.arange(L, device=u.device).unsqueeze(0).unsqueeze(-1)
        A_powers = torch.exp(A_dt.unsqueeze(1) * step) 
        K = (C.unsqueeze(1) * A_powers * B_bar.unsqueeze(1)).sum(-1).real

        # FFT Convolution
        k_f = torch.fft.rfft(K, n=2*L)
        u_f = torch.fft.rfft(u, n=2*L)
        y = torch.fft.irfft(k_f * u_f, n=2*L)[..., :L]

        y = y + u * self.D.unsqueeze(-1)
        return y.transpose(1, 2)

class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = S4D(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        residual = x
        x        = self.norm(x)
        ssm_out  = self.ssm(x)
        gate     = torch.sigmoid(self.gate(x))
        x        = self.proj(ssm_out * gate)
        return x + residual

class SSMClassifier(nn.Module):
    def __init__(self, input_dim, d_model, d_state, num_layers, num_classes):
        super().__init__()
        
        # Spatial Projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # HiPPO State Space Tracking
        self.layers = nn.ModuleList([SSMBlock(d_model, d_state) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
        # Deep Feature Extraction (Matched to GRU/Transformer bottleneck)
        self.fc_bottleneck = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        
        # Final Classifier
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.input_proj(x)      
        for layer in self.layers:
            x = layer(x)            
        x = self.norm(x)
        
        # Temporal Pooling
        pooled_context = x.mean(dim=1)  
        
        # Capture the 32-D deep feature embedding for Center Loss
        deep_features = self.relu(self.fc_bottleneck(pooled_context))
        deep_features = self.dropout(deep_features)
        
        # Final logits
        out = self.fc(deep_features)   
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

# Initialize Model (Matching capacity parameters to the GRU/Transformer)
model = SSMClassifier(
    input_dim=len(feature_cols), 
    d_model=64, 
    d_state=64, 
    num_layers=2, 
    num_classes=num_classes
)

# Initialize Dual Loss Components
criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_center = CenterLoss(num_classes=num_classes, feat_dim=32)

# Unify optimizers
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion_center.parameters()), 
    lr=0.001, 
    weight_decay=1e-4
)

# The optimal static Lambda established by the ablation study
lambda_c = 0.01

# ==========================================
# 7. TRAINING LOOP WITH DUAL LOSS
# ==========================================
epochs = 15
best_val_loss = float('inf')
best_model_state = None

print(f"\nTraining S4D State-Space network (Static lambda_c = {lambda_c}) for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs, features = model(batch_X)
        
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

# ==========================================
# 8. COMPREHENSIVE PERFORMANCE VERIFICATION
# ==========================================
print("\nExecuting final evaluation on Test Set...")

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("-> Successfully loaded best model weights based on Validation Loss.")

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs, _ = model(batch_X) 
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=target_names, columns=target_names))