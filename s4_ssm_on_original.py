import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# SETUP & DATA LOADING
print("Loading datasets")
files = [
    'TTC_Replay_Dataset (2).xlsx',
    'TTC_FiniteCovert_Dataset (2).xlsx',
    'TTC_FDI_Dataset (2).xlsx',
    'TTC_BiasRamp_Dataset (2).xlsx',
    'TTC_OptimizedZDA_Dataset (2).xlsx'
]
for f in files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")
dfs = [pd.read_excel(f) for f in files]
full_df = pd.concat(dfs, ignore_index=True)
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()
label_map = {
    'Nominal': 0,
    'Replay Attack': 1,
    'Covert Attack': 2,
    'FDI Attack': 3,
    'Bias Attack': 4,
    'ZD Attack': 5
}
full_df['Label_ID'] = full_df['Attack_Type'].map(label_map)
dropped_rows = full_df['Label_ID'].isna().sum()
if dropped_rows > 0:
    print(f"WARNING: {dropped_rows} rows had unmapped labels and will be dropped.")
    print("Unmapped labels found:", full_df[full_df['Label_ID'].isna()]['Attack_Type'].unique())
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

# FEATURE ENGINEERING (CUSUM)
print("Engineering CUSUM feature")
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(
    lambda x: x.abs().cumsum()
)

print("Engineering advanced autocorrelation features")
lag = 10
rolling_window = 30

def compute_lag_acf(group, window, lag_val):
    shifted_r = group['r_k'].shift(lag_val)
    return group['r_k'].rolling(window=window).corr(shifted_r)

full_df['Lag10_ACF_r'] = full_df.groupby('Run_ID').apply(
    lambda x: compute_lag_acf(x, rolling_window, lag)
).reset_index(level=0, drop=True)
full_df['Lag10_ACF_r'] = full_df['Lag10_ACF_r'].fillna(0)

feature_cols = ['y_k', 'r_k', 'g_k', 'Mean_g', 'Var_g', 'Lag3_ACF_r', 'CUSUM_r', 'Lag10_ACF_r']

# AMBIGUITY DROP
print("Applying physical logic corrections and dropping ambiguous ramp-up phases...")
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
drop_mask = (
    (full_df['Time_Step'] >= 20) &
    (full_df['Time_Step'] < 40) &
    (full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack']))
)
full_df = full_df[~drop_mask].reset_index(drop=True)

# SPLITTING
train_runs = list(range(1, 36))  
val_runs = list(range(36, 43))   
test_runs = list(range(43, 51))
train_df = full_df[full_df['Run_ID'].isin(train_runs)].copy()
val_df   = full_df[full_df['Run_ID'].isin(val_runs)].copy()
test_df  = full_df[full_df['Run_ID'].isin(test_runs)].copy()
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

# SEQUENCE BUILDER
# Use sliding windows so the training setup matches the transformer and ConvGRU scripts.
def create_sequences(df, window_size):
    X, y = [], []
    grouped = df.groupby(['Run_ID', 'Attack_Type'])
    for _, group in grouped:
        features = group[feature_cols].values
        labels = group['Label_ID'].values
        if len(group) >= window_size:
            for i in range(len(group) - window_size + 1):
                X.append(features[i:i + window_size])
                y.append(labels[i + window_size - 1])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

print("Building temporal sequences...")
WINDOW_SIZE = 30
X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
X_val, y_val     = create_sequences(val_df, WINDOW_SIZE)
X_test, y_test   = create_sequences(test_df, WINDOW_SIZE)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

# CLASS WEIGHT CALCULATION
# Collect all training labels across all windows
num_classes   = 6
class_counts  = np.bincount(y_train.numpy(), minlength=num_classes)
print(f"Training Data Class Distribution (Classes 0 to 5): {class_counts}")
total_samples = len(y_train)
safe_counts   = np.where(class_counts == 0, 1, class_counts)
class_weights = total_samples / (num_classes * safe_counts)
class_weights[class_counts == 0] = 0.0
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# STATE SPACE MODEL ARCHITECTURE (current)
# S4D: HiPPO-like complex diagonal A with complex B/C and real D.
#   - Discretised via Zero-Order Hold in complex domain and converted
#     into a real convolution kernel K = sum_t C * A^t * B_bar, then
#     convolved with the input using FFT for O(L log L) evaluation.
# SSMBlock:
#   - Pre-LayerNorm -> S4D (FFT conv) -> elementwise sigmoid gate -> linear proj
#   - Residual connection added to the projected S4D output.
# SSMClassifier:
#   - `input_proj` projects features to `d_model` → stack of SSMBlocks → LayerNorm
#   - Temporal mean pooling across the window produces `deep_features` (size d_model)
#   - `fc` maps pooled `deep_features` → logits. Forward returns `(logits, deep_features)`.

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.h = d_model
        self.n = d_state

        # 1. Complex HiPPO-LegS Initialization
        # The real part is forced to -0.5 for stability.
        # The imaginary part contains the structured frequencies.
        real = torch.ones(self.n) * -0.5
        imag = math.pi * torch.arange(self.n)
        
        # We store A as a complex parameter
        self.A = nn.Parameter(torch.complex(real, imag)) 

        # B and C are also initialized as complex numbers
        self.B = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        
        # D is the standard real skip connection
        self.D = nn.Parameter(torch.randn(self.h))

        # Learnable step size initialization (log scale)
        self.log_dt = nn.Parameter(torch.randn(self.h) - math.log(10))

    def forward(self, x):
        # x is [Batch, Time, Channels]. We transpose for the 1D convolution math
        u = x.transpose(1, 2) 
        L = u.size(-1)

        # 2. Continuous to Discrete (Zero-Order Hold in Complex Space)
        dt = torch.exp(self.log_dt).unsqueeze(-1) # [H, 1]
        A = self.A.unsqueeze(0) # [1, N]
        B = self.B # [H, N]
        C = self.C # [H, N]

        A_dt = A * dt
        # Discrete B matrix
        B_bar = (torch.exp(A_dt) - 1.0) / A * B

        # 3. Construct the Convolutional Kernel (K)
        # Instead of a for-loop, we build the entire filter kernel at once
        step = torch.arange(L, device=u.device).unsqueeze(0).unsqueeze(-1) # [1, L, 1]
        A_powers = torch.exp(A_dt.unsqueeze(1) * step) # [H, L, N]

        # Combine C, A^t, and B into the filter kernel K. 
        # We take the .real part because the physical output must be real numbers.
        K = (C.unsqueeze(1) * A_powers * B_bar.unsqueeze(1)).sum(-1).real

        # 4. The FFT Convolution
        # This replaces the chronological loop. It computes the entire sequence instantly.
        k_f = torch.fft.rfft(K, n=2*L)
        u_f = torch.fft.rfft(u, n=2*L)
        y = torch.fft.irfft(k_f * u_f, n=2*L)[..., :L]

        # 5. Add the D skip connection
        y = y + u * self.D.unsqueeze(-1)

        # Return to [Batch, Time, Channels]
        return y.transpose(1, 2)
class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=64): # Increased d_state to 64 for HiPPO capacity
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the S4D layer
        self.ssm  = S4D(d_model, d_state)
        
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        residual = x
        x        = self.norm(x)
        
        # Pass through S4D (FFT Convolution)
        ssm_out  = self.ssm(x)
        
        gate     = torch.sigmoid(self.gate(x))
        x        = self.proj(ssm_out * gate)
        return x + residual
class SSMClassifier(nn.Module):
    def __init__(self, input_dim, d_model, d_state, num_layers, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers     = nn.ModuleList([
            SSMBlock(d_model, d_state) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, num_classes)
    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(0)

        # x: [B, T, input_dim]
        x = self.input_proj(x)      # [B, T, d_model]
        for layer in self.layers:
            x = layer(x)            # [B, T, d_model]
        x = self.norm(x)
        deep_features = x.mean(dim=1)  # [B, d_model]
        out = self.fc(deep_features)   # [B, num_classes]
        return out, deep_features
model     = SSMClassifier(input_dim=len(feature_cols), d_model=64,
                          d_state=32, num_layers=2, num_classes=num_classes)

class CenterLoss(nn.Module):
    def __init__(self, num_classes=6, feat_dim=64):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss

criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_center = CenterLoss(num_classes=num_classes, feat_dim=64)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion_center.parameters()),
    lr=0.001
)

lambda_c = 0.01

# TRAINING LOOP
# One window per forward pass, matching the other classifiers.
epochs = 25
print("\nStarting Training")
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs, features = model(batch_X)
        loss_ce = criterion_ce(outputs, batch_y)
        loss_center = criterion_center(features, batch_y)
        loss = loss_ce + (lambda_c * loss_center)

        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * batch_X.size(0)
        _, predicted   = torch.max(outputs, 1)
        total_train   += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs, features = model(batch_X)
            loss_ce = criterion_ce(outputs, batch_y)
            loss_center = criterion_center(features, batch_y)
            loss = loss_ce + (lambda_c * loss_center)
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val   += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/total_train:.4f} | Train Acc: {100*correct_train/total_train:.2f}% | "
          f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {100*correct_val/total_val:.2f}%")

# FINAL TESTING & EVALUATION
print("\nRunning final evaluation on Test Set (Runs 43-50)")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs, _ = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())
active_labels = [k for k, v in label_map.items() if v in np.unique(all_labels)]
target_names  = []
for idx in range(num_classes):
    for name, val in label_map.items():
        if val == idx and name not in target_names and name in active_labels:
            target_names.append(name)
            break
print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(
    confusion_matrix(all_labels, all_preds),
    index=target_names, columns=target_names
))