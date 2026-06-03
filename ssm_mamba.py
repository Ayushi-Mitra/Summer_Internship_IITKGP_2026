import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
val_runs   = list(range(36, 43))
test_runs  = list(range(43, 51))
train_df = full_df[full_df['Run_ID'].isin(train_runs)].copy()
val_df   = full_df[full_df['Run_ID'].isin(val_runs)].copy()
test_df  = full_df[full_df['Run_ID'].isin(test_runs)].copy()
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

# SEQUENCE BUILDER
def create_sequences(df, window_size):
    X, y = [], []
    grouped = df.groupby(['Run_ID', 'Attack_Type'])
    for _, group in grouped:
        features = group[feature_cols].values
        labels   = group['Label_ID'].values
        if len(group) >= window_size:
            for i in range(len(group) - window_size + 1):
                X.append(features[i:i + window_size])
                y.append(labels[i + window_size - 1])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

print("Building temporal sequences...")
WINDOW_SIZE = 30
X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
X_val,   y_val   = create_sequences(val_df,   WINDOW_SIZE)
X_test,  y_test  = create_sequences(test_df,  WINDOW_SIZE)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=256, shuffle=False)

# CLASS WEIGHT CALCULATION
num_classes  = 6
class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
print(f"Training Data Class Distribution (Classes 0 to 5): {class_counts}")
total_samples = len(y_train)
safe_counts   = np.where(class_counts == 0, 1, class_counts)
class_weights = total_samples / (num_classes * safe_counts)
class_weights[class_counts == 0] = 0.0
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# MAMBA ARCHITECTURE
#
# Key difference from DiagonalSSM (S4):
#   Old: B, C, dt are fixed learnable parameters (input-independent)
#   Mamba (S6): B, C, dt are projected from the current input x_t
#               → the model selectively decides what to retain per token
#
# MambaSSM  – selective SSM core (input-dependent B, C, dt)
# MambaBlock – full Mamba block:
#     input → norm → expand → conv1d → selective SSM → SiLU gate → project → residual
# MambaClassifier – stack of MambaBlocks → mean pool → classify
# ─────────────────────────────────────────────────────────────────────────────

class MambaSSM(nn.Module):
    """
    Selective State Space (S6) core.

    Parameters fixed per layer (input-independent):
        A  – [d_inner, d_state]   (log-parameterised, always negative → stable)

    Parameters projected from input at each step (input-dependent):
        B  – [B, T, d_state]
        C  – [B, T, d_state]
        dt – [B, T, d_inner]      (softplus-activated, always positive)

    Recurrence (ZOH discretisation):
        A_bar_t = exp(dt_t * A)
        B_bar_t = dt_t * B_t           (simplified ZOH for diagonal A)
        h_t     = A_bar_t * h_{t-1} + B_bar_t * x_t   (element-wise, diagonal A)
        y_t     = sum_over_state(C_t * h_t)            + D * x_t
    """
    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = dt_rank or max(1, d_inner // 16)

        # Fixed (non-selective) A
        self.log_A = nn.Parameter(torch.randn(d_inner, d_state) * 0.5 - 1.0)
        # Skip connection scale
        self.D     = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections: x → (dt, B, C)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialise dt_proj bias so softplus(bias) ≈ 0.1 (common Mamba init)
        nn.init.constant_(self.dt_proj.bias, -2.0)

    def forward(self, x):
        # x: [B, T, d_inner]
        B, T, d = x.shape
        A = -torch.exp(self.log_A)                     # [d_inner, d_state], always negative

        # Project x to get input-dependent dt, B_proj, C_proj
        xz  = self.x_proj(x)                           # [B, T, dt_rank + 2*d_state]
        dt_rank = self.dt_proj.in_features
        dt_raw, B_proj, C_proj = xz.split(
            [dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))           # [B, T, d_inner], always > 0

        # Run selective recurrence step-by-step
        h   = torch.zeros(B, d, self.d_state, device=x.device)
        ys  = []
        for t in range(T):
            dt_t  = dt[:, t, :].unsqueeze(-1)          # [B, d_inner, 1]
            B_t   = B_proj[:, t, :]                    # [B, d_state]
            C_t   = C_proj[:, t, :]                    # [B, d_state]
            x_t   = x[:, t, :]                         # [B, d_inner]

            # Discretise A (ZOH) and B (Euler / simplified ZOH)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))   # [B, d_inner, d_state]
            B_bar = dt_t * B_t.unsqueeze(1)            # [B, d_inner, d_state]  (broadcast)

            # State update
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # [B, d_inner, d_state]

            # Output
            y = (h * C_t.unsqueeze(1)).sum(-1)         # [B, d_inner]
            y = y + self.D * x_t
            ys.append(y)

        return torch.stack(ys, dim=1)                  # [B, T, d_inner]


class MambaBlock(nn.Module):
    """
    Full Mamba residual block following the original paper layout:

        x  ──► LayerNorm ──► Linear(expand) ──► conv1d ──► SiLU ──► MambaSSM ──► ×gate ──► Linear(contract) ──► + residual
                                  └─────────────────────────────────────────────► SiLU ──► gate ┘

    expand=2 doubles the inner dimension (standard Mamba ratio).
    conv1d (kernel 4, causal via left-padding) provides local mixing before the SSM.
    """
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm    = nn.LayerNorm(d_model)

        # Expand to 2*d_inner: one half for SSM branch, one for gate branch
        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv1d
        self.conv1d   = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = conv_kernel,
            groups       = self.d_inner,    # depthwise
            padding      = conv_kernel - 1  # left-pad for causality → trim right
        )
        self.conv_trim = conv_kernel - 1    # amount to trim from right after conv

        self.ssm      = MambaSSM(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, d_model]
        residual = x
        x = self.norm(x)

        # Split into SSM branch (z1) and gate branch (z2)
        z = self.in_proj(x)                             # [B, T, 2*d_inner]
        z1, z2 = z.chunk(2, dim=-1)                     # each [B, T, d_inner]

        # Causal conv on SSM branch: [B, T, d_inner] → [B, d_inner, T] → conv → trim → [B, T, d_inner]
        z1c = self.conv1d(z1.transpose(1, 2))           # [B, d_inner, T + trim]
        if self.conv_trim > 0:
            z1c = z1c[:, :, :-self.conv_trim]           # remove right-side padding
        z1  = F.silu(z1c.transpose(1, 2))               # [B, T, d_inner]

        # Selective SSM
        ssm_out = self.ssm(z1)                          # [B, T, d_inner]

        # Gated output
        out = ssm_out * F.silu(z2)                      # [B, T, d_inner]
        return self.out_proj(out) + residual            # [B, T, d_model]


class MambaClassifier(nn.Module):
    """
    Stack of MambaBlocks → mean-pool over time → linear classifier.
    Returns (logits, deep_features) to stay compatible with the CenterLoss training loop.
    """
    def __init__(self, input_dim, d_model, d_state, num_layers, num_classes, expand=2, conv_kernel=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers     = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, conv_kernel) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.input_proj(x)          # [B, T, d_model]
        for layer in self.layers:
            x = layer(x)               # [B, T, d_model]
        x = self.norm(x)
        deep_features = x.mean(dim=1)  # [B, d_model]
        out = self.fc(deep_features)   # [B, num_classes]
        return out, deep_features


model = MambaClassifier(
    input_dim  = len(feature_cols),
    d_model    = 64,
    d_state    = 16,
    num_layers = 2,
    num_classes= num_classes,
    expand     = 2,
    conv_kernel= 4
)

# CENTER LOSS (unchanged)
class CenterLoss(nn.Module):
    def __init__(self, num_classes=6, feat_dim=64):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        return (features - centers_batch).pow(2).sum() / 2.0 / features.size(0)

criterion_ce     = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_center = CenterLoss(num_classes=num_classes, feat_dim=64)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion_center.parameters()),
    lr=0.001
)
lambda_c = 0.01

# TRAINING LOOP (unchanged)
epochs = 25
print("\nStarting Training")
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs, features = model(batch_X)
        loss = criterion_ce(outputs, batch_y) + lambda_c * criterion_center(features, batch_y)
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
            loss = criterion_ce(outputs, batch_y) + lambda_c * criterion_center(features, batch_y)
            val_loss    += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val   += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()

    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/total_train:.4f} | Train Acc: {100*correct_train/total_train:.2f}% | "
          f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {100*correct_val/total_val:.2f}%")

# FINAL EVALUATION (unchanged)
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