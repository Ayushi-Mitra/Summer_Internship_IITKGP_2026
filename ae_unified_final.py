import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP, FEATURE ENGINEERING & SCALING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Dataset_New.csv'
full_df = pd.read_csv(filename)
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
full_df['Delta_y']  = full_df['y_deviation'].diff().fillna(0)
full_df['Delta_g']  = full_df['g_k'].diff().fillna(0)
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
train_end  = int(total_runs * 0.70)
val_end    = int(total_runs * 0.85)

train_df = full_df[full_df['Run_ID'].isin(range(1, train_end + 1))].copy()
val_df   = full_df[full_df['Run_ID'].isin(range(train_end + 1, val_end + 1))].copy()
test_df  = full_df[full_df['Run_ID'].isin(range(val_end + 1, total_runs + 1))].copy()

scaler     = StandardScaler()
X_train_np = scaler.fit_transform(train_df[feature_cols])
y_train_np = train_df['Label_ID'].values
X_val_np   = scaler.transform(val_df[feature_cols])
y_val_np   = val_df['Label_ID'].values
X_test_np  = scaler.transform(test_df[feature_cols])
y_test_np  = test_df['Label_ID'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. CLASS WEIGHTS
num_classes  = 6
input_dim    = len(feature_cols)  # 17

class_counts  = np.bincount(y_train_np, minlength=num_classes)
safe_counts   = np.where(class_counts == 0, 1, class_counts)
class_weights = torch.tensor(
    len(y_train_np) / (num_classes * safe_counts), dtype=torch.float32
).to(device)

# 3. ARCHITECTURE
#
#   Encoder:     17 -> 64 -> 32 -> 16  (bottleneck)
#   Decoder:     16 -> 32 -> 64 -> 17  (reconstruction branch)
#   Classifier:  16 -> 32 -> 6         (attached to bottleneck)
#
#   Joint loss = CrossEntropy + recon_lambda * MSE
#   The decoder forces the bottleneck to retain a complete
#   representation of the input, preventing the classifier
#   from collapsing the latent space prematurely.
class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=16, num_classes=6):
        super(AutoencoderClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        z       = self.encoder(x)
        x_recon = self.decoder(z)
        logits  = self.classifier(z)
        return logits, x_recon

# 4. DATA LOADERS
batch_size = 120

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.long)
    ),
    batch_size=batch_size, shuffle=True
)

X_val_t = torch.tensor(X_val_np, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_np, dtype=torch.long).to(device)

# 5. TRAINING
model           = AutoencoderClassifier(input_dim, bottleneck_dim=128, num_classes=num_classes).to(device) #bottleneck tuned to 128
criterion_cls   = nn.CrossEntropyLoss(weight=class_weights)
criterion_recon = nn.MSELoss()
optimizer       = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler       = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

recon_lambda  = 0.0
epochs        = 60
best_val_loss = float('inf')
best_state    = None

print("\n--- Training Autoencoder Classifier ---")
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits, x_recon = model(batch_X)
        loss = criterion_cls(logits, batch_y) + recon_lambda * criterion_recon(x_recon, batch_X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits, val_recon = model(X_val_t)
        val_cls_loss   = criterion_cls(val_logits, y_val_t).item()
        val_recon_loss = criterion_recon(val_recon, X_val_t).item()
        val_loss       = val_cls_loss + recon_lambda * val_recon_loss
        _, val_preds   = torch.max(val_logits, 1)
        val_acc        = (val_preds == y_val_t).float().mean().item()

    print(f"Epoch [{epoch+1:02d}/{epochs}] | "
          f"Val Loss: {val_loss:.4f}  cls={val_cls_loss:.4f}  recon={val_recon_loss:.4f} | "
          f"Val Acc: {val_acc*100:.2f}%")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = model.state_dict().copy()

# 6. FINAL EVALUATION
print("\n--- Final Evaluation on Test Set ---")
model.load_state_dict(best_state)
model.eval()

X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
with torch.no_grad():
    test_logits, _ = model(X_test_t)
    _, test_preds  = torch.max(test_logits, 1)

test_preds   = test_preds.cpu().numpy()
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== CLASSIFICATION REPORT (AUTOENCODER) ===")
print(classification_report(y_test_np, test_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test_np, test_preds),
                   index=target_names, columns=target_names))