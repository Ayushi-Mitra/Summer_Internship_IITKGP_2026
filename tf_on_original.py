import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP & DATA LOADING
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

# Delete hidden trailing spaces from Matlab strings
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

# Print a diagnostic to ensure no datasets were dropped
dropped_rows = full_df['Label_ID'].isna().sum()
if dropped_rows > 0:
    print(f"WARNING: {dropped_rows} rows had unmapped labels and will be dropped.")
    print("Unmapped labels found:", full_df[full_df['Label_ID'].isna()]['Attack_Type'].unique())
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

# 2. FEATURE ENGINEERING (CUSUM & ACF)
print("Engineering CUSUM feature")

# Calculate Cumulative Sum of the absolute residual to catch tiny sine wave leaks
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(lambda x: x.abs().cumsum())

print("Engineering advanced autocorrelation features...")
lag = 10
rolling_window = 30 
def compute_lag_acf(group, window, lag_val):

    # Shift the residual backward by the lag amount
    shifted_r = group['r_k'].shift(lag_val)
    
    # Calculate the rolling Pearson correlation between current r_k and shifted r_k
    return group['r_k'].rolling(window=window).corr(shifted_r)

# Apply to the full dataframe (grouped by Run_ID)
full_df['Lag10_ACF_r'] = full_df.groupby('Run_ID').apply(
    lambda x: compute_lag_acf(x, rolling_window, lag)
).reset_index(level=0, drop=True)

# Fill the unavoidable NaN values created by the shift and rolling window
full_df['Lag10_ACF_r'] = full_df['Lag10_ACF_r'].fillna(0)

# The updated feature list with Lag10_ACF_r added to the end
feature_cols = ['y_k', 'r_k', 'g_k', 'Mean_g', 'Var_g', 'Lag3_ACF_r', 'CUSUM_r', 'Lag10_ACF_r']

# 3. AMBIGUITY DROP
print("Applying physical logic corrections and dropping ambiguous ramp-up phases...")

# ALL attacks start at k=20. Any data before time step 20 is completely Nominal.
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0

# DROP the Grey Area for slow attacks (Replay and Covert)
drop_mask = (
    (full_df['Time_Step'] >= 20) & 
    (full_df['Time_Step'] < 40) & 
    (full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack']))
)
full_df = full_df[~drop_mask].reset_index(drop=True)

# 4. SPLITTING & SCALING
train_runs = list(range(1, 36))  
val_runs = list(range(36, 43))   
test_runs = list(range(43, 51))  
train_df = full_df[full_df['Run_ID'].isin(train_runs)].copy()
val_df = full_df[full_df['Run_ID'].isin(val_runs)].copy()
test_df = full_df[full_df['Run_ID'].isin(test_runs)].copy()
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

# 5. TEMPORAL SLIDING WINDOW GENERATOR
def create_sequences(df, window_size):
    X, y = [], []
    grouped = df.groupby(['Run_ID', 'Attack_Type'])
    for _, group in grouped:
        features = group[feature_cols].values
        labels = group['Label_ID'].values
        
        # Only create sequences if the group is larger than the window
        if len(group) >= window_size:
            for i in range(len(group) - window_size + 1):
                X.append(features[i : i + window_size])
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

# 6. CLASS WEIGHT CALCULATION
num_classes = 6
class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
print(f"Training Data Class Distribution (Classes 0 to 5): {class_counts}")
total_samples = len(y_train)
safe_counts = np.where(class_counts == 0, 1, class_counts)
class_weights = total_samples / (num_classes * safe_counts)
class_weights[class_counts == 0] = 0.0 
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 7. TRANSFORMER NEURAL NETWORK ARCHITECTURE
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, window_size, num_heads=2, num_layers=2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, window_size, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,      
            batch_first=True,     
            dim_feedforward=128,
            dropout=0.4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.input_projection(x)  
        x = x + self.pos_encoder 
        x = self.transformer_encoder(x)
        
        # This is the deep feature representation (mean pooled across the window)
        deep_features = x.mean(dim=1) 
        
        # Final classification
        out = self.fc(deep_features)
        
        # Return BOTH outputs and features for Center Loss
        return out, deep_features

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

# Initialize Transformer 
model = TimeSeriesTransformer(
    input_dim=len(feature_cols), 
    hidden_dim=64, 
    num_classes=num_classes,
    window_size=WINDOW_SIZE
)

# Initialize Dual Loss Functions
criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_center = CenterLoss(num_classes=num_classes, feat_dim=64)

# Combine parameters for the Optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion_center.parameters()), 
    lr=0.001
)

# Set Center Loss Weight
lambda_c = 0.01

# 8. TRAINING LOOP
epochs = 25
print("\nStarting Training")
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Unpack both
        outputs, features = model(batch_X)
        
        # Calculate individual losses
        loss_ce = criterion_ce(outputs, batch_y)
        loss_center = criterion_center(features, batch_y)
        
        # Combine
        loss = loss_ce + (lambda_c * loss_center)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()
        
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs, _ = model(batch_X) # Ignore features in val
            loss = criterion_ce(outputs, batch_y) # Track standard CE for val metrics
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()
            
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/total_train:.4f} | Train Acc: {100*correct_train/total_train:.2f}% | "
          f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {100*correct_val/total_val:.2f}%")

# 9. FINAL TESTING & EVALUATION
print("\nRunning final evaluation on Test Set (Runs 43-50)")
model.eval()
all_preds, all_labels = [], []
inv_label_map = {v: k for k, v in label_map.items()}
print("\n--- Diagnostic: Stealth Attack Confidence Analysis ---")
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs, _ = model(batch_X) # Ignore features in final testing
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())
        for i in range(len(batch_y)):
            true_label = batch_y[i].item()
            pred_label = predicted[i].item()
            if true_label == 2 and pred_label == 0:
                conf_nominal = probabilities[i][0].item() * 100
                conf_covert = probabilities[i][2].item() * 100
                print(f"[Covert Error] Guessed Nominal | Conf: Nominal {conf_nominal:.1f}%, Covert {conf_covert:.1f}%")
            elif true_label == 1 and pred_label != 1:
                pred_name = inv_label_map[pred_label]
                conf_wrong_guess = probabilities[i][pred_label].item() * 100
                conf_replay = probabilities[i][1].item() * 100
                print(f"[Replay Error] Guessed {pred_name} | Conf: {pred_name} {conf_wrong_guess:.1f}%, Replay {conf_replay:.1f}%")

active_labels = [k for k, v in label_map.items() if v in np.unique(all_labels)]
target_names = []
for idx in range(num_classes):
    for name, val in label_map.items():
        if val == idx and name not in target_names and name in active_labels:
            target_names.append(name)
            break
print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=target_names, columns=target_names))