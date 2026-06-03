import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP & DATA LOADING
print("Loading datasets")
files = [
    'TTC_Replay_Dataset.xlsx',
    'TTC_FiniteCovert_Dataset.xlsx',
    'TTC_FDI_Dataset.xlsx',
    'TTC_BiasRamp_Dataset.xlsx',
    'TTC_OptimizedZDA_Dataset.xlsx'
]
for f in files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")
dfs = [pd.read_excel(f) for f in files]
full_df = pd.concat(dfs, ignore_index=True)

# Delete hidden trailing spaces from Matlab strings
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()

# BINARY MAPPING: Nominal is 0, everything else is an Attack (1)
full_df['Label_ID'] = full_df['Attack_Type'].apply(lambda x: 0 if x == 'Nominal' else 1)
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

# 2. FEATURE ENGINEERING (CUSUM)
print("Engineering CUSUM feature")
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(lambda x: x.abs().cumsum())
feature_cols = ['y_k', 'r_k', 'g_k', 'Mean_g', 'Var_g', 'Lag3_ACF_r', 'CUSUM_r']

# 3. AMBIGUITY DROP
print("Applying physical logic corrections and dropping ambiguous ramp-up phases...")

# ALL attacks start at k=20. Any data before time step 20 is completely Nominal (0).
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0

# DROP the Grey Area for slow attacks (Replay and Covert)
drop_mask = (
    (full_df['Time_Step'] >= 20) & 
    (full_df['Time_Step'] < 40) & 
    (full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack']))
)
full_df = full_df[~drop_mask].reset_index(drop=True)

# 4. SPLITTING & SCALING
train_runs = list(range(1, 15))  
val_runs = list(range(15, 18))   
test_runs = list(range(18, 21))  
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
        if len(group) >= window_size:
            for i in range(len(group) - window_size + 1):
                X.append(features[i : i + window_size])
                y.append(labels[i + window_size - 1])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)
print("Building temporal sequences...")
WINDOW_SIZE = 35
X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
X_val, y_val     = create_sequences(val_df, WINDOW_SIZE)
X_test, y_test   = create_sequences(test_df, WINDOW_SIZE)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# 6. CLASS WEIGHT CALCULATION (BINARY)
num_classes = 2
class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
print(f"Training Data Class Distribution (Classes 0 to 1): {class_counts}")
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
        x = x.mean(dim=1) 
        out = self.fc(x)
        return out
model = TimeSeriesTransformer(
    input_dim=len(feature_cols), 
    hidden_dim=64, 
    num_classes=num_classes,
    window_size=WINDOW_SIZE
)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. TRAINING LOOP
epochs = 25
print("\nStarting Training")
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
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
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/total_train:.4f} | Train Acc: {100*correct_train/total_train:.2f}% | "
          f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {100*correct_val/total_val:.2f}%")

# 9. FINAL TESTING & EVALUATION (BINARY)
print("\nRunning final binary evaluation on Test Set (Runs 18-20)")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())
target_names = ['Nominal', 'Attack']
print("\n=== FINAL BINARY CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=target_names))
print("\n=== BINARY CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=target_names, columns=target_names))