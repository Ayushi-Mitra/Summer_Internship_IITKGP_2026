import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold

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
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()
label_map = {
    'Nominal': 0, 'Replay Attack': 1, 'Covert Attack': 2, 
    'FDI Attack': 3, 'Bias Attack': 4, 'ZD Attack': 5
}
full_df['Label_ID'] = full_df['Attack_Type'].map(label_map)
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

# 2. FEATURE ENGINEERING
print("Engineering CUSUM feature")
full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(lambda x: x.abs().cumsum())

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

# 3. AMBIGUITY DROP & SPLITTING
print("Applying physical logic corrections...")
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
drop_mask = ((full_df['Time_Step'] >= 20) & (full_df['Time_Step'] < 40) & (full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])))
full_df = full_df[~drop_mask].reset_index(drop=True)

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

WINDOW_SIZE = 30
X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
X_val, y_val     = create_sequences(val_df, WINDOW_SIZE)
X_test, y_test   = create_sequences(test_df, WINDOW_SIZE)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

num_classes = 6
class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
weights_tensor = torch.tensor(len(y_train) / (num_classes * np.where(class_counts == 0, 1, class_counts)), dtype=torch.float32)

# 4. VARIATIONAL AUTOENCODER ARCHITECTURE
class SupervisedVAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, num_classes):
        super(SupervisedVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.flatten_dim = 64 * seq_len
        
        # ENCODER: Compress the 30-step window into deep features
        self.enc_conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # LATENT SPACE: Map features to Mean (Mu) and Variance (LogVar)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # DECODER: Attempt to rebuild the physical sequence from the Latent Space
        self.dec_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(32, input_dim, kernel_size=3, padding=1)
        
        # CLASSIFIER: Attach standard classification to the Latent Space
        self.classifier = nn.Linear(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        # Add random Gaussian noise to force the latent space to be continuous
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Transpose for Conv1D: [Batch, Time, Features] -> [Batch, Features, Time]
        x_t = x.transpose(1, 2)
        
        # Encode
        e = F.relu(self.enc_conv1(x_t))
        e = F.relu(self.enc_conv2(e))
        e = e.view(-1, self.flatten_dim)
        
        # Map to Latent Space
        mu = self.fc_mu(e)
        logvar = self.fc_logvar(e)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        d = self.dec_fc(z)
        d = d.view(-1, 64, self.seq_len)
        d = F.relu(self.dec_conv1(d))
        x_recon = self.dec_conv2(d).transpose(1, 2) # Back to [B, T, D]
        
        # Classify
        logits = self.classifier(z)
        return x_recon, mu, logvar, logits

# Initialize Model (Compressing 240 data points into 16 latent dimensions)
model = SupervisedVAE(input_dim=len(feature_cols), seq_len=WINDOW_SIZE, latent_dim=16, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# VAE Loss Function Components
criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_mse = nn.MSELoss() # For Reconstruction

# Weighting factors to balance the 3 different losses
alpha = 0.5   # Reconstruction Weight
beta = 0.001  # KL Divergence Weight (Keep low so it doesn't destroy the classification)

# 5. TRAINING LOOP
epochs = 25
print("\nStarting VAE Training...")
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward Pass
        x_recon, mu, logvar, logits = model(batch_X)
        
        # 1. Classification Loss
        loss_class = criterion_ce(logits, batch_y)
        
        # 2. Reconstruction Loss (How well did it rebuild the physics?)
        loss_recon = criterion_mse(x_recon, batch_X)
        
        # 3. KL Divergence (Forces the latent space into a smooth Gaussian distribution)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
        
        # Combined Loss
        loss = loss_class + (alpha * loss_recon) + (beta * loss_kld)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(logits, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()
        
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            x_recon, mu, logvar, logits = model(batch_X)
            loss = criterion_ce(logits, batch_y) # Track standard CE for clean val comparison
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(logits, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()
            
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/total_train:.4f} | Train Acc: {100*correct_train/total_train:.2f}% | "
          f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {100*correct_val/total_val:.2f}%")

# 6. FINAL TESTING & EVALUATION
print("\nRunning final evaluation on Test Set (Runs 43-50)")
model.eval()
all_preds, all_labels = [], []
inv_label_map = {v: k for k, v in label_map.items()}

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        _, _, _, logits = model(batch_X)
        probabilities = F.softmax(logits, dim=1)
        _, predicted = torch.max(logits, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

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