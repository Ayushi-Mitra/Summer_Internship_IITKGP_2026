import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import gc

# 1. SETUP, FEATURE ENGINEERING & SCALING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Final_Dataset.xlsx'
full_df = pd.read_excel(filename)
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()

label_map = {'Nominal': 0, 'Replay Attack': 1, 'Covert Attack': 2, 
             'FDI Attack': 3, 'Bias Attack': 4, 'ZD Attack': 5}
full_df['Label_ID'] = full_df['Attack_Type'].map(label_map)
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

baseline_mask = full_df['Time_Step'] < 20
baselines = full_df[baseline_mask].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baselines.rename(columns={'y_k': 'y_baseline'}, inplace=True)

full_df = pd.merge(full_df, baselines, on=['Run_ID', 'Attack_Type'], how='left')
full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']
full_df['Delta_y'] = full_df['y_deviation'].diff().fillna(0)
full_df['Delta_g'] = full_df['g_k'].diff().fillna(0)
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
train_end, val_end = int(total_runs * 0.70), int(total_runs * 0.85)

train_df = full_df[full_df['Run_ID'].isin(list(range(1, train_end + 1)))].copy()
val_df = full_df[full_df['Run_ID'].isin(list(range(train_end + 1, val_end + 1)))].copy()
test_df = full_df[full_df['Run_ID'].isin(list(range(val_end + 1, total_runs + 1)))].copy()

scaler = StandardScaler()
X_train_np = scaler.fit_transform(train_df[feature_cols])
y_train_np = train_df['Label_ID'].values
X_val_np = scaler.transform(val_df[feature_cols])
y_val_np = val_df['Label_ID'].values
X_test_np = scaler.transform(test_df[feature_cols])
y_test_np = test_df['Label_ID'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. ACIL DATA SPLITTING (Majority vs Minority)
# Nominal (0) is Majority. Attacks (1-5) are Minority.
minority_mask = y_train_np > 0
majority_mask = y_train_np == 0

X_train_min = torch.tensor(X_train_np[minority_mask], dtype=torch.float32)
y_train_min = torch.tensor(y_train_np[minority_mask], dtype=torch.long)
X_train_maj = torch.tensor(X_train_np[majority_mask], dtype=torch.float32)
y_train_maj = torch.tensor(y_train_np[majority_mask], dtype=torch.long)

min_loader = DataLoader(TensorDataset(X_train_min, y_train_min), batch_size=120, shuffle=True, drop_last=True)
maj_loader = DataLoader(TensorDataset(X_train_maj, y_train_maj), batch_size=120, shuffle=True, drop_last=True)

# 3. ACIL ARCHITECTURE
input_dim = len(feature_cols)
num_classes = 6
z_dim = 32

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, features, labels):
        x = torch.cat([features, labels], dim=1)
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 4. ACIL TRAINING LOOP (Matches Algorithm 1)
print("\n--- Training ACIL Network ---")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
classifier = Classifier().to(device)

opt_G = optim.Adam(generator.parameters(), lr=0.005)
opt_D = optim.Adam(discriminator.parameters(), lr=0.005)
opt_C = optim.Adam(classifier.parameters(), lr=0.005)

adv_loss = nn.BCELoss()
cls_loss = nn.CrossEntropyLoss()

# ACIL Hyperparameters (Step 13)
delta = 1.0  # Weight for Classifier Feedback (J_Phi)
mu = 1.0     # Weight for Discriminator Feedback (J_D)

epochs = 30
best_val_acc = 0
best_c_state = None

maj_iter = iter(maj_loader)

for epoch in range(epochs):
    discriminator.train(); classifier.train(); generator.train()
    
    for real_min_X, real_min_y in min_loader:
        batch_sz = real_min_X.size(0)
        real_min_X, real_min_y = real_min_X.to(device), real_min_y.to(device)
        
        try: real_maj_X, real_maj_y = next(maj_iter)
        except StopIteration:
            maj_iter = iter(maj_loader)
            real_maj_X, real_maj_y = next(maj_iter)
            
        real_maj_X, real_maj_y = real_maj_X.to(device), real_maj_y.to(device)
        
        valid = torch.ones(batch_sz, 1).to(device)
        fake = torch.zeros(batch_sz, 1).to(device)
        
        # One-hot encode minority labels for G and D
        min_labels_onehot = torch.zeros(batch_sz, num_classes).to(device)
        min_labels_onehot.scatter_(1, real_min_y.unsqueeze(1), 1)
        
        # ALGORITHM STEPS 2-6: Train Discriminator on Minority
        z = torch.randn(batch_sz, z_dim).to(device)
        fake_min_X = generator(z, min_labels_onehot)
        
        opt_D.zero_grad()
        real_validity = discriminator(real_min_X, min_labels_onehot)
        fake_validity = discriminator(fake_min_X.detach(), min_labels_onehot)
        
        # J_D (Equation in Step 5)
        J_D = (adv_loss(real_validity, valid) + adv_loss(fake_validity, fake)) / 2
        J_D.backward()
        opt_D.step()
        
        # ALGORITHM STEPS 7-11: Train Classifier on S_theta
        # S_theta = Fake Minority U Real Minority U Real Majority
        S_theta_X = torch.cat([fake_min_X.detach(), real_min_X, real_maj_X], dim=0)
        S_theta_y = torch.cat([real_min_y, real_min_y, real_maj_y], dim=0)
        
        opt_C.zero_grad()
        logits = classifier(S_theta_X)
        
        # J_AC (Equation in Step 10)
        J_AC = cls_loss(logits, S_theta_y)
        J_AC.backward()
        opt_C.step()
        
        # ALGORITHM STEPS 12-13: Update Generator with Feedback
        opt_G.zero_grad()
        
        # J_Phi: Classifier feedback on newly produced samples
        fake_logits = classifier(fake_min_X)
        J_Phi = cls_loss(fake_logits, real_min_y)
        
        # - J_D (Maximizing D's error by training with 'valid' targets)
        g_validity = discriminator(fake_min_X, min_labels_onehot)
        g_adv = adv_loss(g_validity, valid)
        
        # J_G = \delta * J_Phi - \mu * J_D (Equation in Step 13)
        J_G = (delta * J_Phi) + (mu * g_adv) 
        J_G.backward()
        opt_G.step()
        
    # --- Validation Phase ---
    classifier.eval()
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.long).to(device)
    with torch.no_grad():
        val_logits = classifier(X_val_tensor)
        _, val_preds = torch.max(val_logits, 1)
        val_acc = (val_preds == y_val_tensor).float().mean().item()
        
    print(f"Epoch [{epoch+1}/{epochs}] | J_D: {J_D.item():.4f} | J_AC: {J_AC.item():.4f} | J_G: {J_G.item():.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_c_state = classifier.state_dict().copy()

# 5. FINAL EVALUATION 
print("\n--- Final Evaluation on Test Set ---")
classifier.load_state_dict(best_c_state)
classifier.eval()

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
with torch.no_grad():
    test_class_logits = classifier(X_test_tensor)
    _, test_preds = torch.max(test_class_logits, 1)

test_preds = test_preds.cpu().numpy()
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== CLASSIFICATION REPORT (ACIL) ===")
print(classification_report(y_test_np, test_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test_np, test_preds), index=target_names, columns=target_names))