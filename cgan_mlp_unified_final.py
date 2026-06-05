import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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

print("Engineering features...")
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

# PyTorch Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. CGAN ARCHITECTURE (From Paper Methodology)
input_dim = len(feature_cols)
num_classes = 6
z_dim = 32 # Noise dimension

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 3 Hidden Layers (Expanding architecture)
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim) # Output matches feature count
        )

    def forward(self, z, labels):
        # Concatenate noise and one-hot labels
        x = torch.cat([z, labels], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 3 Hidden Layers (Contracting architecture)
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features, labels):
        # Concatenate features and one-hot labels
        x = torch.cat([features, labels], dim=1)
        return self.net(x)

# 3. CGAN TRAINING PHASE
print("\n--- Phase 1: Training CGAN ---")
batch_size =120
cgan_epochs = 20 # Keep relatively low to prevent mode collapse on balanced data

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion_gan = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.005, betas=(0.5, 0.999))

for epoch in range(cgan_epochs):
    for i, (real_features, real_labels) in enumerate(train_loader):
        batch_sz = real_features.size(0)
        real_features = real_features.to(device)
        
        # One-hot encode labels for conditioning
        real_labels_onehot = torch.zeros(batch_sz, num_classes).to(device)
        real_labels_onehot.scatter_(1, real_labels.to(device).unsqueeze(1), 1)
        
        # Adversarial ground truths
        valid = torch.ones(batch_sz, 1).to(device)
        fake = torch.zeros(batch_sz, 1).to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Loss for real images
        real_pred = discriminator(real_features, real_labels_onehot)
        d_real_loss = criterion_gan(real_pred, valid)
        
        # Loss for fake images
        z = torch.randn(batch_sz, z_dim).to(device)
        gen_features = generator(z, real_labels_onehot)
        fake_pred = discriminator(gen_features.detach(), real_labels_onehot)
        d_fake_loss = criterion_gan(fake_pred, fake)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        # G wants D to classify fake features as valid
        valid_pred = discriminator(gen_features, real_labels_onehot)
        g_loss = criterion_gan(valid_pred, valid)
        
        g_loss.backward()
        optimizer_G.step()
        
    print(f"CGAN Epoch [{epoch+1}/{cgan_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# 4. SYNTHETIC DATA AUGMENTATION
print("\n--- Phase 2: Generating Synthetic Data ---")
samples_per_class = 5000 # Adds 30,000 total synthetic samples to training set
synthetic_X, synthetic_y = [], []

generator.eval()
with torch.no_grad():
    for c in range(num_classes):
        z = torch.randn(samples_per_class, z_dim).to(device)
        labels = torch.full((samples_per_class,), c, dtype=torch.long).to(device)
        
        labels_onehot = torch.zeros(samples_per_class, num_classes).to(device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        gen_feats = generator(z, labels_onehot).cpu().numpy()
        
        synthetic_X.append(gen_feats)
        synthetic_y.append(labels.cpu().numpy())

# Append to original training data
X_train_augmented = np.vstack([X_train_np, *synthetic_X])
y_train_augmented = np.concatenate([y_train_np, *synthetic_y])
print(f"Original Train Size: {X_train_np.shape[0]} | Augmented Train Size: {X_train_augmented.shape[0]}")

# 5. MLP ARCHITECTURE & TRAINING
print("\n--- Phase 3: Training MLP Classifier ---")

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        # 2 Hidden Layers (From Paper)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

mlp = MLPClassifier().to(device)
criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(mlp.parameters(), lr=0.005)

# Load Augmented Data
aug_dataset = TensorDataset(torch.tensor(X_train_augmented, dtype=torch.float32), 
                            torch.tensor(y_train_augmented, dtype=torch.long))
aug_loader = DataLoader(aug_dataset, batch_size=120, shuffle=True)

# Validation Data
X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.long).to(device)

mlp_epochs = 30
best_mlp_val = float('inf')
best_mlp_state = None

for epoch in range(mlp_epochs):
    mlp.train()
    for batch_X, batch_y in aug_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer_mlp.zero_grad()
        outputs = mlp(batch_X)
        loss = criterion_mlp(outputs, batch_y)
        loss.backward()
        optimizer_mlp.step()
        
    # Evaluate
    mlp.eval()
    with torch.no_grad():
        val_outputs = mlp(X_val_tensor)
        val_loss = criterion_mlp(val_outputs, y_val_tensor).item()
        _, predicted = torch.max(val_outputs, 1)
        val_acc = (predicted == y_val_tensor).float().mean().item()
        
    print(f"MLP Epoch [{epoch+1}/{mlp_epochs}] | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_loss < best_mlp_val:
        best_mlp_val = val_loss
        best_mlp_state = mlp.state_dict().copy()

# 6. FINAL EVALUATION
print("\n--- Phase 4: Final Evaluation on Test Set ---")
mlp.load_state_dict(best_mlp_state)
mlp.eval()

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
with torch.no_grad():
    test_outputs = mlp(X_test_tensor)
    _, test_preds = torch.max(test_outputs, 1)

test_preds = test_preds.cpu().numpy()
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== CLASSIFICATION REPORT (CGAN-MLP) ===")
print(classification_report(y_test_np, test_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test_np, test_preds), index=target_names, columns=target_names))