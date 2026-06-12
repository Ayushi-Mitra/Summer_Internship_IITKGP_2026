import gc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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

# 2. THREE-PLAYER GAMO ARCHITECTURE (3-Layer Scaled)
input_dim = len(feature_cols) # 17 features
num_classes = 6

# Player 1: The Convex Generator 
# Expands to latent space [32, 64, 128], blends, and decodes back [128, 64, 32]
class ConvexGenerator(nn.Module):
    def __init__(self):
        super(ConvexGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128) # Deep latent blending space
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x1, x2, alpha):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        # Convex combination in the deep latent space
        z_blend = alpha * z1 + (1 - alpha) * z2
        return self.decoder(z_blend)

# Player 2: The Discriminator 
# 3 Hidden Layers: [128, 64, 32] -> Real/Fake
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Player 3: The Auxiliary Classifier 
# 2 Hidden Layers: [128, 64] -> 6 Attack Classes
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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

# 3. THREE-PLAYER TRAINING LOOP
print("\n--- Training 3-Layer Traditional GAMO Network ---")
batch_size = 120
epochs = 30 

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_np, dtype=torch.float32), 
                                        torch.tensor(y_train_np, dtype=torch.long)), 
                          batch_size=batch_size, shuffle=True)

X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.long).to(device)

generator = ConvexGenerator().to(device)
discriminator = Discriminator().to(device)
classifier = Classifier().to(device)

adv_loss_fn = nn.BCELoss()
cls_loss_fn = nn.CrossEntropyLoss()

opt_G = optim.Adam(generator.parameters(), lr=0.005)
opt_D = optim.Adam(discriminator.parameters(), lr=0.005)
opt_C = optim.Adam(classifier.parameters(), lr=0.005)

best_val_acc = 0
best_c_state = None

for epoch in range(epochs):
    discriminator.train()
    classifier.train()
    generator.train()
    
    for real_features, real_labels in train_loader:
        batch_sz = real_features.size(0)
        real_features, real_labels = real_features.to(device), real_labels.to(device)
        
        valid = torch.ones(batch_sz, 1).to(device)
        fake = torch.zeros(batch_sz, 1).to(device)
        
        # --- Generate Fake Data ---
        shuffle_idx = torch.randperm(batch_sz)
        x2_features = real_features[shuffle_idx]
        alpha = torch.rand(batch_sz, 1).to(device)
        gen_features = generator(real_features, x2_features, alpha)
        
        # STEP 1: Train Discriminator
        opt_D.zero_grad()
        real_validity = discriminator(real_features)
        fake_validity = discriminator(gen_features.detach())
        
        d_loss = (adv_loss_fn(real_validity, valid) + adv_loss_fn(fake_validity, fake)) / 2
        d_loss.backward()
        opt_D.step()
        
        # STEP 2: Train Classifier
        opt_C.zero_grad()
        real_logits = classifier(real_features)
        fake_logits = classifier(gen_features.detach())
        
        c_loss = (cls_loss_fn(real_logits, real_labels) + cls_loss_fn(fake_logits, real_labels)) / 2
        c_loss.backward()
        opt_C.step()
        
        # STEP 3: Train Generator
        opt_G.zero_grad()
        g_validity = discriminator(gen_features)
        g_adv_loss = adv_loss_fn(g_validity, valid)
        
        g_logits = classifier(gen_features)
        g_cls_loss = cls_loss_fn(g_logits, real_labels)
        
        g_loss = g_adv_loss + g_cls_loss
        g_loss.backward()
        opt_G.step()
        
    # --- Validation Phase ---
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(X_val_tensor)
        _, val_preds = torch.max(val_logits, 1)
        val_acc = (val_preds == y_val_tensor).float().mean().item()
        
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | C Loss: {c_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_c_state = classifier.state_dict().copy()

# 4. FINAL EVALUATION 
print("\n--- Final Evaluation on Test Set ---")
classifier.load_state_dict(best_c_state)
classifier.eval()

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
with torch.no_grad():
    test_class_logits = classifier(X_test_tensor)
    _, test_preds = torch.max(test_class_logits, 1)

test_preds = test_preds.cpu().numpy()
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== CLASSIFICATION REPORT (3-LAYER GAMO) ===")
print(classification_report(y_test_np, test_preds, target_names=target_names))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test_np, test_preds), index=target_names, columns=target_names))