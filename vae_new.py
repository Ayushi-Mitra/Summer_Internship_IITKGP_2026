import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP & UNIFIED DATA LOADING
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

# 2. FEATURE ENGINEERING
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

# 3. DYNAMIC GRACE PERIOD RELABELING
print("Applying physical grace period logic...")
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# 4. GROUP-BASED SPLITTING & SCALING 
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

# *** CRITICAL VAE CHANGE: Filter Training Data to Nominal ONLY ***
train_df_nominal = train_df[train_df['Label_ID'] == 0].copy()

# 5. TEMPORAL SLIDING WINDOW SEQUENCE GENERATION
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
X_train, y_train = create_sequences(train_df_nominal, WINDOW_SIZE) # Trained ONLY on safe physics
X_val, y_val     = create_sequences(val_df, WINDOW_SIZE)           # Includes attacks for threshold tuning
X_test, y_test   = create_sequences(test_df, WINDOW_SIZE)          # Full test evaluation

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# 6. NEURAL NETWORK SPECIFICATION (VAE)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_output):
        attn_scores = self.attention(lstm_output) 
        attn_weights = torch.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_weights * lstm_output, dim=1) 
        return context_vector, attn_weights

class AttentionVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=32, seq_len=35):
        super(AttentionVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # --- ENCODER ---
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.encoder_gru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = TemporalAttention(hidden_dim)
        
        # --- LATENT SPACE (The Distribution) ---
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- DECODER ---
        self.decoder_gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. Encode
        encoded = x.transpose(1, 2)
        encoded = self.relu(self.bn1(self.conv1(encoded)))
        encoded = self.relu(self.bn2(self.conv2(encoded)))
        encoded = encoded.transpose(1, 2)
        
        gru_out, _ = self.encoder_gru(encoded)
        context, _ = self.attention(gru_out)
        
        # 2. Get Distribution & Sample
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode (Repeat z across time steps)
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch, 35, 32]
        dec_out, _ = self.decoder_gru(z_repeated)
        reconstruction = self.decoder_fc(dec_out) # [batch, 35, 15]
        
        return reconstruction, mu, logvar

def vae_loss_function(reconstruction, x, mu, logvar):
    # Mean Squared Error for reconstructing the physics
    recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
    # KL Divergence to force standard normal distribution
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Average over batch size
    batch_size = x.size(0)
    return (recon_loss + kl_loss) / batch_size, recon_loss / batch_size, kl_loss / batch_size

# Initialize Model
model = AttentionVAE(input_dim=len(feature_cols), hidden_dim=96, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 7. UNSUPERVISED TRAINING LOOP
epochs = 15
best_val_loss = float('inf')
best_model_state = None

print(f"\nTraining Attention-Augmented VAE (Unsupervised) for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    train_loss, train_recon, train_kl = 0, 0, 0
    
    for batch_X, _ in train_loader: # Labels ignored during training
        optimizer.zero_grad()
        
        reconstruction, mu, logvar = model(batch_X)
        loss, recon, kl = vae_loss_function(reconstruction, batch_X, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        train_recon += recon.item() * batch_X.size(0)
        train_kl += kl.item() * batch_X.size(0)
        
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, _ in val_loader:
            reconstruction, mu, logvar = model(batch_X)
            loss, _, _ = vae_loss_function(reconstruction, batch_X, mu, logvar)
            val_loss += loss.item() * batch_X.size(0)
            
    avg_train_loss = train_loss / len(X_train)
    avg_val_loss = val_loss / len(X_val)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} (Recon: {train_recon/len(X_train):.4f}, KL: {train_kl/len(X_train):.4f}) | "
          f"Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()

# 8. ANOMALY DETECTION INFERENCE (THRESHOLDING)
print("\nExecuting final evaluation on Test Set...")
model.load_state_dict(best_model_state)
model.eval()

# Step A: Find the optimal threshold using the Validation Set
val_mse_scores = []
val_binary_labels = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        reconstruction, _, _ = model(batch_X)
        # Calculate MSE per sequence
        mse_per_seq = torch.mean((batch_X - reconstruction)**2, dim=[1, 2]).numpy()
        val_mse_scores.extend(mse_per_seq)
        
        # Convert multiclass to binary (0 = Nominal, 1 = Any Attack)
        binary_y = (batch_y.numpy() > 0).astype(int)
        val_binary_labels.extend(binary_y)

val_mse_scores = np.array(val_mse_scores)
val_binary_labels = np.array(val_binary_labels)

# Set threshold at the 95th percentile of Nominal validation errors
nominal_val_mse = val_mse_scores[val_binary_labels == 0]
threshold = np.percentile(nominal_val_mse, 95)
print(f"-> Calculated Anomaly Threshold (95th percentile of safe data): {threshold:.4f}")

# Step B: Evaluate the Test Set
test_mse_scores = []
test_binary_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        reconstruction, _, _ = model(batch_X)
        mse_per_seq = torch.mean((batch_X - reconstruction)**2, dim=[1, 2]).numpy()
        test_mse_scores.extend(mse_per_seq)
        binary_y = (batch_y.numpy() > 0).astype(int)
        test_binary_labels.extend(binary_y)

test_mse_scores = np.array(test_mse_scores)
test_binary_labels = np.array(test_binary_labels)

# Generate Predictions based on Threshold
test_preds = (test_mse_scores > threshold).astype(int)

print("\n=== BINARY ANOMALY DETECTION REPORT (VAE) ===")
print(classification_report(test_binary_labels, test_preds, target_names=['Nominal (0)', 'Anomaly (1)']))
print("\n=== BINARY CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(test_binary_labels, test_preds), 
                   index=['True Nominal', 'True Anomaly'], 
                   columns=['Pred Nominal', 'Pred Anomaly']))