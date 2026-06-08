import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import xgboost as xgb

# 1. SETUP & UNIFIED DATA LOADING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Final_Dataset.xlsx'

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
print("Engineering features...")

baseline_mask = full_df['Time_Step'] < 20
baselines = full_df[baseline_mask].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baselines.rename(columns={'y_k': 'y_baseline'}, inplace=True)

full_df = pd.merge(full_df, baselines, on=['Run_ID', 'Attack_Type'], how='left')
full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']

print("   -> Calculating vectorized slopes...")
full_df['Delta_y']  = full_df['y_deviation'].diff().fillna(0)
full_df['Delta_g']  = full_df['g_k'].diff().fillna(0)
full_df['Slope_10'] = full_df['y_deviation'].diff(10).fillna(0)
full_df['Slope_20'] = full_df['y_deviation'].diff(20).fillna(0)

print("   -> Calculating localized CUSUM...")
def fast_cusum(series):
    return series.abs().cumsum()

full_df['CUSUM_r'] = full_df.groupby(['Run_ID', 'Attack_Type'])['r_k'].transform(fast_cusum)

del baselines
gc.collect()

feature_cols = [
    'y_deviation', 'Delta_y', 'Delta_g', 'Slope_10', 'Slope_20',
    'r_k', 'g_k', 'Mean_g', 'Var_g',
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r',
    'ACF_Energy', 'CUSUM_r'
]

# 3. DYNAMIC GRACE PERIOD
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# 4. DATA SPLITTING
total_runs = int(full_df['Run_ID'].max())
train_end, val_end = int(total_runs * 0.70), int(total_runs * 0.85)

train_runs = list(range(1, train_end + 1))
val_runs   = list(range(train_end + 1, val_end + 1))
test_runs  = list(range(val_end + 1, total_runs + 1))

train_df = full_df[full_df['Run_ID'].isin(train_runs)].copy()
val_df   = full_df[full_df['Run_ID'].isin(val_runs)].copy()
test_df  = full_df[full_df['Run_ID'].isin(test_runs)].copy()

# Train + Val combined for base model fitting (meta-learner uses Out of Fold on train only)
train_val_df = full_df[full_df['Run_ID'].isin(train_runs + val_runs)].copy()

# 5. SCALING
# RF and XGB use StandardScaler (paper Section III.D)
# LSTM uses MinMaxScaler separately (paper Section III.D, IV)
# Both scalers fit on training data only to prevent leakage
print("Scaling features...")

# StandardScaler for RF and XGB
std_scaler = StandardScaler()
X_train_std     = std_scaler.fit_transform(train_df[feature_cols])
y_train         = train_df['Label_ID'].values

X_val_std       = std_scaler.transform(val_df[feature_cols])
y_val           = val_df['Label_ID'].values

X_trainval_std  = std_scaler.transform(train_val_df[feature_cols])
y_trainval      = train_val_df['Label_ID'].values

X_test_std      = std_scaler.transform(test_df[feature_cols])
y_test          = test_df['Label_ID'].values

# MinMaxScaler for LSTM
mm_scaler       = MinMaxScaler()
X_train_mm      = mm_scaler.fit_transform(train_df[feature_cols])
X_val_mm        = mm_scaler.transform(val_df[feature_cols])
X_trainval_mm   = mm_scaler.transform(train_val_df[feature_cols])
X_test_mm       = mm_scaler.transform(test_df[feature_cols])

# 6. CLASS WEIGHTS (inverse frequency, applied to LSTM and XGBoost)
num_classes  = 6
input_dim    = len(feature_cols)  # 17

class_counts  = np.bincount(y_train, minlength=num_classes)
safe_counts   = np.where(class_counts == 0, 1, class_counts)
class_weights_np = len(y_train) / (num_classes * safe_counts)  # shape (6,)

# Sample weights array for XGBoost
sample_weights_train    = np.array([class_weights_np[l] for l in y_train])
sample_weights_trainval = np.array([class_weights_np[l] for l in y_trainval])

target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 7. LSTM ARCHITECTURE (paper Section III.E / IV)
#   Input  : (batch, seq_len=10, features=17)   [MinMaxScaled]
#   LSTM   : 64 units, single layer
#   Dropout: 20%
#   Dense  : 6 units, softmax  (multiclass adaptation of paper's sigmoid)
SEQ_LEN = 10  # 10 consecutive timestep window (paper Section III.E)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm    = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :]          # last hidden state
        out         = self.dropout(final_state)
        logits      = self.fc(out)
        return logits

def build_sliding_windows(X_mm, y_labels, seq_len=SEQ_LEN):
    """
    Constructs overlapping sliding windows of length seq_len from
    the flat (N, features) array. Label assigned is the label of the
    last timestep in each window (consistent with sequence classification).
    """
    Xs, ys = [], []
    for i in range(len(X_mm) - seq_len + 1):
        Xs.append(X_mm[i: i + seq_len])
        ys.append(y_labels[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

def train_lstm(X_mm_train, y_train_arr, X_mm_val, y_val_arr,
               class_weights_np, epochs=30, batch_size=128):
    """
    Trains the LSTM model and returns the best-val-loss state dict.
    """
    Xw_tr, yw_tr = build_sliding_windows(X_mm_train, y_train_arr)
    Xw_va, yw_va = build_sliding_windows(X_mm_val,   y_val_arr)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(Xw_tr), torch.tensor(yw_tr)),
        batch_size=batch_size, shuffle=True
    )
    va_loader = DataLoader(
        TensorDataset(torch.tensor(Xw_va), torch.tensor(yw_va)),
        batch_size=batch_size, shuffle=False
    )

    weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    model     = LSTMClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss  = float('inf')
    best_state     = None

    for epoch in range(epochs):
        model.train()
        for bX, by in tr_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, by in va_loader:
                bX, by = bX.to(device), by.to(device)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= len(Xw_va)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   LSTM Epoch [{epoch+1:02d}/{epochs}] | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model

def lstm_predict_proba(model, X_mm, batch_size=256):
    """
    Returns softmax probability matrix (N, 6) for flat input X_mm.
    Windows are built; predictions are assigned to the last timestep index.
    Timesteps not covered by any window (first SEQ_LEN-1 rows) receive
    uniform probability as a safe default.
    """
    Xw, _ = build_sliding_windows(X_mm, np.zeros(len(X_mm), dtype=np.int64))
    loader = DataLoader(
        TensorDataset(torch.tensor(Xw)),
        batch_size=batch_size, shuffle=False
    )
    model.eval()
    all_probs = []
    with torch.no_grad():
        for (bX,) in loader:
            logits = model(bX.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    window_probs = np.vstack(all_probs)   # (N - SEQ_LEN + 1, 6)

    # Pad front rows with uniform probability so output length == len(X_mm)
    pad_rows  = np.full((SEQ_LEN - 1, num_classes), 1.0 / num_classes)
    full_probs = np.vstack([pad_rows, window_probs])
    return full_probs                     # (N, 6)

# 8. BASE MODEL 1 — RANDOM FOREST (paper Section III.E)
#    n_estimators=300, StandardScaler input
print("\n" + "="*50)
print("BASE MODEL 1: Random Forest (n_estimators=300)")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'   # inverse frequency; equivalent to paper's class weighting
)
rf.fit(X_trainval_std, y_trainval)

rf_test_preds = rf.predict(X_test_std)
print("\n=== CLASSIFICATION REPORT (Random Forest) ===")
print(classification_report(y_test, rf_test_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX (Random Forest) ===")
print(pd.DataFrame(confusion_matrix(y_test, rf_test_preds),
                   index=target_names, columns=target_names))

# 9. BASE MODEL 2 — XGBOOST (paper Section III.E)
#    n_estimators=500, max_depth=6, learning_rate=0.05, StandardScaler input
print("\n" + "="*50)
print("BASE MODEL 2: XGBoost (n_estimators=500, depth=6, lr=0.05)")
print("="*50)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_trainval_std, y_trainval, sample_weight=sample_weights_trainval)

xgb_test_preds = xgb_model.predict(X_test_std)
print("\n=== CLASSIFICATION REPORT (XGBoost) ===")
print(classification_report(y_test, xgb_test_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX (XGBoost) ===")
print(pd.DataFrame(confusion_matrix(y_test, xgb_test_preds),
                   index=target_names, columns=target_names))

# 10. BASE MODEL 3 — LSTM (paper Section III.E / IV)
#     64 units -> Dropout(0.2) -> Dense(6), MinMaxScaler input
print("\n" + "="*50)
print("BASE MODEL 3: LSTM (64 units, seq_len=10, MinMaxScaler)")
print("="*50)

lstm_model = train_lstm(
    X_train_mm, y_train,
    X_val_mm,   y_val,
    class_weights_np,
    epochs=30, batch_size=128
)

# Evaluate LSTM on test set directly (window-based inference)
Xw_test, yw_test = build_sliding_windows(X_test_mm, y_test)
test_loader_lstm = DataLoader(
    TensorDataset(torch.tensor(Xw_test), torch.tensor(yw_test)),
    batch_size=256, shuffle=False
)

lstm_model.eval()
lstm_preds_list = []
with torch.no_grad():
    for bX, _ in test_loader_lstm:
        preds = torch.argmax(lstm_model(bX.to(device)), dim=1).cpu().numpy()
        lstm_preds_list.extend(preds)

# yw_test aligns with the last timestep of each window
print("\n=== CLASSIFICATION REPORT (LSTM) ===")
print(classification_report(yw_test, lstm_preds_list, target_names=target_names))
print("\n=== CONFUSION MATRIX (LSTM) ===")
print(pd.DataFrame(confusion_matrix(yw_test, lstm_preds_list),
                   index=target_names, columns=target_names))

# 11. STACKED ENSEMBLE — OUT-OF-FOLD META-FEATURE GENERATION
#  Standard stacking practice (not explicitly stated in paper):
#  5-fold CV on training data → generate OOF probability predictions
#  from each base model → use as input to meta-learner training.
#  This prevents the meta-learner from seeing predictions made on
#  data the base models were trained on (avoids leakage into Level 1).
#  Meta-learner input shape: (N_trainval, 18)
#  = [P_RF(6) | P_XGB(6) | P_LSTM(6)]
print("\n" + "="*50)
print("STACKED ENSEMBLE: Generating Out-of-Fold Meta-Features")
print("="*50)

n_trainval = len(y_trainval)
oof_rf   = np.zeros((n_trainval, num_classes))
oof_xgb  = np.zeros((n_trainval, num_classes))
oof_lstm = np.zeros((n_trainval, num_classes))

kf = KFold(n_splits=5, shuffle=False)   # shuffle=False preserves temporal order

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_trainval_std)):
    print(f"\n  Fold {fold+1}/5")

    # --- Fold data (StandardScaler split) ---
    Xf_tr_std = X_trainval_std[tr_idx]
    Xf_va_std = X_trainval_std[va_idx]
    yf_tr     = y_trainval[tr_idx]
    yf_va     = y_trainval[va_idx]

    # --- Fold data (MinMaxScaler split) ---
    Xf_tr_mm  = X_trainval_mm[tr_idx]
    Xf_va_mm  = X_trainval_mm[va_idx]

    sw_fold   = np.array([class_weights_np[l] for l in yf_tr])

    # RF OOF
    rf_fold = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     random_state=42, class_weight='balanced')
    rf_fold.fit(Xf_tr_std, yf_tr)
    oof_rf[va_idx] = rf_fold.predict_proba(Xf_va_std)

    # XGBoost OOF
    xgb_fold = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  use_label_encoder=False, eval_metric='mlogloss',
                                  random_state=42, n_jobs=-1)
    xgb_fold.fit(Xf_tr_std, yf_tr, sample_weight=sw_fold)
    oof_xgb[va_idx] = xgb_fold.predict_proba(Xf_va_std)

    # LSTM OOF  (re-fit on fold; val split uses 10% of fold-train for early stopping)
    val_cutoff    = int(0.9 * len(Xf_tr_mm))
    Xf_tr_mm_sub  = Xf_tr_mm[:val_cutoff]
    yf_tr_sub     = yf_tr[:val_cutoff]
    Xf_es_mm      = Xf_tr_mm[val_cutoff:]
    yf_es         = yf_tr[val_cutoff:]

    lstm_fold = train_lstm(Xf_tr_mm_sub, yf_tr_sub,
                           Xf_es_mm,     yf_es,
                           class_weights_np, epochs=20, batch_size=128)

    oof_lstm[va_idx] = lstm_predict_proba(lstm_fold, Xf_va_mm)

    del rf_fold, xgb_fold, lstm_fold
    gc.collect()

# Concatenate OOF meta-features: (N_trainval, 18)
meta_train = np.hstack([oof_rf, oof_xgb, oof_lstm])
print(f"\nMeta-train feature matrix shape: {meta_train.shape}")

# 12. GENERATE TEST META-FEATURES
#  Base models retrained on full train+val, predictions on test set.
#  RF and XGB already trained above; LSTM retrained on full train+val.
print("\nGenerating test meta-features from base models...")

# RF test probabilities (model already trained on train_val in Section 8)
rf_test_proba  = rf.predict_proba(X_test_std)

# XGBoost test probabilities (model already trained on train_val in Section 9)
xgb_test_proba = xgb_model.predict_proba(X_test_std)

# LSTM retrained on full train+val for final test inference
print("  Re-training LSTM on full train+val for test meta-features...")
lstm_final = train_lstm(
    X_trainval_mm, y_trainval,
    X_val_mm,      y_val,       # use val as early-stopping reference
    class_weights_np, epochs=30, batch_size=128
)
lstm_test_proba = lstm_predict_proba(lstm_final, X_test_mm)

# Concatenate test meta-features: (N_test, 18)
meta_test = np.hstack([rf_test_proba, xgb_test_proba, lstm_test_proba])
print(f"Meta-test feature matrix shape: {meta_test.shape}")

# 13. META-LEARNER — LOGISTIC REGRESSION (paper Section III.F / IV.A)
#     Trained on OOF meta-features, evaluated on test meta-features.
#     multi_class='auto' handles 6-class output via one-vs-rest or multinomial.
print("\n" + "="*50)
print("META-LEARNER: Logistic Regression")
print("="*50)

meta_learner = LogisticRegression(
    max_iter=1000,
    multi_class='auto',
    random_state=42,
    class_weight='balanced'
)
meta_learner.fit(meta_train, y_trainval)

ensemble_preds = meta_learner.predict(meta_test)

# 14. FINAL EVALUATION
print("\n" + "="*50)
print("FINAL EVALUATION: Hybrid Stacked Ensemble (RF + XGB + LSTM)")
print("="*50)

print("\n=== CLASSIFICATION REPORT (Stacked Ensemble) ===")
print(classification_report(y_test, ensemble_preds, target_names=target_names))
print("\n=== CONFUSION MATRIX (Stacked Ensemble) ===")
print(pd.DataFrame(confusion_matrix(y_test, ensemble_preds),
                   index=target_names, columns=target_names))