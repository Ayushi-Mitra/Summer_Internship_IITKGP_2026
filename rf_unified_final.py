import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP & LOADING
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import gc

# 1. SETUP & UNIFIED DATA LOADING
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Dataset_New.csv' # Switch to .csv if needed for memory

if not os.path.exists(filename):
    raise FileNotFoundError(f"Missing unified dataset: {filename}")

full_df = pd.read_csv(filename)
full_df['Attack_Type'] = full_df['Attack_Type'].astype(str).str.strip()

label_map = {
    'Nominal': 0, 'Replay Attack': 1, 'Covert Attack': 2, 
    'FDI Attack': 3, 'Bias Attack': 4, 'ZD Attack': 5
}

full_df['Label_ID'] = full_df['Attack_Type'].map(label_map)
full_df = full_df.dropna(subset=['Label_ID'])
full_df['Label_ID'] = full_df['Label_ID'].astype(int)

# MEMORY-OPTIMIZED FEATURE ENGINEERING
print("Engineering features (Memory Optimized Mode)...")

baseline_mask = full_df['Time_Step'] < 20
baselines = full_df[baseline_mask].groupby(['Run_ID', 'Attack_Type'])['y_k'].mean().reset_index()
baselines.rename(columns={'y_k': 'y_baseline'}, inplace=True)

full_df = pd.merge(full_df, baselines, on=['Run_ID', 'Attack_Type'], how='left')
full_df['y_deviation'] = full_df['y_k'] - full_df['y_baseline']

print("   -> Calculating vectorized slopes...")
full_df['Delta_y'] = full_df['y_deviation'].diff().fillna(0)
full_df['Delta_g'] = full_df['g_k'].diff().fillna(0)
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
    'r_k', 'g_k', 'Mean_g', 'Var_g', 'K_Attack',
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

# DYNAMIC GRACE PERIOD
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# DATA SPLITTING & SCALING
total_runs = int(full_df['Run_ID'].max())
train_end, val_end = int(total_runs * 0.70), int(total_runs * 0.85)

# 2. AGGREGATE SEQUENCES (Random Forest required adaptation)
def aggregate_for_rf(df, feature_cols):
    """
    Transforms 3D sequences into 2D tabular features per run.
    """
    agg_funcs = ['mean', 'std', 'max', 'min']
    agg_dict = {col: agg_funcs for col in feature_cols}
    agg_dict['Label_ID'] = 'first'
    
    tabular_df = df.groupby(['Run_ID', 'Attack_Type']).agg(agg_dict)
    tabular_df.columns = ['_'.join(col).strip() for col in tabular_df.columns.values]
    return tabular_df.reset_index()

tabular_data = aggregate_for_rf(full_df, feature_cols)

# 3. SPLITTING (Consistent with your 70/15/15 ratio logic)
train_runs = list(range(1, int(total_runs * 0.70) + 1))
test_runs = list(range(int(total_runs * 0.85) + 1, total_runs + 1))

train_df = tabular_data[tabular_data['Run_ID'].isin(train_runs)]
test_df = tabular_data[tabular_data['Run_ID'].isin(test_runs)]

X_train = train_df.drop(columns=['Run_ID', 'Attack_Type', 'Label_ID_first'])
y_train = train_df['Label_ID_first']
X_test = test_df.drop(columns=['Run_ID', 'Attack_Type', 'Label_ID_first'])
y_test = test_df['Label_ID_first']

# 4. RANDOM FOREST CLASSIFIER
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# 5. EVALUATION (Standardized output)
y_pred = rf_model.predict(X_test)
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== RANDOM FOREST CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=target_names))
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=target_names, columns=target_names))