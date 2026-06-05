import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import gc

# ==========================================
# 1. SETUP & UNIFIED DATA LOADING
# ==========================================
print("Loading Unified Dataset...")
filename = 'TTC_Unified_Final_Dataset.xlsx' # Switch to .csv if needed for memory

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

# ==========================================
# 2. MEMORY-OPTIMIZED FEATURE ENGINEERING
# ==========================================
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
    'r_k', 'g_k', 'Mean_g', 'Var_g', 
    'Lag1_ACF_r', 'Lag2_ACF_r', 'Lag3_ACF_r', 'Lag4_ACF_r', 'Lag5_ACF_r', 'Lag6_ACF_r', 
    'ACF_Energy', 'CUSUM_r'
]

# ==========================================
# 3. DYNAMIC GRACE PERIOD
# ==========================================
full_df.loc[full_df['Time_Step'] < 20, 'Label_ID'] = 0
slow_attack_mask = full_df['Attack_Type'].isin(['Replay Attack', 'Covert Attack'])
full_df.loc[slow_attack_mask & (full_df['Time_Step'] < 35), 'Label_ID'] = 0

# ==========================================
# 4. DATA SPLITTING & SCALING (2D Tabular Format)
# ==========================================
total_runs = int(full_df['Run_ID'].max())
train_end, val_end = int(total_runs * 0.70), int(total_runs * 0.85)

train_runs = list(range(1, train_end + 1))  
val_runs = list(range(train_end + 1, val_end + 1))   
test_runs = list(range(val_end + 1, total_runs + 1))  

# Combine Train and Val for Random Forest (it doesn't need early stopping)
train_val_df = full_df[full_df['Run_ID'].isin(train_runs + val_runs)].copy()
test_df = full_df[full_df['Run_ID'].isin(test_runs)].copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(train_val_df[feature_cols])
y_train = train_val_df['Label_ID'].values

X_test = scaler.transform(test_df[feature_cols])
y_test = test_df['Label_ID'].values

del train_val_df, full_df
gc.collect()

# ==========================================
# 5. RANDOM FOREST BAGGING (Testing 3, 5, 7 Subsets)
# ==========================================
print("\n" + "="*50)
print("Evaluating Random Forest Bagging (Paper Subsets)")
print("="*50)

subset_list = [3, 5, 7]
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

for n_subsets in subset_list:
    print(f"\n--- Training RF with {n_subsets} Subsets (Estimators) ---")
    
    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_subsets,  # Maps directly to the paper's 3, 5, 7 subsets
        max_depth=None,          # Allow trees to capture deep, sharp boundaries
        bootstrap=True,          # Ensures each subset is a random bag of data
        max_samples=0.8,         # Each subset uses 80% of the data
        n_jobs=-1,               # Use all CPU cores for speed
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    test_preds = rf_model.predict(X_test)
    
    # Calculate and Print F1 Score
    f_measure = f1_score(y_test, test_preds, average='macro')
    print(f"Macro F-measure (FM): {f_measure:.4f}")
    
    # Print the full classification report for detailed analysis
    print(classification_report(y_test, test_preds, target_names=target_names))

    print(f"\n=== CONFUSION MATRIX ({n_subsets} Subsets) ===")
    # Prints the confusion matrix for the model
    print(pd.DataFrame(confusion_matrix(y_test, test_preds), index=target_names, columns=target_names))