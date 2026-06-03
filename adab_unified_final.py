import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
X_train = scaler.fit_transform(train_df[feature_cols])
y_train = train_df['Label_ID'].values

X_val = scaler.transform(val_df[feature_cols])
y_val = val_df['Label_ID'].values

X_test = scaler.transform(test_df[feature_cols])
y_test = test_df['Label_ID'].values

# 2. PAPER METHODOLOGY: ADABOOST ENSEMBLE
# Testing the estimator counts {100, 150} defined in Table 1 for the Boosting methods
print("\nTraining AdaBoost Ensemble (Testing n_estimators: 100, 150)...")
estimators_list = [100, 150]
best_n = None
best_val_acc = 0
best_model = None

for n in estimators_list:
    print(f"-> Training AdaBoost with n_estimators={n}...")
    
    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3), #standard for adaboost
        n_estimators=n, 
        random_state=42
    )
    
    adaboost.fit(X_train, y_train)
    
    val_preds = adaboost.predict(X_val)
    val_acc = np.mean(val_preds == y_val)
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_n = n
        best_model = adaboost

print(f"\nBest model selected with n_estimators={best_n}")

# 3. FINAL EVALUATION
print("\nExecuting final evaluation on Test Set...")
test_preds = best_model.predict(X_test)
target_names = ['Nominal', 'Replay Attack', 'Covert Attack', 'FDI Attack', 'Bias Attack', 'ZD Attack']

print("\n=== BASELINE CLASSIFICATION REPORT (ADABOOST ENSEMBLE) ===")
print(classification_report(y_test, test_preds, target_names=target_names))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, test_preds), index=target_names, columns=target_names))