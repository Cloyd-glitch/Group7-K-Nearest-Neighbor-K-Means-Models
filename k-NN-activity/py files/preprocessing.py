# 
# Median Imputation + Standardization


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Load raw data ─────────────────────────────────────────
df = pd.read_csv('diabetes-k-nn.csv')
print("STEP 1 — SCAN FOR IMPOSSIBLE ZEROS")
cols_fix = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
print(f"{'Feature':<28} {'Zero Count':>12} {'% Missing':>10}")
print("-" * 52)
for col in cols_fix:
    zeros = (df[col] == 0).sum()
    pct   = zeros / len(df) * 100
    print(f"  {col:<26} {zeros:>12} {pct:>9.1f}%")

# ── Step 2: Compute medians from non-zero values ──────────
print("STEP 2 — COMPUTE MEDIAN (excluding zeros)")
print("median = middle value when all valid values sorted ascending")
medians = {}
print(f"{'Feature':<28} {'Valid Rows':>10} {'Min':>6} {'Max':>6} {'Median':>8}")
print("-" * 62)
for col in cols_fix:
    valid = df[df[col] != 0][col]
    med   = valid.median()
    medians[col] = med
    print(f"  {col:<26} {len(valid):>10} {valid.min():>6.1f} {valid.max():>6.1f} {med:>8.1f}")

# ── Step 3: Replace zeros with median ────────────────────
print("STEP 3 — REPLACE ZEROS WITH MEDIAN (row by row)")
df_clean = df.copy()
for col in cols_fix:
    zero_rows = df[df[col] == 0].index.tolist()
    med = medians[col]
    print(f"\n  {col} → replacing {len(zero_rows)} zeros with {med}:")
    print(f"    {'Row':<6} {'Before':>8} {'After':>8}")
    for row in zero_rows[:8]:   # show first 8 per column
        print(f"    {row:<6} {0:>8} {med:>8.1f}")
    if len(zero_rows) > 8:
        print(f"    ... and {len(zero_rows) - 8} more rows — all 0 → {med}")
    df_clean[col] = df_clean[col].replace(0, med)

# Verify
print("STEP 3 VERIFICATION — Zero counts after imputation")
for col in cols_fix:
    print(f"  {col:<28} : {(df_clean[col]==0).sum()} zeros remaining")

# ── Step 4: Before vs After Imputation ───────────────────
print("BEFORE vs AFTER IMPUTATION — Mean comparison")
print("(zeros were dragging the means down artificially)")
print(f"{'Feature':<28} {'Mean Before':>12} {'Mean After':>11} {'Change':>10}")
print("-" * 63)
for col in cols_fix:
    bef = df[col].mean()
    aft = df_clean[col].mean()
    print(f"  {col:<26} {bef:>12.2f} {aft:>11.2f} {aft-bef:>+10.2f}")

# ── Step 5: Standardization ───────────────────────────────
print("STEP 4 — STANDARDIZATION   z = (x − μ) ÷ σ")
features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

# Show mean and std for each column
print(f"\n  {'Feature':<28} {'Mean (μ)':>10} {'Std Dev (σ)':>12}")
print("  " + "-" * 52)
for col in features:
    mu  = df_clean[col].mean()
    sig = df_clean[col].std()
    print(f"  {col:<28} {mu:>10.4f} {sig:>12.4f}")

# Manual formula demo — Row 0, ALL features (matching PDF exactly)
print("MANUAL STANDARDIZATION — Row 0, every feature")
print("Patient: Preg=6, Glucose=148, BP=72, Skin=35,")
print("         Insulin=125(imputed), BMI=33.6, DPF=0.627, Age=50")
row0 = df_clean.iloc[0]
for col in features:
    x   = row0[col]
    mu  = df_clean[col].mean()
    sig = df_clean[col].std()
    z   = (x - mu) / sig
    print(f"\n  {col}:")
    print(f"    FORMULA : z = (x − μ) ÷ σ")
    print(f"    PLUG IN : z = ({x} − {mu:.4f}) ÷ {sig:.4f}")
    print(f"    STEP 1  : {x} − {mu:.4f} = {x - mu:.6f}")
    print(f"    STEP 2  : {x - mu:.6f} ÷ {sig:.4f} = {z:.6f}")
    print(f"    VALUE   : {z:.4f}")

# Apply scaler to all rows
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled['Outcome'] = df_clean['Outcome'].values

# ── Step 6: Sample rows comparison ───────────────────────
print("SAMPLE ROWS — Stage 1 (Raw), Stage 2 (Imputed), Stage 3 (Scaled)")
short = ['Preg','Glucose','BPres','SkinThk','Insulin','BMI','DPF','Age','Outcome']

print("\n  Stage 1: Raw Data")
print("  " + "-" * 80)
print("  " + "  ".join(f"{s:>8}" for s in short))
for i in range(5):
    vals = [df.iloc[i][f] for f in features] + [df.iloc[i]['Outcome']]
    print("  " + "  ".join(f"{v:>8.3f}" if isinstance(v, float) else f"{v:>8}" for v in vals))

print("\n  Stage 2: After Imputation")
print("  " + "-" * 80)
print("  " + "  ".join(f"{s:>8}" for s in short))
for i in range(5):
    vals = [df_clean.iloc[i][f] for f in features] + [df_clean.iloc[i]['Outcome']]
    print("  " + "  ".join(f"{v:>8.3f}" if isinstance(v, float) else f"{v:>8}" for v in vals))

print("\n  Stage 3: After Scaling")
print("  " + "-" * 80)
print("  " + "  ".join(f"{s:>8}" for s in short))
for i in range(5):
    vals = [df_scaled.iloc[i][f] for f in features] + [int(df_scaled.iloc[i]['Outcome'])]
    print("  " + "  ".join(f"{v:>8.4f}" if isinstance(v, float) else f"{v:>8}" for v in vals))

