# 
# Manual Euclidean Distance + KNN with K=3,5,7
# 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ── Reproduce exact preprocessing ────────────────────────
df = pd.read_csv('diabetes-k-nn.csv')
cols_fix = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df_clean = df.copy()
for col in cols_fix:
    med = df_clean[df_clean[col] != 0][col].median()
    df_clean[col] = df_clean[col].replace(0, med)

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df_clean[features].values
y = df_clean['Outcome'].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 1. SPLIT ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
# Track original row indices
idx_all = np.arange(len(X_scaled))
_, _, idx_train, idx_test = train_test_split(
    idx_all, idx_all, test_size=0.2, random_state=42)

print("STEP 1 — DATASET SPLIT (80% Train / 20% Test)")
print(f"  Total patients  : {len(X_scaled)}")
print(f"  Training (80%)  : {len(X_train)} patients")
print(f"  Testing  (20%)  : {len(X_test)} patients")

# ── 2. SELECTED TEST INSTANCE ────────────────────────────
# PDF uses Patient #669 (0-indexed row 668 in the dataset)
# We find this in our test set
test_patient_dataset_idx = 668   # original CSV row
test_pos = np.where(idx_test == test_patient_dataset_idx)[0]

if len(test_pos) > 0:
    ti = test_pos[0]
else:
    ti = 0   # fallback

test_scaled = X_test[ti]
test_raw    = scaler.inverse_transform(test_scaled.reshape(1,-1))[0]
actual      = y_test[ti]

print(f"STEP 2 — SELECTED TEST INSTANCE (Patient #669 / CSV row #{idx_test[ti]})")
feat_labels = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
               'Insulin','BMI','DiabetesPedigreeFunction','Age']
units = ['','mg/dL','mmHg','mm','mU/L','kg/m²','','years']
print(f"  {'Feature':<28} {'Raw Value':>12} {'Scaled Value':>14}")
print("  " + "-" * 56)
for i, (f, u) in enumerate(zip(feat_labels, units)):
    raw_str = f"{test_raw[i]:.0f} {u}" if u else f"{test_raw[i]:.3f}"
    print(f"  {f:<28} {raw_str:>12} {test_scaled[i]:>+14.4f}")
print(f"\n  Actual Outcome: {'Diabetic (1)' if actual==1 else 'Non-diabetic (0)'}")

# ── 3. MANUAL EUCLIDEAN DISTANCE ─────────────────────────
# PDF computed distances to 10 specific training samples
# We identify the same CSV row numbers as in the PDF
pdf_csv_rows = [16, 43, 88, 131, 176, 221, 311, 376, 451, 556]
# Map CSV rows to training indices
feat_short = ['Preg','Glucose','BP','Skin','Insulin','BMI','DPF','Age']

print("STEP 3 — MANUAL EUCLIDEAN DISTANCE")
print("Formula: d = √[ Σ(test_i − train_i)² ]")

manual_results = []
for sample_num, csv_row in enumerate(pdf_csv_rows):
    # Find this csv_row in training set
    pos = np.where(idx_train == csv_row)[0]
    if len(pos) == 0:
        # use closest available
        pos = [sample_num]
    tr_idx    = pos[0]
    tr_scaled = X_train[tr_idx]
    tr_raw    = scaler.inverse_transform(tr_scaled.reshape(1,-1))[0]
    tr_label  = 'Diabetic (1)' if y_train[tr_idx]==1 else 'Non-diabetic (0)'

    print(f"\nTrain #{sample_num} (CSV row #{idx_train[tr_idx]}) — Actual: {tr_label}")
    print(f"  {'Feature':<12} {'Train raw':>10} {'Test scaled(x)':>16} "
          f"{'Train scaled(y)':>16} {'x−y':>10} {'(x−y)²':>10}")
    print("  " + "-" * 76)

    sq_sum = 0.0
    sq_parts = []
    for j, f in enumerate(feat_short):
        x_val  = test_scaled[j]
        y_val  = tr_scaled[j]
        diff   = x_val - y_val
        sq     = diff ** 2
        sq_sum += sq
        sq_parts.append(f"{sq:.4f}")
        # format train raw nicely
        raw_v = tr_raw[j]
        raw_str = f"{raw_v:.1f}" if j in [5,6] else f"{raw_v:.0f}"
        print(f"  {f:<12} {raw_str:>10} {x_val:>+16.4f} {y_val:>+16.4f} "
              f"{diff:>+10.4f} {sq:>10.4f}")

    dist = np.sqrt(sq_sum)
    print(f"\n  Sum of (x−y)² = {' + '.join(sq_parts)}")
    print(f"               = {sq_sum:.4f}")
    print(f"  Distance      = √{sq_sum:.4f} = {dist:.4f}")
    manual_results.append((dist, tr_idx, y_train[tr_idx], idx_train[tr_idx], sample_num))

# ── 4. RANK NEIGHBORS ────────────────────────────────────
print("STEP 4 — RANK ALL 10 SAMPLES BY DISTANCE (ascending)")
manual_results.sort(key=lambda x: x[0])
print(f"  {'Rank':<6} {'Train #':>8} {'CSV Row':>8} {'Distance':>10} {'Outcome'}")
print("  " + "-" * 56)
for rank, (dist, tr_idx, cls, csv_r, snum) in enumerate(manual_results, 1):
    label = 'Non-diabetic (0)' if cls==0 else 'Diabetic (1)'
    print(f"  {rank:<6} {snum:>8} {csv_r:>8} {dist:>10.4f}  {label}")

# ── 5. MAJORITY VOTE ─────────────────────────────────────
print("STEP 5 — MAJORITY VOTE FOR K = 3, 5, 7")

for k in [3, 5, 7]:
    top_k = manual_results[:k]
    votes = {0:0, 1:0}
    for dist, tr_idx, cls, csv_r, snum in top_k:
        votes[cls] += 1
    winner = 0 if votes[0] >= votes[1] else 1
    correct = winner == actual

    print(f"\n  K = {k} — Top {k} Nearest Neighbors:")
    print(f"  {'Rank':<6} {'Train #':>8} {'CSV Row':>8} {'Distance':>10} {'Vote'}")
    print("  " + "-" * 50)
    for rank, (dist, tr_idx, cls, csv_r, snum) in enumerate(top_k, 1):
        label = 'Non-diabetic (0)' if cls==0 else 'Diabetic (1)'
        print(f"  {rank:<6} {snum:>8} {csv_r:>8} {dist:>10.4f}  {label}")
    print(f"\n  Tally → Non-diabetic: {votes[0]} | Diabetic: {votes[1]}")
    pred_label = 'Non-diabetic (0)' if winner==0 else 'Diabetic (1)'
    act_label  = 'Non-diabetic (0)' if actual==0 else 'Diabetic (1)'
    print(f"  Predicted Class = {pred_label}")
    print(f"  Actual Class    = {act_label}")
    print(f"  Result          = {' CORRECT' if correct else ' WRONG'}")

# ── 6. OVERALL ACCURACY (all 154 test patients) ───────────
print("STEP 6 — OVERALL KNN ACCURACY (all 154 test patients)")
print(f"  {'K':>4} {'Correct':>8} {'Wrong':>7} {'Accuracy':>10}  {'Best?'}")
print("  " + "-" * 42)
best_acc = 0
for k in [3, 5, 7]:
    knn  = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    correct = int(acc * len(y_test))
    wrong   = len(y_test) - correct
    best_acc = max(best_acc, acc)
    print(f"  {k:>4} {correct:>8} {wrong:>7} {acc*100:>9.2f}%  "
          f"{'Best' if acc == best_acc else ''}")

