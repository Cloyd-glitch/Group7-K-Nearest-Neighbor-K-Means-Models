import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Load data ──────────────────────────────────────────────────────────────────
scaled = pd.read_csv('diabetes_scaled.csv')
features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
X = scaled[features].values
y = scaled['Outcome'].values

# 80/20 split — first 614 train, last 154 test
train_X, test_X = X[:614], X[614:]
train_y, test_y = y[:614], y[614:]

# ── KNN ────────────────────────────────────────────────────────────────────────
def knn_predict(train_X, train_y, test_instance, k):
    dists = np.sqrt(((train_X - test_instance) ** 2).sum(axis=1))
    nn_idx = np.argsort(dists)[:k]
    votes = train_y[nn_idx]
    return Counter(votes).most_common(1)[0][0]

# Test a wider range of K so the curve is more meaningful
k_values = list(range(1, 21))
accuracies = []

print("Computing accuracy for each K...")
for k in k_values:
    preds = np.array([knn_predict(train_X, train_y, test_X[i], k)
                      for i in range(len(test_X))])
    acc = np.sum(preds == test_y) / len(test_y) * 100
    accuracies.append(acc)
    print(f"  K={k:2d}  →  {acc:.2f}%")

# The three homework K values
homework_ks  = [3, 5, 7]
homework_acc = [accuracies[k - 1] for k in homework_ks]
best_acc     = max(accuracies)
best_k       = k_values[accuracies.index(best_acc)]

# ── Colors ─────────────────────────────────────────────────────────────────────
C_LINE      = '#1D9E75'   # teal — main line
C_FILL      = '#9FE1CB'   # teal light — fill under curve
C_HW_MARKER = '#185FA5'   # blue — homework K dots
C_BEST      = '#D85A30'   # coral — best K marker
C_GRID      = '#D3D1C7'
C_TEXT      = '#2C2C2A'
C_MUTED     = '#5F5E5A'
C_BG        = '#FAFAF9'

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# Fill under curve
ax.fill_between(k_values, accuracies, min(accuracies) - 1,
                color=C_FILL, alpha=0.25)

# Main accuracy line
ax.plot(k_values, accuracies,
        color=C_LINE, linewidth=2.2, zorder=3)

# All data points (small)
ax.scatter(k_values, accuracies,
           color=C_LINE, s=40, zorder=4)

# Homework K markers (K=3, 5, 7)
ax.scatter(homework_ks, homework_acc,
           color=C_HW_MARKER, s=90, zorder=5,
           label='Homework K values (3, 5, 7)')
for k, acc in zip(homework_ks, homework_acc):
    ax.annotate(f'K={k}\n{acc:.2f}%',
                xy=(k, acc),
                xytext=(0, 14),
                textcoords='offset points',
                ha='center', fontsize=8.5,
                color=C_HW_MARKER,
                arrowprops=dict(arrowstyle='-', color=C_HW_MARKER, lw=0.8))

# Best K marker (only if different from homework Ks)
if best_k not in homework_ks:
    ax.scatter([best_k], [best_acc],
               color=C_BEST, s=110, zorder=6, marker='*',
               label=f'Best K overall (K={best_k}, {best_acc:.2f}%)')
    ax.annotate(f'K={best_k}\n{best_acc:.2f}%',
                xy=(best_k, best_acc),
                xytext=(0, 14),
                textcoords='offset points',
                ha='center', fontsize=8.5,
                color=C_BEST,
                arrowprops=dict(arrowstyle='-', color=C_BEST, lw=0.8))

# ── Axes & grid ────────────────────────────────────────────────────────────────
ax.set_xlabel('K (number of neighbors)', fontsize=11, color=C_MUTED, labelpad=8)
ax.set_ylabel('Accuracy (%)', fontsize=11, color=C_MUTED, labelpad=8)
ax.set_title('KNN — Accuracy vs K Value\nDiabetes Dataset · 80/20 split · 154 test patients',
             fontsize=12, fontweight='normal', color=C_TEXT, pad=14)

ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values], fontsize=9, color=C_MUTED)
ax.tick_params(colors=C_MUTED, length=0)

y_min = max(0, min(accuracies) - 3)
y_max = min(100, max(accuracies) + 3)
ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax.tick_params(axis='y', labelsize=9)

ax.grid(axis='y', color=C_GRID, linewidth=0.6, linestyle='--')
ax.grid(axis='x', color=C_GRID, linewidth=0.3, linestyle=':')
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Legend ─────────────────────────────────────────────────────────────────────
hw_patch   = mpatches.Patch(color=C_HW_MARKER, label='Homework K values (3, 5, 7)')
line_patch = mpatches.Patch(color=C_LINE,      label='All tested K values (1–20)')
handles    = [line_patch, hw_patch]
if best_k not in homework_ks:
    best_patch = mpatches.Patch(color=C_BEST, label=f'Best K overall (K={best_k})')
    handles.append(best_patch)

ax.legend(handles=handles, fontsize=9, frameon=False,
          loc='lower right', labelcolor=C_MUTED)

plt.tight_layout()
plt.savefig('accuracy_vs_k.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\nSaved: accuracy_vs_k.png")
