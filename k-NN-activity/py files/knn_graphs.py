import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Load data ──────────────────────────────────────────────────────────────────
raw     = pd.read_csv('diabetes-k-nn.csv')
imputed = pd.read_csv('diabetes_imputed.csv')
scaled  = pd.read_csv('diabetes_scaled.csv')

features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# ── Shared style ───────────────────────────────────────────────────────────────
BG       = '#FAFAF9'
MUTED    = '#5F5E5A'
TEXT     = '#2C2C2A'
BORDER   = '#D3D1C7'
C_ND     = '#1D9E75'   # teal  — non-diabetic
C_D      = '#D85A30'   # coral — diabetic
C_BLUE   = '#185FA5'
C_GREEN  = '#3B6D11'
C_RED    = '#A32D2D'
C_AMBER  = '#BA7517'

def base_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    return fig, ax

def save(fig, name):
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {name}')


#— Dataset class distribution  (Part 1)

fig, ax = base_fig(6, 4)

counts = [500, 268]
labels = ['Non-diabetic (0)', 'Diabetic (1)']
colors = [C_ND, C_D]
bars   = ax.bar(labels, counts, color=colors, width=0.45,
                edgecolor=BG, linewidth=1.5)

for bar, count in zip(bars, counts):
    pct = count / 768 * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, color=TEXT)

ax.set_ylim(0, 620)
ax.set_ylabel('Number of patients', fontsize=10, color=MUTED, labelpad=8)
ax.set_title('Dataset Class Distribution\n768 total patients',
             fontsize=12, fontweight='normal', color=TEXT, pad=12)
ax.yaxis.grid(True, color=BORDER, linewidth=0.6, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=9, labelcolor=MUTED)

save(fig, 'graph1_class_distribution.png')


#Missing (zero) values per feature  (Part 2)

fig, ax = base_fig(8, 4.5)

zero_cols   = ['Insulin', 'SkinThickness', 'BloodPressure', 'BMI', 'Glucose']
zero_counts = [374,       227,             35,              11,    5]
zero_pcts   = [c / 768 * 100 for c in zero_counts]
bar_colors  = [C_D if p > 20 else C_AMBER if p > 3 else C_ND for p in zero_pcts]

bars = ax.barh(zero_cols, zero_counts, color=bar_colors,
               edgecolor=BG, linewidth=1.2, height=0.5)

for bar, count, pct in zip(bars, zero_counts, zero_pcts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f'{count}  ({pct:.1f}%)',
            va='center', fontsize=10, color=TEXT)

ax.set_xlim(0, 450)
ax.set_xlabel('Number of zero (missing) values', fontsize=10, color=MUTED, labelpad=8)
ax.set_title('Zero / Missing Values per Feature  (before imputation)\n'
             'Zeros are biologically impossible — treated as missing data',
             fontsize=12, fontweight='normal', color=TEXT, pad=12)
ax.xaxis.grid(True, color=BORDER, linewidth=0.6, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=9, labelcolor=MUTED)

legend_patches = [
    mpatches.Patch(color=C_D,     label='Critical (>20% missing)'),
    mpatches.Patch(color=C_AMBER, label='Moderate (3–20% missing)'),
    mpatches.Patch(color=C_ND,    label='Minor (<3% missing)'),
]
ax.legend(handles=legend_patches, fontsize=9, frameon=False,
          loc='lower right', labelcolor=MUTED)

save(fig, 'graph2_missing_values.png')


# Standardization: mean & std dev per feature  (Part 2)
means = [3.845, 121.656, 72.387, 29.108, 140.672, 32.455, 0.472, 33.241]
stds  = [3.370,  30.438, 12.097,  8.791,  86.383,  6.875, 0.331, 11.760]
feat_labels = ['Pregnancies', 'Glucose', 'BP', 'SkinThick.',
               'Insulin', 'BMI', 'DPF', 'Age']

x   = np.arange(len(feat_labels))
w   = 0.38

fig, ax = base_fig(10, 5)

b1 = ax.bar(x - w/2, means, width=w, color=C_BLUE,  alpha=0.85,
            label='Mean (μ)', edgecolor=BG)
b2 = ax.bar(x + w/2, stds,  width=w, color=C_AMBER, alpha=0.85,
            label='Std Dev (σ)', edgecolor=BG)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, color=MUTED)

ax.set_xticks(x)
ax.set_xticklabels(feat_labels, fontsize=9, color=TEXT)
ax.set_ylabel('Value', fontsize=10, color=MUTED, labelpad=8)
ax.set_title('Mean (μ) and Std Dev (σ) per Feature\n'
             'Used in z = (x − μ) ÷ σ  to standardize every column to mean=0, std=1',
             fontsize=12, fontweight='normal', color=TEXT, pad=12)
ax.yaxis.grid(True, color=BORDER, linewidth=0.6, linestyle='--')
ax.set_axisbelow(True)
ax.legend(fontsize=10, frameon=False, labelcolor=MUTED)
ax.tick_params(axis='y', labelsize=9, labelcolor=MUTED)

save(fig, 'graph4_mean_std.png')

# 
# Train / Test split  (Part 3)
# 
fig, ax = base_fig(6, 3.5)

segments = [614, 154]
seg_labels = ['Training set\n614 patients (80%)', 'Test set\n154 patients (20%)']
seg_colors = [C_BLUE, C_AMBER]
left = 0
bar_h = 0.55

for seg, label, color in zip(segments, seg_labels, seg_colors):
    ax.barh(0, seg, left=left, color=color, height=bar_h, edgecolor=BG, linewidth=1.5)
    ax.text(left + seg / 2, 0, label,
            ha='center', va='center', fontsize=10, color='white', fontweight='normal')
    left += seg

ax.set_xlim(0, 800)
ax.set_ylim(-0.6, 0.6)
ax.axis('off')
ax.set_title('80 / 20 Train-Test Split\n768 total patients',
             fontsize=12, fontweight='normal', color=TEXT, pad=12)
ax.text(768 / 2, -0.42, '768 patients total',
        ha='center', fontsize=9, color=MUTED)

save(fig, 'graph6_train_test_split.png')

# 
# Manual distance computation bar chart for Patient #669  (Part 3)
# 
fig, ax = base_fig(9, 5)

train_nums  = [f'Train #0\n(Row 15)', f'Train #1\n(Row 42)', f'Train #2\n(Row 87)',
               f'Train #3\n(Row 130)', f'Train #4\n(Row 175)', f'Train #5\n(Row 220)',
               f'Train #6\n(Row 310)', f'Train #7\n(Row 375)', f'Train #8\n(Row 450)',
               f'Train #9\n(Row 555)']
distances   = [1.8573, 3.8448, 2.7300, 3.6494, 3.3867,
               5.3438, 1.7071, 3.8799, 3.9357, 2.0831]
outcomes    = [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]
bar_colors  = [C_D if o == 1 else C_ND for o in outcomes]

x = np.arange(len(train_nums))
bars = ax.bar(x, distances, color=bar_colors, width=0.6, edgecolor=BG, linewidth=1.2)

# Annotate K boundaries
for k, xpos, col in [(3, 2.5, '#888780'), (5, 4.5, '#888780'), (7, 6.5, '#888780')]:
    ax.axvline(xpos, color=col, linewidth=1, linestyle=':')
    ax.text(xpos + 0.05, 5.1, f'K={k} cutoff', fontsize=7.5, color=col, rotation=0)

for bar, d in zip(bars, distances):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
            f'{d:.4f}', ha='center', va='bottom', fontsize=8, color=MUTED)

ax.set_xticks(x)
ax.set_xticklabels(train_nums, fontsize=8, color=TEXT)
ax.set_ylabel('Euclidean Distance', fontsize=10, color=MUTED, labelpad=8)
ax.set_ylim(0, 5.9)
ax.set_title('Graph 7 — Euclidean Distances from Patient #669 to 10 Training Samples\n'
             'Sorted by train index  ·  color = actual outcome of training patient',
             fontsize=12, fontweight='normal', color=TEXT, pad=12)
ax.yaxis.grid(True, color=BORDER, linewidth=0.5, linestyle='--')
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color=C_ND, label='Non-diabetic neighbor (0)'),
    mpatches.Patch(color=C_D,  label='Diabetic neighbor (1)'),
]
ax.legend(handles=legend_patches, fontsize=9, frameon=False,
          loc='upper right', labelcolor=MUTED)

save(fig, 'graph7_manual_distances.png')

# 
# Ranked neighbors + K voting visualization  (Part 3)
# 
fig, ax = base_fig(9, 5)

ranked_labels = ['#6\nRow310', '#0\nRow15', '#9\nRow555', '#2\nRow87', '#4\nRow175',
                 '#3\nRow130', '#1\nRow42', '#7\nRow375', '#8\nRow450', '#5\nRow220']
ranked_dists  = [1.7071, 1.8573, 2.0831, 2.7300, 3.3867,
                 3.6494, 3.8448, 3.8799, 3.9357, 5.3438]
ranked_outcomes = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
bar_colors    = [C_ND if o == 0 else C_D for o in ranked_outcomes]

x = np.arange(len(ranked_labels))
bars = ax.bar(x, ranked_dists, color=bar_colors, width=0.6, edgecolor=BG, linewidth=1.2)

# K boundary shading
k_colors = ['#9FE1CB', '#C0DD97', '#E1F5EE']
k_vals   = [3, 5, 7]
k_ends   = [2.5, 4.5, 6.5]
for ki, (kv, ke, kc) in enumerate(zip(k_vals, k_ends, k_colors)):
    ax.axvspan(-0.5, ke, alpha=0.12, color=kc, zorder=0)
    ax.text(ke - 0.1, 5.55, f'K={kv}', fontsize=8, color=C_GREEN,
            ha='right', fontweight='normal')

for bar, d in zip(bars, ranked_dists):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{d:.4f}', ha='center', va='bottom', fontsize=8, color=MUTED)

ax.set_xticks(x)
ax.set_xticklabels(ranked_labels, fontsize=8.5, color=TEXT)
ax.set_ylabel('Euclidean Distance', fontsize=10, color=MUTED, labelpad=8)
ax.set_ylim(0, 5.9)
ax.set_xlabel('Neighbors ranked closest → farthest', fontsize=10, color=MUTED, labelpad=8)
ax.set_title('Graph 8 — Ranked Neighbors & K Voting Windows for Patient #669\n'
             'Green shading = neighbors included in each K  ·  All K values predict Non-diabetic (0) ✓',
             fontsize=12, fontweight='normal', color=TEXT, pad=12)
ax.yaxis.grid(True, color=BORDER, linewidth=0.5, linestyle='--')
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color=C_ND, label='Non-diabetic neighbor (0)'),
    mpatches.Patch(color=C_D,  label='Diabetic neighbor (1)'),
]
ax.legend(handles=legend_patches, fontsize=9, frameon=False,
          loc='upper left', labelcolor=MUTED)

save(fig, 'graph8_ranked_neighbors.png')

# 
# Accuracy vs K (Part 4) — already in individual script, included here
# 
X_data = scaled[features].values
y_data = scaled['Outcome'].values
train_X, test_X = X_data[:614], X_data[614:]
train_y, test_y = y_data[:614], y_data[614:]

def knn_predict(tX, ty, instance, k):
    dists = np.sqrt(((tX - instance) ** 2).sum(axis=1))
    nn_idx = np.argsort(dists)[:k]
    return Counter(ty[nn_idx]).most_common(1)[0][0]

k_values   = list(range(1, 21))
accuracies = []
print("Computing accuracy for Graph 9...")
for k in k_values:
    preds = np.array([knn_predict(train_X, train_y, test_X[i], k)
                      for i in range(len(test_X))])
    accuracies.append(np.sum(preds == test_y) / len(test_y) * 100)

fig, ax = base_fig(9, 5)

ax.fill_between(k_values, accuracies, min(accuracies) - 1,
                color='#9FE1CB', alpha=0.25)
ax.plot(k_values, accuracies, color=C_ND, linewidth=2.2, zorder=3)
ax.scatter(k_values, accuracies, color=C_ND, s=40, zorder=4)

hw_ks  = [3, 5, 7]
hw_acc = [accuracies[k - 1] for k in hw_ks]
ax.scatter(hw_ks, hw_acc, color=C_BLUE, s=90, zorder=5)
for k, a in zip(hw_ks, hw_acc):
    ax.annotate(f'K={k}\n{a:.2f}%', xy=(k, a), xytext=(0, 14),
                textcoords='offset points', ha='center', fontsize=8.5,
                color=C_BLUE,
                arrowprops=dict(arrowstyle='-', color=C_BLUE, lw=0.8))

best_acc = max(accuracies)
best_k   = k_values[accuracies.index(best_acc)]
ax.scatter([best_k], [best_acc], color=C_D, s=120, zorder=6, marker='*')
ax.annotate(f'K={best_k}\n{best_acc:.2f}%', xy=(best_k, best_acc), xytext=(0, 14),
            textcoords='offset points', ha='center', fontsize=8.5, color=C_D,
            arrowprops=dict(arrowstyle='-', color=C_D, lw=0.8))

ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values], fontsize=9, color=MUTED)
ax.set_xlabel('K (number of neighbors)', fontsize=11, color=MUTED, labelpad=8)
ax.set_ylabel('Accuracy (%)', fontsize=11, color=MUTED, labelpad=8)
ax.set_ylim(min(accuracies) - 3, max(accuracies) + 3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax.tick_params(axis='y', labelsize=9, labelcolor=MUTED)
ax.yaxis.grid(True, color=BORDER, linewidth=0.6, linestyle='--')
ax.set_title('Graph 9 — Accuracy vs K Value\nDiabetes Dataset  ·  80/20 split  ·  154 test patients',
             fontsize=12, fontweight='normal', color=TEXT, pad=14)
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color=C_ND,   label='All K values (1–20)'),
    mpatches.Patch(color=C_BLUE, label='Homework K values (3, 5, 7)'),
    mpatches.Patch(color=C_D,    label=f'Best K (K={best_k})'),
]
ax.legend(handles=legend_patches, fontsize=9, frameon=False,
          loc='lower right', labelcolor=MUTED)

save(fig, 'graph9_accuracy_vs_k.png')

print("\nAll 9 graphs saved successfully.")
