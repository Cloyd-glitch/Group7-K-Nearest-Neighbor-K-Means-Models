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

k_values = [3, 5, 7]
results = {}
for k in k_values:
    preds = np.array([knn_predict(train_X, train_y, test_X[i], k)
                      for i in range(len(test_X))])
    tp = int(np.sum((preds == 1) & (test_y == 1)))
    tn = int(np.sum((preds == 0) & (test_y == 0)))
    fp = int(np.sum((preds == 1) & (test_y == 0)))
    fn = int(np.sum((preds == 0) & (test_y == 1)))
    acc = (tp + tn) / len(test_y) * 100
    results[k] = dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc)

# ── Colors & labels (shared) ───────────────────────────────────────────────────
COLORS = {
    'tp': '#C0DD97',
    'tn': '#C0DD97',
    'fp': '#F7C1C1',
    'fn': '#F7C1C1',
    'tp_txt': '#27500A',
    'tn_txt': '#27500A',
    'fp_txt': '#791F1F',
    'fn_txt': '#791F1F',
    'bg': '#FAFAF9',
    'border': '#B4B2A9',
}

labels     = ['Predicted\nNon-diabetic (0)', 'Predicted\nDiabetic (1)']
row_labels = ['Actual\nNon-diabetic (0)', 'Actual\nDiabetic (1)']

legend_patches = [
    mpatches.Patch(facecolor=COLORS['tn'], edgecolor=COLORS['border'],
                   label='Correct prediction (TN / TP)'),
    mpatches.Patch(facecolor=COLORS['fp'], edgecolor=COLORS['border'],
                   label='Incorrect prediction (FP / FN)'),
]

# ── One figure per K ───────────────────────────────────────────────────────────
for k in k_values:
    r = results[k]
    matrix = np.array([[r['tn'], r['fp']],
                       [r['fn'], r['tp']]])
    cell_colors = [[COLORS['tn'], COLORS['fp']],
                   [COLORS['fn'], COLORS['tp']]]
    text_colors = [[COLORS['tn_txt'], COLORS['fp_txt']],
                   [COLORS['fn_txt'], COLORS['tp_txt']]]
    cell_labels = [['TN', 'FP'], ['FN', 'TP']]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

    for i in range(2):
        for j in range(2):
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle='round,pad=0.04',
                linewidth=1, edgecolor=COLORS['border'],
                facecolor=cell_colors[i][j]
            )
            ax.add_patch(rect)
            ax.text(j, i - 0.12, str(matrix[i][j]),
                    ha='center', va='center',
                    fontsize=32, fontweight='normal',
                    color=text_colors[i][j])
            ax.text(j, i + 0.28, cell_labels[i][j],
                    ha='center', va='center',
                    fontsize=11, fontweight='normal',
                    color=text_colors[i][j], alpha=0.75)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10, color='#5F5E5A')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(row_labels, fontsize=10, color='#5F5E5A')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    best_tag = '  ★ tied best' if k in [3, 7] else ''
    ax.set_title(
        f'KNN Confusion Matrix — K = {k}{best_tag}\n'
        f'Accuracy: {r["acc"]:.2f}%  ·  154 test patients',
        fontsize=11, fontweight='normal', color='#2C2C2A', pad=14
    )

    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04),
               labelcolor='#5F5E5A')

    plt.tight_layout()
    filename = f'confusion_matrix_k{k}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {filename}")
