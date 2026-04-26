import math
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# DATASET: 30 books — Weight (g) and Price (₱)
# ─────────────────────────────────────────────
data = [
    [120, 24],  [150, 30],  [100, 20],  [180, 36],  [130, 26],  [160, 32],  # pts 1–6
    [450, 108], [500, 120], [420, 101], [550, 132], [480, 115], [520, 125], # pts 7–12
    [460, 110], [530, 127], [400, 96],  [1200, 264],[1350, 270],[1100, 242],# pts 13–18
    [1400, 280],[1250, 266],[1300, 273],[1280, 269],[1380, 276],[1150, 253],# pts 19–24
    [1420, 284],[220, 44],  [250, 50],  [200, 40],  [680, 163], [720, 173] # pts 25–30
]

# ─────────────────────────────────────────────
# HELPER: Euclidean distance
# ─────────────────────────────────────────────
def euclidean(p, c):
    return math.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2)

# ─────────────────────────────────────────────
# HELPER: Assign each point to nearest centroid
# ─────────────────────────────────────────────
def assign(data, centroids):
    assignments = []
    for p in data:
        distances = [euclidean(p, c) for c in centroids]
        assignments.append(distances.index(min(distances)))
    return assignments

# ─────────────────────────────────────────────
# HELPER: Recompute centroids as cluster means
# ─────────────────────────────────────────────
def recompute(data, assignments, k):
    new_centroids = []
    for i in range(k):
        cluster = [data[j] for j in range(len(data)) if assignments[j] == i]
        mean_x = sum(p[0] for p in cluster) / len(cluster)
        mean_y = sum(p[1] for p in cluster) / len(cluster)
        new_centroids.append([round(mean_x, 2), round(mean_y, 2)])
    return new_centroids

# ─────────────────────────────────────────────
# HELPER: Compute WCSS
# ─────────────────────────────────────────────
def compute_wcss(data, assignments, centroids):
    total = 0
    for i, p in enumerate(data):
        c = centroids[assignments[i]]
        total += (p[0] - c[0])**2 + (p[1] - c[1])**2
    return total

# ─────────────────────────────────────────────
# STEP 1: ELBOW METHOD (k = 1 to 10)
# Using simplified seeding for reproducibility
# ─────────────────────────────────────────────
print("=" * 50)
print("ELBOW METHOD")
print("=" * 50)

wcss_values = []
for k in range(1, 11):
    # Seed centroids evenly across dataset for reproducibility
    step = len(data) // k
    centroids = [data[i * step] for i in range(k)]

    for _ in range(100):
        assignments = assign(data, centroids)
        new_centroids = recompute(data, assignments, k)
        if new_centroids == centroids:
            break
        centroids = new_centroids

    wcss = compute_wcss(data, assignments, centroids)
    wcss_values.append(wcss)
    print(f"  k={k:2d}  |  WCSS = {wcss:,.2f}")

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss_values, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=3, color='tomato', linestyle='--', label='Elbow at k=3')
plt.title('Elbow Method — Books Dataset')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.legend()
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# STEP 2: INITIAL CENTROIDS VIA K-MEANS++
# C1 = pt 15 (400, 96)   — medium
# C2 = pt 25 (1420, 284) — heavy/expensive
# C3 = pt 3  (100, 20)   — light/cheap
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("INITIAL CENTROIDS (K-MEANS++)")
print("=" * 50)

k = 3
centroids = [[400, 96], [1420, 284], [100, 20]]
labels = ['C1', 'C2', 'C3']
for i, c in enumerate(centroids):
    print(f"  {labels[i]} = {c}")

# ─────────────────────────────────────────────
# STEP 3: ITERATION 1
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("ITERATION 1")
print("=" * 50)
print(f"  {'#':<4} {'Weight':>8} {'Price':>8} {'→C1':>10} {'→C2':>10} {'→C3':>10} {'Cluster'}")
print("  " + "-" * 62)

assignments_iter1 = assign(data, centroids)
for i, p in enumerate(data):
    d = [euclidean(p, c) for c in centroids]
    cluster = labels[assignments_iter1[i]]
    print(f"  {i+1:<4} {p[0]:>8} {p[1]:>8} {d[0]:>10.2f} {d[1]:>10.2f} {d[2]:>10.2f} {cluster}")

wcss1 = compute_wcss(data, assignments_iter1, centroids)
print(f"\n  WCSS after Iteration 1: {wcss1:,.2f}")

# ─────────────────────────────────────────────
# STEP 4: UPDATE CENTROIDS
# ─────────────────────────────────────────────
new_centroids = recompute(data, assignments_iter1, k)
print("\n  Updated Centroids:")
for i, c in enumerate(new_centroids):
    print(f"    {labels[i]} = {c}")

# ─────────────────────────────────────────────
# STEP 5: ITERATION 2
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("ITERATION 2")
print("=" * 50)
print(f"  {'#':<4} {'Weight':>8} {'Price':>8} {'→C1':>10} {'→C2':>10} {'→C3':>10} {'Iter2':<8} {'Iter1':<8} {'Changed?'}")
print("  " + "-" * 78)

assignments_iter2 = assign(data, new_centroids)
flips = 0
for i, p in enumerate(data):
    d = [euclidean(p, c) for c in new_centroids]
    c2 = labels[assignments_iter2[i]]
    c1 = labels[assignments_iter1[i]]
    changed = "YES ←" if c2 != c1 else "—"
    if c2 != c1:
        flips += 1
    print(f"  {i+1:<4} {p[0]:>8} {p[1]:>8} {d[0]:>10.2f} {d[1]:>10.2f} {d[2]:>10.2f} {c2:<8} {c1:<8} {changed}")

wcss2 = compute_wcss(data, assignments_iter2, new_centroids)
print(f"\n  WCSS after Iteration 2: {wcss2:,.2f}")
print(f"  Points that changed cluster: {flips}")

# ─────────────────────────────────────────────
# STEP 6: CONVERGENCE CHECK
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("CONVERGENCE")
print("=" * 50)
if flips == 0:
    print("  No points changed clusters — algorithm has CONVERGED.")
else:
    print(f"  {flips} point(s) changed — further iterations needed.")

# ─────────────────────────────────────────────
# STEP 7: FINAL CLUSTERS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL CLUSTERS")
print("=" * 50)
cluster_names = ['Medium-weight, mid-priced books',
                 'Heavy, high-priced books',
                 'Lightweight, low-priced books']
for i in range(k):
    pts_in = [j+1 for j in range(len(data)) if assignments_iter2[j] == i]
    print(f"\n  {labels[i]} — {cluster_names[i]}")
    print(f"    Points: {pts_in}")
    print(f"    Final centroid: {new_centroids[i]}")

# ─────────────────────────────────────────────
# STEP 8: PLOT FINAL CLUSTERS
# ─────────────────────────────────────────────
colors     = ['#378ADD', '#1D9E75', '#D85A30']
markers    = ['o', 's', '^']
cluster_names = ['C1 — Medium (novels/textbooks)',
                 'C2 — Heavy (hardcovers/encyclopaedias)',
                 'C3 — Light (paperbacks/pocketbooks)']

plt.figure(figsize=(9, 6))

for i in range(k):
    pts_x = [data[j][0] for j in range(len(data)) if assignments_iter2[j] == i]
    pts_y = [data[j][1] for j in range(len(data)) if assignments_iter2[j] == i]
    pt_nums = [j+1 for j in range(len(data)) if assignments_iter2[j] == i]

    plt.scatter(pts_x, pts_y,
                color=colors[i], marker=markers[i],
                s=80, label=cluster_names[i], zorder=3)

    # Annotate each point with its row number
    for x, y, n in zip(pts_x, pts_y, pt_nums):
        plt.annotate(str(n), (x, y),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color=colors[i])

# Plot final centroids
for i, c in enumerate(new_centroids):
    plt.scatter(c[0], c[1],
                color=colors[i], marker='X',
                s=200, edgecolors='black', linewidths=0.8,
                zorder=5, label=f'{labels[i]} centroid {c}')

plt.title('K-Means Clustering — Books by Weight and Price (k=3)', fontsize=13)
plt.xlabel('Weight (g)')
plt.ylabel('Price (₱)')
plt.legend(fontsize=8, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()