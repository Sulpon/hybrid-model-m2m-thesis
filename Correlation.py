"""
================================================================================
CORRELATION ANALYSIS: shared information between the mechanistic block (M2)
and the measured block (X2)
================================================================================

PURPOSE
-------
SO-PLS processes M2 (physics) before X2 (sensors) and gives M2 first claim on
explaining the response. That credit split is only meaningful if the two blocks
carry DIFFERENT information. This script measures how much they OVERLAP, as a
first, cheap screen before the multivariate PLS-redundancy test.

WHAT IT PRODUCES
----------------
1. Full 45 x 42 Pearson correlation matrix (every M2 column vs every X2 column)
2. A flagged list of strongly correlated variable pairs
3. A condensed 6 x 6 species-level summary (mean |r| per species pair)
4. A per-species same-vs-other overlap table
5. Effective-rank check (how many independent dimensions each M2 species block
   really has)
6. Two heatmaps saved to PNG"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
# Both files: 100 rows (batches) x columns.
#   M2 columns: A_fit_1..A_fit_7, B_fit_1..., ... F_fit_7, plus t2, T2, D0
#   X2 columns: A_1..A_7, B_1..., ... F_7
M2 = pd.read_excel("M2.xlsx")
X2 = pd.read_excel("X2.xlsx")

m2_cols = M2.columns.tolist()
x2_cols = X2.columns.tolist()
species = ['A', 'B', 'C', 'D', 'E', 'F']

print(f"M2: {M2.shape[0]} batches x {M2.shape[1]} columns")
print(f"X2: {X2.shape[0]} batches x {X2.shape[1]} columns")


# ==============================================================================
# 2. PEARSON CORRELATION  (the core definition, done by hand once for clarity)
# ==============================================================================
def pearson(x, y):
    """
    Pearson correlation coefficient between two 1-D arrays (one value per batch).

        r = sum((x-xbar)(y-ybar)) / sqrt(sum((x-xbar)^2) * sum((y-ybar)^2))

    Range [-1, 1]:
       +1  perfectly in step   |  0  unrelated  |  -1  perfect mirror
    It is SCALE-FREE: subtracting the mean and dividing by the spreads means the
    different units of M2 (fitted values) and X2 (raw concentrations) don't matter.
    Only CO-MOVEMENT across batches is measured.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    if denom == 0:            # a constant column has zero spread -> undefined
        return 0.0
    return np.sum(xc * yc) / denom

# (numpy's np.corrcoef(x, y)[0, 1] computes the same thing; we use it below for speed)


# ==============================================================================
# 3. FULL 45 x 42 CORRELATION MATRIX
# ==============================================================================
# rows = M2 variables, cols = X2 variables
n_m2, n_x2 = len(m2_cols), len(x2_cols)
corr_matrix = np.zeros((n_m2, n_x2))

for i in range(n_m2):
    for j in range(n_x2):
        # np.corrcoef returns a 2x2 matrix; [0,1] is the cross term (x with y)
        corr_matrix[i, j] = np.corrcoef(M2.iloc[:, i], X2.iloc[:, j])[0, 1]

corr_df = pd.DataFrame(corr_matrix, index=m2_cols, columns=x2_cols)
print("\n===== FULL M2 vs X2 CORRELATION MATRIX (rounded) =====")
print(corr_df.round(2).to_string())


# ==============================================================================
# 4. FLAG STRONG PAIRWISE RELATIONSHIPS
# ==============================================================================
threshold = 0.7
print(f"\n===== VARIABLE PAIRS WITH |r| > {threshold} =====")
strong = []
for i in range(n_m2):
    for j in range(n_x2):
        r = corr_matrix[i, j]
        if abs(r) > threshold:
            strong.append((m2_cols[i], x2_cols[j], r))
            print(f"  {m2_cols[i]:<10} <-> {x2_cols[j]:<8}  r = {r:+.3f}")
if not strong:
    print("  none above threshold")


# ==============================================================================
# 5. SPECIES-LEVEL SUMMARY  (condense the 45x42 grid to a readable 6x6)
# ==============================================================================
# For each (M2 species, X2 species) pair there are 7x7 = 49 individual r-values.
# We summarise them two ways:
#
#   mean(|r|)  -> RELATIONSHIP STRENGTH, ignoring direction.
#                THE RIGHT CHOICE for a redundancy question, because SO-PLS can
#                exploit a negative correlation just as well as a positive one.
#
#   mean(r)    -> NET DIRECTION. Positives and negatives cancel. Useful only for
#                physical interpretation, NOT for measuring overlap. (In this
#                reactor, physics-D vs sensor-E is +ve early and -ve late, so the
#                signed mean hides a real relationship the |r| mean reveals.)
mean_abs = np.zeros((6, 6))
mean_signed = np.zeros((6, 6))

for i, sp_m2 in enumerate(species):
    m2_group = [c for c in m2_cols if c.startswith(f'{sp_m2}_fit')]
    for j, sp_x2 in enumerate(species):
        x2_group = [c for c in x2_cols if c.startswith(f'{sp_x2}_')]
        vals = [np.corrcoef(M2[mc], X2[xc])[0, 1]
                for mc in m2_group for xc in x2_group]   # 49 correlations
        vals = np.array(vals)
        mean_abs[i, j] = np.mean(np.abs(vals))
        mean_signed[i, j] = np.mean(vals)

abs_df = pd.DataFrame(mean_abs,
                      index=[f'M2:{s}' for s in species],
                      columns=[f'X2:{s}' for s in species])
print("\n===== SPECIES-LEVEL SHARED INFORMATION  (mean |r|) =====")
print(abs_df.round(3).to_string())
print("\n(Diagonal = same-species overlap; off-diagonal = cross-species leakage)")


# ==============================================================================
# 6. PER-SPECIES OVERLAP TABLE  (same-species diagonal, in detail)
# ==============================================================================
print("\n===== SAME-SPECIES OVERLAP (M2 fit vs X2 raw) =====")
print(f"{'species':<9}{'mean|r|':>9}{'max|r|':>9}{'min|r|':>9}")
for i, sp in enumerate(species):
    m2_group = [c for c in m2_cols if c.startswith(f'{sp}_fit')]
    x2_group = [c for c in x2_cols if c.startswith(f'{sp}_')]
    vals = np.array([abs(np.corrcoef(M2[mc], X2[xc])[0, 1])
                     for mc in m2_group for xc in x2_group])
    print(f"{sp:<9}{vals.mean():>9.3f}{vals.max():>9.3f}{vals.min():>9.3f}")


# ==============================================================================
# 7. EFFECTIVE RANK  (why correlation isn't the whole story)
# ==============================================================================
# The 7 timepoints of a species are a smooth trajectory, so they are highly
# correlated WITH EACH OTHER -> the block has far fewer than 7 independent
# dimensions. We measure this with the singular values (like PCA): count how many
# are above 1% of the largest. This is a preview of why the multivariate PLS test
# (next script) finds MORE redundancy than pairwise correlation does.
print("\n===== EFFECTIVE RANK OF EACH M2 SPECIES BLOCK (of 7) =====")
for sp in species:
    block = M2[[c for c in m2_cols if c.startswith(f'{sp}_fit')]].values
    block_centered = block - block.mean(0)
    s = np.linalg.svd(block_centered, compute_uv=False)   # singular values
    s_norm = s / s[0]                                      # relative to largest
    eff_rank = int(np.sum(s_norm > 0.01))
    print(f"  {sp}_fit: effective rank = {eff_rank}/7   "
          f"(singular values: {np.round(s_norm, 3)})")


# ==============================================================================
# 8. HEATMAP 1 — full 45 x 42 matrix
# ==============================================================================
plt.figure(figsize=(max(8, n_x2 * 0.28), max(6, n_m2 * 0.24)))
im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, label="Pearson r")
plt.xticks(range(n_x2), x2_cols, rotation=90, fontsize=6)
plt.yticks(range(n_m2), m2_cols, fontsize=6)
plt.xlabel("X2 variables (measured)")
plt.ylabel("M2 variables (physics)")
plt.title("Full correlation: M2 (physics) vs X2 (measured)")
plt.tight_layout()
plt.savefig("correlation_full.png", dpi=150)
plt.close()

# ==============================================================================
# 9. HEATMAP 2 — condensed species-level (mean |r|)
# ==============================================================================
plt.figure(figsize=(7, 6))
im = plt.imshow(mean_abs, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="mean |Pearson r|")
plt.xticks(range(6), [f'X2:{s}' for s in species])
plt.yticks(range(6), [f'M2:{s}_fit' for s in species])
for i in range(6):
    for j in range(6):
        val = mean_abs[i, j]
        plt.text(j, i, f'{val:.2f}', ha='center', va='center',
                 color='white' if val < 0.5 else 'black',
                 fontsize=11, fontweight='bold')
plt.title("Species-level shared information: M2 (physics) vs X2 (measured)")
plt.tight_layout()
plt.savefig("correlation_species.png", dpi=150)
plt.close()

print("\nSaved: correlation_full.png, correlation_species.png")
print("\nNOTE: correlation only compares ONE column to ONE column. It cannot see")
print("information spread across SEVERAL columns (e.g. E recoverable from B and D")
print("jointly). That blind spot is why the next step is a multivariate PLS")
print("redundancy test, which finds MORE overlap than this pairwise view.")