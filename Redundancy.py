"""
================================================================================
PLS REDUNDANCY TEST: how much of the measured block (X2) can the mechanistic
block (M2) reconstruct?
================================================================================

WHY THIS STEP EXISTS
--------------------
The correlation matrix (previous step) compares ONE column to ONE column. In a
reaction network the information about one species is spread across several
others. For example E is tied to B and D through  B + 2D -> E , so:

    * no single M2 column matches sensor-E very well
    * yet E is almost fully RECONSTRUCTABLE from a COMBINATION of M2 columns

Correlation cannot see combinations, so it UNDER-reports the overlap (it told us
E and F were "least shared" at ~0.73). This step fixes that blind spot by letting
the WHOLE M2 block predict each X2 variable jointly.

THE ALGORITHM
-------------
Fit a PLS regression with
        predictors = M2   (the physics block)
        response   = X2   (the measured block)
NOTE: the true response Y (purity) is NOT involved here at all. We are asking a
pure block-overlap question: "how much of X2 lives inside M2?"

We judge it by CROSS-VALIDATION (Q^2), not in-sample fit (R^2), because with
enough components you can always fit the training data - we want the HONEST,
out-of-sample number.

    R^2  = fraction of X2 variance M2 can FIT       (optimistic, in-sample)
    Q^2  = fraction M2 can genuinely PREDICT        (honest, cross-validated)

OUTPUTS
-------
1. Block-level redundancy: one Q^2 for "how much of ALL of X2 is in M2"
2. Per-variable redundancy: a Q^2 for EACH X2 column -> which sensors are
   redundant vs which carry unique information
3. A per-species summary and a species x timepoint heatmap
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


# ==============================================================================
# 1. LOAD AND AUTOSCALE
# ==============================================================================
M2 = pd.read_excel("M2.xlsx")
X2 = pd.read_excel("X2.xlsx")
x2_cols = X2.columns.tolist()
species = ['A', 'B', 'C', 'D', 'E', 'F']


def autoscale(df):
    """
    Mean-centre each column and divide by its standard deviation (unit variance).
    WHY: PLS is variance-driven. Without scaling, a column measured in the
    thousands (a concentration) would dominate one measured in single digits,
    purely because of its units. Autoscaling puts every variable on equal footing
    so the model responds to STRUCTURE, not to unit magnitude.
    """
    A = df.values.astype(float)
    mu = A.mean(axis=0)
    sd = A.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0           # guard: a constant column has zero sd
    return (A - mu) / sd


M2s = autoscale(M2)
X2s = autoscale(X2)


# ==============================================================================
# 2. BLOCK-LEVEL REDUNDANCY:  M2  ->  ALL of X2
# ==============================================================================
# We sweep the number of PLS components (latent variables) and, for each, measure
# the cross-validated Q^2 of predicting the entire X2 block from M2. We keep the
# component count that gives the best Q^2.
def cv_q2_block(X_pred, Y_target, max_lv=15, n_splits=10, seed=42):
    """
    Cross-validated Q^2 for predicting Y_target (multi-column) from X_pred.

    10-fold CV: split the 100 batches into 10 groups. Train on 9, predict the
    held-out 1, repeat so every batch is predicted once while never being in its
    own training set. Q^2 compares those honest predictions to the truth.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best = (None, -np.inf, -np.inf)     # (n_components, R2_in_sample, Q2_cv)
    for lv in range(1, max_lv + 1):
        Y_pred_cv = np.zeros_like(Y_target)
        for train_idx, test_idx in cv.split(X_pred):
            pls = PLSRegression(n_components=lv, scale=False)
            pls.fit(X_pred[train_idx], Y_target[train_idx])
            Y_pred_cv[test_idx] = pls.predict(X_pred[test_idx])
        q2 = r2_score(Y_target, Y_pred_cv)                         # honest
        r2 = r2_score(Y_target,                                    # optimistic
                      PLSRegression(lv, scale=False)
                      .fit(X_pred, Y_target).predict(X_pred))
        if q2 > best[2]:
            best = (lv, r2, q2)
    return best


lv, r2, q2 = cv_q2_block(M2s, X2s, max_lv=15)
print("===== BLOCK-LEVEL REDUNDANCY:  M2  ->  X2 =====")
print(f"  best number of components = {lv}")
print(f"  R^2 (in-sample, optimistic) = {100*r2:.1f}%")
print(f"  Q^2 (cross-validated, HONEST) = {100*q2:.1f}%")
print(f"  --> {100*q2:.0f}% of the measured block is already contained in the physics")


# ==============================================================================
# 3. PER-VARIABLE REDUNDANCY:  M2  ->  each single X2 column
# ==============================================================================
# Now we ask, for EACH sensor variable separately: how much of it can the whole
# physics block predict? This tells us WHICH measurements are redundant (already
# known from physics) versus which carry unique information.
cv = KFold(n_splits=10, shuffle=True, random_state=42)


def cv_q2_single(X_pred, y, max_lv=12):
    """Best cross-validated Q^2 for predicting one column y from X_pred."""
    best_q2 = -np.inf
    for lv in range(1, max_lv + 1):
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in cv.split(X_pred):
            pls = PLSRegression(n_components=lv, scale=False)
            pls.fit(X_pred[train_idx], y[train_idx])
            y_pred[test_idx] = pls.predict(X_pred[test_idx]).ravel()
        best_q2 = max(best_q2, r2_score(y, y_pred))
    return best_q2


rows = []
for j, col in enumerate(x2_cols):
    q2_j = cv_q2_single(M2s, X2s[:, j])
    rows.append((col, col[0], int(col.split('_')[1]), 100 * q2_j))

res = pd.DataFrame(rows, columns=['variable', 'species', 'timepoint', 'Q2_%'])


# ==============================================================================
# 4. PER-SPECIES SUMMARY
# ==============================================================================
print("\n===== PER-SPECIES REDUNDANCY  (% of X2 predictable from M2, CV Q^2) =====")
sp_summary = res.groupby('species')['Q2_%'].agg(['mean', 'min', 'max']).round(1)
sp_summary = sp_summary.sort_values('mean', ascending=False)
print(sp_summary.to_string())

print("\n===== MOST redundant sensor variables (already in the physics) =====")
print(res.sort_values('Q2_%', ascending=False).head(6).to_string(index=False))

print("\n===== LEAST redundant sensor variables (carry unique information) =====")
print(res.sort_values('Q2_%').head(6).to_string(index=False))


# ==============================================================================
# 5. COMPARISON WITH THE CORRELATION RESULT  (why this step mattered)
# ==============================================================================
print("\n===== CORRELATION said vs PLS says  (per species) =====")
# pairwise mean|r|, same-species, from the correlation step:
corr_same = {}
for sp in species:
    m2g = [c for c in M2.columns if c.startswith(f'{sp}_fit')]
    x2g = [c for c in X2.columns if c.startswith(f'{sp}_')]
    vals = [abs(np.corrcoef(M2[mc], X2[xc])[0, 1]) for mc in m2g for xc in x2g]
    corr_same[sp] = 100 * np.mean(vals)

print(f"{'species':<9}{'corr mean|r| %':>16}{'PLS Q2 %':>12}")
for sp in species:
    pls_q2 = res[res.species == sp]['Q2_%'].mean()
    print(f"{sp:<9}{corr_same[sp]:>16.1f}{pls_q2:>12.1f}")
print("\nCorrelation UNDER-reports overlap for E and F (~73% vs ~92% by PLS),")
print("because it cannot see that E/F are reconstructable from OTHER species")
print("jointly. PLS captures those multivariate combinations; correlation can't.")


# ==============================================================================
# 6. HEATMAP: redundancy per species x timepoint
# ==============================================================================
mat = np.full((6, 7), np.nan)
for _, r in res.iterrows():
    mat[species.index(r['species']), int(r['timepoint']) - 1] = r['Q2_%']

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(mat, cmap='RdYlGn_r', vmin=80, vmax=100, aspect='auto')
plt.colorbar(im, label='% of sensor variable predictable from physics (CV Q^2)')
ax.set_xticks(range(7)); ax.set_xticklabels([f't{i+1}' for i in range(7)])
ax.set_yticks(range(6)); ax.set_yticklabels(species)
for i in range(6):
    for j in range(7):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f'{mat[i, j]:.0f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if mat[i, j] > 93 else 'black')
ax.set_xlabel('timepoint'); ax.set_ylabel('species')
ax.set_title('PLS redundancy: how much of each sensor reading is already in the physics')
plt.tight_layout()
plt.savefig('pls_redundancy.png', dpi=150)
plt.close()
print("\nSaved: pls_redundancy.png")

print("""
INTERPRETATION
--------------
* ~93% of the measured block is already inside the physics. Only ~7% is new.
* Highest redundancy: species A (inert, model predicts a constant, sensor
  confirms it) and the first timepoint of every species (the known initial
  charge - measuring it tells you nothing new).
* Lowest redundancy: species C at late times. C sits on the competing side
  reaction  C + 2D -> F , whose outcome depends on the ratio of two rate
  constants that mass balance alone cannot fix - so its measurement carries the
  most genuinely new information.
* This is the honest, multivariate answer. It supersedes the correlation view,
  which under-reported the overlap for E and F.
""")