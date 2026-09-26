"""
================================================================================
ORTHOGONALIZATION ANALYSIS: what does the measured block (X2) UNIQUELY add,
after the physics (M2) has already explained what it can?
================================================================================

WHERE THIS FITS
---------------
* Correlation step  -> found the blocks overlap a lot (pairwise view)
* PLS redundancy    -> found X2 is ~93% reconstructable from M2 (multivariate)
* THIS step         -> flips the question round: after M2 explains everything it
                       can, WHAT SURVIVES in X2, and is that survivor USEFUL for
                       predicting the response Y (purity)?

WHY IT MATTERS
--------------
SO-PLS's whole logic is: fit M2 first, then remove from X2 everything M2 already
explains (this removal is called ORTHOGONALIZATION), and let X2 contribute only
what is left. So "what survives orthogonalization" IS, by construction, the
unique information the sensors add. This is the sharpest possible instrument for
the shared-information question.

THE KEY ALGORITHMIC IDEA: ORTHOGONALIZATION
-------------------------------------------
Given scores T from the M2 model (T = the batch-summaries M2 extracted), we
remove from any block B the part explained by T:

        B_orthogonal = B - T @ (pinv(T) @ B)

The term  T @ pinv(T) @ B  is the projection of B onto the space spanned by T
(the "M2-explainable part"). Subtracting it leaves only the part ORTHOGONAL to
M2 - i.e. what M2 could NOT account for. pinv = Moore-Penrose pseudoinverse,
which solves the least-squares projection even when T's columns are collinear.

TWO THINGS WE MEASURE ON THE SURVIVOR
-------------------------------------
1. How much VARIANCE of each X2 variable survives (0% = fully explained by M2,
   100% = M2 explained nothing).
2. How much that survivor CORRELATES WITH Y - i.e. of the unique information,
   how much is actually USEFUL for predicting purity. (Surviving variance and
   USEFUL variance are different things, as this reveals.)

Plus a "WHEN" analysis: rebuild X2 using only the timepoints available up to each
decision point, and watch X2's contribution grow.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


# ==============================================================================
# 1. LOAD, SCALE, BLOCK-NORMALISE
# ==============================================================================
X1 = pd.read_excel("X1.xlsx")[['CA0', 'T1', 't1', 'A', 'B', 'C']]
M2 = pd.read_excel("M2.xlsx")
X2 = pd.read_excel("X2.xlsx")
Y = pd.read_excel("Y.xlsx")
species = ['A', 'B', 'C', 'D', 'E', 'F']
x2_cols = X2.columns.tolist()


def autoscale(A):
    A = A.astype(float)
    sd = A.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return (A - A.mean(0)) / sd


def block_norm(A):
    """
    After autoscaling, divide the whole block by sqrt(number of columns).
    WHY: X1 has 6 columns, M2 has 45, X2 has 42. Without this, the 45-column
    block would carry far more total variance into the model just for being
    bigger. Dividing by sqrt(ncols) gives each BLOCK comparable total weight, so
    the model balances them by information content, not by column count.
    """
    return autoscale(A) / np.sqrt(A.shape[1])


X1n, M2n, X2n = block_norm(X1.values), block_norm(M2.values), block_norm(X2.values)
Ys = autoscale(Y.values)


# ==============================================================================
# 2. THE SO-PLS ORTHOGONALIZATION, DONE STEP BY STEP
# ==============================================================================
# Latent-variable allocation {4, 6, 2}: 4 components for X1, 6 for M2, 2 for X2
# (the published configuration). We fit X1, then M2, orthogonalising the later
# blocks against each fitted block's scores, and finally read off the residual X2.
LV = [4, 6, 2]


def sopls_orthogonalize(blocks, Y, LV):
    """
    Sequentially fit each block against the (deflated) response and orthogonalise
    all LATER blocks against the current block's scores. Returns the final,
    orthogonalised blocks - in particular the X2 that has had X1 and M2 removed.
    """
    Xc = [b.copy() for b in blocks]     # working copies we will deflate
    Yc = Y.copy()
    for s in range(len(blocks)):
        pls = PLSRegression(LV[s], scale=False).fit(Xc[s], Yc)
        T = pls.transform(Xc[s])                 # scores: this block's batch-summaries
        T_pinv = np.linalg.pinv(T.T @ T) @ T.T   # tool for the projection

        # remove the part of every LATER block that this block's scores explain
        for n in range(s + 1, len(blocks)):
            G = T_pinv @ Xc[n]                   # regression of block n onto T
            Xc[n] = Xc[n] - T @ G                # subtract the explained part

        # also deflate the response, so the next block sees only the residual Y
        Yc = Yc - T @ (T_pinv @ Yc)
    return Xc


Xc = sopls_orthogonalize([X1n, M2n, X2n], Ys, LV)
X2_orth = Xc[2] * np.sqrt(X2.shape[1])    # undo the block-norm to compare to raw
X2_raw = autoscale(X2.values)


# ==============================================================================
# 3. HOW MUCH OF EACH X2 VARIABLE SURVIVES?
# ==============================================================================
# survivor variance / original variance, per column. Low = M2 explained it away
# (redundant); high = M2 could not account for it (unique).
surv = np.sum(X2_orth**2, axis=0) / np.sum(X2_raw**2, axis=0) * 100

print("===== UNIQUE VARIANCE SURVIVING ORTHOGONALIZATION (per species) =====")
print(f"{'species':<9}{'unique %':>10}{'shared %':>10}")
for sp in species:
    idx = [k for k, c in enumerate(x2_cols) if c.startswith(f'{sp}_')]
    u = surv[idx].mean()
    print(f"{sp:<9}{u:>10.1f}{100-u:>10.1f}")


# ==============================================================================
# 4. OF WHAT SURVIVES, HOW MUCH IS USEFUL FOR PREDICTING Y?
# ==============================================================================
# Correlate each orthogonalised X2 column with the response. A big survivor that
# does NOT correlate with Y is unique-but-useless (noise); one that DOES is the
# genuine correction the sensors provide.
corrY = np.array([abs(np.corrcoef(X2_orth[:, j], Ys.ravel())[0, 1])
                  for j in range(X2_orth.shape[1])])

order = np.argsort(corrY)[::-1]
print("\n===== TOP X2 VARIABLES BY |corr with Y| AFTER orthogonalization =====")
print("(what the sensors actually ADD to the purity prediction)")
print(f"{'variable':<10}{'survived %':>12}{'|corr with Y|':>15}")
for j in order[:8]:
    print(f"{x2_cols[j]:<10}{surv[j]:>12.1f}{corrY[j]:>15.3f}")

print("""
INTERPRETATION of sections 3-4
------------------------------
* Under the correct model only 3-14% of each sensor variable survives - the
  physics already contained the rest.
* What survives AND predicts purity is concentrated in species E (the response
  species itself) and F_1 (the initial impurity charge). When the model mis-
  predicts how much E formed, the measured E is the most direct evidence.
* Species C survives the most (least redundant) yet barely predicts Y: its
  residual is real but NOISE-dominated, because C's concentration does not
  itself set purity. UNIQUE variance is not the same as USEFUL variance.
""")


# ==============================================================================
# 5. WHEN does X2 step in?  (decision-point analysis)
# ==============================================================================
# Rebuild X2 using only the timepoints observed up to decision point k, and
# measure X2's contribution to explaining Y at each k. M2 is always full (a
# simulation is available from t=0).
def sopls_contrib(blocks, Y, LV):
    """Return each block's contribution to explaining Y (as % of total Y SS)."""
    Xc = [b.copy() for b in blocks]; Yc = Y.copy()
    total = np.sum((Y - Y.mean(0))**2); yhats = []
    for s in range(len(blocks)):
        lv = min(LV[s], Xc[s].shape[1])
        if lv < 1:
            yhats.append(np.zeros_like(Y)); continue
        pls = PLSRegression(lv, scale=False).fit(Xc[s], Yc)
        T = pls.transform(Xc[s]); Q = pls.y_loadings_
        yhats.append(T @ Q.T)
        Tp = np.linalg.pinv(T.T @ T) @ T.T
        for n in range(s + 1, len(blocks)):
            Xc[n] = Xc[n] - T @ (Tp @ Xc[n])
        Yc = Yc - T @ (Tp @ Yc)
    return [100 * np.sum(yh**2) / total for yh in yhats]


print("===== WHEN X2 STEPS IN  (contribution using only timepoints 1..k) =====")
print(f"{'samples':<10}{'X1 %':>7}{'M2 %':>8}{'X2 %':>8}")
for k in range(1, 8):
    cols = [c for c in x2_cols if int(c.split('_')[1]) <= k]
    X2k = block_norm(X2[cols].values)
    contrib = sopls_contrib([X1n, M2n, X2k], Ys, LV)
    print(f"t1..t{k:<7}{contrib[0]:>7.1f}{contrib[1]:>8.1f}{contrib[2]:>8.2f}")

print("""
INTERPRETATION of section 5
---------------------------
* Correct model: X2 says almost everything it has to say by the SECOND sample
  (jumps at t2, then crawls). You could stop sampling early at little cost.
* (Run the same on the misspecified model and X2 keeps rising all the way to t7,
  because the missing reaction keeps mattering as the batch proceeds - a temporal
  signature of a broken mechanism.)
""")


# ==============================================================================
# 6. HEATMAPS
# ==============================================================================
S = np.full((6, 7), np.nan); C = np.full((6, 7), np.nan)
for j, col in enumerate(x2_cols):
    i = species.index(col[0]); t = int(col.split('_')[1]) - 1
    S[i, t] = surv[j]; C[i, t] = corrY[j]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
im0 = ax[0].imshow(S, cmap='magma_r', vmin=0, vmax=15, aspect='auto')
ax[0].set_title('Unique variance surviving in X2 [%]')
im1 = ax[1].imshow(C, cmap='viridis', vmin=0, vmax=0.25, aspect='auto')
ax[1].set_title('|corr(survivor, Y)| = what X2 actually adds')
for a, M, fmt, vm in [(ax[0], S, '%.0f', 15), (ax[1], C, '%.2f', 0.25)]:
    a.set_xticks(range(7)); a.set_xticklabels([f't{i+1}' for i in range(7)])
    a.set_yticks(range(6)); a.set_yticklabels(species)
    a.set_xlabel('timepoint')
    for i in range(6):
        for j in range(7):
            a.text(j, i, fmt % M[i, j], ha='center', va='center', fontsize=8,
                   color='white' if M[i, j] > vm * 0.55 else 'black')
ax[0].set_ylabel('species')
plt.colorbar(im0, ax=ax[0]); plt.colorbar(im1, ax=ax[1])
plt.tight_layout()
plt.show()