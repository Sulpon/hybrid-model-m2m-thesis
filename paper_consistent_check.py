"""
Paper-consistent LV allocation check for M0 (case i, well-specified) and
M6 (case ii, misspecified), per the M2M paper's own reported allocations:
  Case (i)  [well-specified]: {LV_X1=4, LV_M2=6, LV_X2=2}
  Case (ii) [misspecified]:   {LV_X1=2, LV_M2=3, LV_X2=6}
(M2M paper, Section 5.2.1/5.2.2, Table 7 area of the extracted text.)

These are the paper's allocations for ITS OWN real X1/X2/M2/M2_mis data --
NOT necessarily optimal for our re-simulated calibration set. Applied here
to our own calibration data (X1c, M2c for M0/M6, X2c) purely to test
whether the qualitative ablation pattern (well-specified relies more on M2,
misspecified relies more on X2) reproduces structurally.

Q2_FULL is read directly from the existing full_grid (LV_allocation_analysis.xlsx)
at the exact paper LV triple. Q2_NO_M2/Q2_NO_X2 are computed fresh (2-block
CV, same _fit/_scale/10-fold/seed=42 as every prior script) ONLY at the two
paper LV pairs needed -- no independent optimization.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

N_SPLITS, SEED = 10, 42

X1_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X1')
Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')
CAL_X2_COLS = list(X2_cal.columns); CAL_X1_COLS = list(X1_cal.columns)
X1c = X1_cal[CAL_X1_COLS].values
X2c = X2_cal.values

def load_M2_cal(name): return pd.read_excel(f"M2_{name}.xlsx", sheet_name='calibration')[CAL_X2_COLS].values

full_grid = pd.read_excel('LV_allocation_analysis.xlsx', sheet_name='full_grid')

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0] - 1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n

def _scale_fit(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    k = np.sqrt(B.shape[1])
    return mu, sd, k

def _scale_apply(B, mu, sd, k):
    return (B - mu) / sd / k

def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g

def cv_2block_at(Xa, Xb, Y, na, nb):
    """Q2 of the 2-block SO-PLS Xa->Xb at EXACTLY na, nb components (no scan)."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, 1))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muXa, sdXa, ka = _scale_fit(Xa[tr]); xat, xae = _scale_apply(Xa[tr], muXa, sdXa, ka), _scale_apply(Xa[te], muXa, sdXa, ka)
        muXb, sdXb, kb = _scale_fit(Xb[tr]); xbt0, xbe0 = _scale_apply(Xb[tr], muXb, sdXb, kb), _scale_apply(Xb[te], muXb, sdXb, kb)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr0 = (Y[tr]-muY)/sdy
        p1, Tat, Qa, af = _fit(xat, ytr0, na); ae = min(na, af); Tat, Tae = Tat[:, :ae], p1.transform(xae)[:, :ae]
        yh_a = Tae @ Qa[:, :ae].T
        xbt, gXb = _resid(Tat, xbt0); xbe = xbe0 - Tae @ gXb
        ytr_a, gY = _resid(Tat, ytr0)
        p2, Tbt, Qb, bf = _fit(xbt, ytr_a, nb); be = min(nb, bf); Tbe = p2.transform(xbe)[:, :be]
        yh = (yh_a + Tbe @ Qb[:, :be].T) * sdy + muY
        P[te] = yh
    sst = np.sum((Y - Y.mean(0))**2)
    return float(1 - ((P - Y)**2).sum() / sst)

PAPER_LV = {
    'M0_correct':    dict(LV_X1=4, LV_M2=6, LV_X2=2),   # paper case (i), well-specified
    'M6_author_mis': dict(LV_X1=2, LV_M2=3, LV_X2=6),   # paper case (ii), misspecified
}
OUR_OPT_LV = {
    'M0_correct':    dict(LV_X1=4, LV_M2=2, LV_X2=9),
    'M6_author_mis': dict(LV_X1=5, LV_M2=10, LV_X2=4),
}

rows = []
for name, lv in PAPER_LV.items():
    a, b, c = lv['LV_X1'], lv['LV_M2'], lv['LV_X2']
    M2c = load_M2_cal(name)

    match = full_grid[(full_grid['model']==name) & (full_grid['LV_X1']==a) & (full_grid['LV_M2']==b) & (full_grid['LV_X2']==c)]
    q2_full = float(match['Q2'].iloc[0]); rmsecv_full = float(match['RMSECV'].iloc[0])

    q2_no_m2 = cv_2block_at(X1c, X2c, Y_cal, a, c)     # X1 -> X2, using paper's LV_X1, LV_X2
    q2_no_x2 = cv_2block_at(X1c, M2c, Y_cal, a, b)     # X1 -> M2, using paper's LV_X1, LV_M2

    d_m2 = q2_full - q2_no_m2
    d_x2 = q2_full - q2_no_x2
    amr = abs(d_m2) / abs(d_x2) if abs(d_x2) > 1e-12 else np.nan

    opt = OUR_OPT_LV[name]
    rows.append(dict(model=name, paper_LV=(a, b, c), our_min_RMSECV_LV=(opt['LV_X1'], opt['LV_M2'], opt['LV_X2']),
                      ablation_comparison_LV=(a, b, c), RMSECV_FULL_at_paperLV=rmsecv_full,
                      Q2_FULL=q2_full, Q2_NO_M2=q2_no_m2, Q2_NO_X2=q2_no_x2,
                      dQ2_drop_M2_pp=d_m2*100, dQ2_drop_X2_pp=d_x2*100, AMR=amr))
    print(f"{name}: paper LV={a,b,c}  Q2_FULL={q2_full:.4f}  Q2_NO_M2={q2_no_m2:.4f}  Q2_NO_X2={q2_no_x2:.4f}  "
          f"dQ2_M2={d_m2*100:.2f}pp  dQ2_X2={d_x2*100:.2f}pp  AMR={amr:.3f}")

result_df = pd.DataFrame(rows)
print("\n", result_df.to_string(index=False))
