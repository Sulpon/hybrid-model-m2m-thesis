"""
Run the EXACT SAME SO-PLS CV grid search used throughout this session
(_fit/_scale_fit/_scale_apply/_resid, cv_surface with A_MAX=6,B_MAX=15,
C_MAX=15, 10-fold KFold, seed=42) on the REAL author data (X1.xlsx/
X2.xlsx/M2.xlsx/Y.xlsx for case i, the _mis files for case ii), to check
whether it reproduces the user's own found optimum ((4,6,4) case i,
(3,3,6) case ii) or the paper's stated {4,6,2}/{2,3,6}.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42

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

def cv_surface(X1, M2, X2, Y):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y)
    P_abc = np.zeros((n, A_MAX, B_MAX, C_MAX))
    rf = np.zeros((N_SPLITS, A_MAX, B_MAX, C_MAX))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muX1, sdX1, k1 = _scale_fit(X1[tr]); x1t, x1e = _scale_apply(X1[tr], muX1, sdX1, k1), _scale_apply(X1[te], muX1, sdX1, k1)
        muM2, sdM2, k2 = _scale_fit(M2[tr]); m2t0, m2e0 = _scale_apply(M2[tr], muM2, sdM2, k2), _scale_apply(M2[te], muM2, sdM2, k2)
        muX2, sdX2, k3 = _scale_fit(X2[tr]); x2t0, x2e0 = _scale_apply(X2[tr], muX2, sdX2, k3), _scale_apply(X2[te], muX2, sdX2, k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr] - muY) / sdy

        p1, T1t, Q1, af = _fit(x1t, Ytr0, A_MAX); T1e = p1.transform(x1e)
        for a in range(1, A_MAX + 1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            m2t, gM = _resid(Tat, m2t0); m2e = m2e0 - Tae @ gM
            x2t_a, gX = _resid(Tat, x2t0); x2e_a = x2e0 - Tae @ gX
            Ytr_a, gY = _resid(Tat, Ytr0)
            p2, T2t, Q2, bf = _fit(m2t, Ytr_a, B_MAX); T2e = p2.transform(m2e)
            for b in range(1, B_MAX + 1):
                be = min(b, bf); Tbt, Tbe = T2t[:, :be], T2e[:, :be]
                yh_ab = yh_a + Tbe @ Q2[:, :be].T
                x2t_b, gX2 = _resid(Tbt, x2t_a); x2e_b = x2e_a - Tbe @ gX2
                Ytr_b, gY2 = _resid(Tbt, Ytr_a)
                p3, T3t, Q3, cf = _fit(x2t_b, Ytr_b, C_MAX); T3e = p3.transform(x2e_b)
                for c in range(1, C_MAX + 1):
                    ce = min(c, cf)
                    yh = (yh_ab + T3e[:, :ce] @ Q3[:, :ce].T) * sdy + muY
                    P_abc[te, a - 1, b - 1, c - 1] = yh.ravel()
                    rf[f, a - 1, b - 1, c - 1] = np.sqrt(np.mean((yh.ravel() - Y[te].ravel()) ** 2))
    sst = np.sum((Y - Y.mean(0)) ** 2)
    qabc = 1 - ((P_abc - Y[:, :, None, None]) ** 2).sum(0) / sst
    rmsecv = rf.mean(0)
    return rmsecv, qabc

def run_case(label, x1f, x2f, m2f, yf):
    X1 = pd.read_excel(x1f)[['CA0','T1','t1','A','B','C']].values
    X2 = pd.read_excel(x2f).values
    M2df = pd.read_excel(m2f)
    m2_cols = [c for c in M2df.columns if c not in ('t2','T2','D0')] + ['t2','T2','D0']
    M2 = M2df[m2_cols].values
    Y = pd.read_excel(yf)['E_pur'].values.reshape(-1, 1)
    print(f"{label}: X1{X1.shape} M2{M2.shape} X2{X2.shape} Y{Y.shape}")

    rmsecv, qabc = cv_surface(X1, M2, X2, Y)
    i = np.unravel_index(rmsecv.argmin(), rmsecv.shape)
    best = (i[0]+1, i[1]+1, i[2]+1)
    print(f"{label}: MY CODE min-RMSECV LV = {best}  RMSECV={rmsecv[i]:.4f}  Q2={qabc[i]:.4f}")
    return rmsecv, qabc, best

print("===== CASE (i): well-specified, real author data =====")
rmsecv_i, qabc_i, best_i = run_case('case_i', 'X1.xlsx', 'X2.xlsx', 'M2.xlsx', 'Y.xlsx')

print("\n===== CASE (ii): misspecified, real author data =====")
rmsecv_ii, qabc_ii, best_ii = run_case('case_ii', 'X1_mis.xlsx', 'X2_mis.xlsx', 'M2_mis.xlsx', 'Y_mis.xlsx')

print("\n===== COMPARISON =====")
for label, rmsecv, best, user_lv, paper_lv in [
    ('case (i)', rmsecv_i, best_i, (4,6,4), (4,6,2)),
    ('case (ii)', rmsecv_ii, best_ii, (3,3,6), (2,3,6)),
]:
    r_my = rmsecv[best[0]-1, best[1]-1, best[2]-1]
    r_user = rmsecv[user_lv[0]-1, user_lv[1]-1, user_lv[2]-1]
    r_paper = rmsecv[paper_lv[0]-1, paper_lv[1]-1, paper_lv[2]-1]
    print(f"{label}: my-code-optimum={best} RMSECV={r_my:.4f}   "
          f"at user's LV{user_lv} RMSECV={r_user:.4f}   at paper's LV{paper_lv} RMSECV={r_paper:.4f}")
