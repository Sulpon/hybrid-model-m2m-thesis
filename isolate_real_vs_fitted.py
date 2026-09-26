"""
Isolate which change actually mattered for M0 landing on the paper's (4,6,2):
using the REAL X1.xlsx/X2.xlsx/Y.xlsx directly (vs a regenerated dataset),
or FITTING M2's kinetic parameters (vs fixing them to the true nominal
values). The last successful run changed BOTH at once. This tests the
missing combination: real data + FIXED nominal parameters (no fitting).
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

R = 8.314
Ea3_true, Ea4_true, A3_true, A4_true = 50000., 55000., 10., 20.
A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42
species = ['A', 'B', 'C', 'D', 'E', 'F']

def ode_correct(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB*CD**2; r4 = k4*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]

X1a = pd.read_excel('X1.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
X2a = pd.read_excel('X2.xlsx')
Ya = pd.read_excel('Y.xlsx')['E_pur'].values.reshape(-1, 1)
M2a_meta = pd.read_excel('M2.xlsx')[['t2', 'T2', 'D0']]
n_pts = X2a.shape[1] // 6
N = len(X1a)
tgrid = np.arange(n_pts) * 15.0
T2v = M2a_meta['T2'].values

IC = np.array([[X2a[f'{s}_1'].iloc[i] for s in species] for i in range(N)])
X2flat = np.array([[X2a[f'{s}_{k+1}'].iloc[i] for s in species for k in range(n_pts)] for i in range(N)])

print("Building M0's M2 from REAL data with FIXED nominal parameters (no fitting)...")
M2 = np.zeros((N, 6, n_pts))
for i in range(N):
    sol = odeint(ode_correct, IC[i], tgrid, args=(T2v[i], Ea3_true, Ea4_true, A3_true, A4_true))
    M2[i] = sol.T
M2 = M2.reshape(N, -1)
M2 = np.hstack([M2, M2a_meta.values])

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0] - 1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n
def _scale_fit(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    k = np.sqrt(B.shape[1]); return mu, sd, k
def _scale_apply(B, mu, sd, k): return (B - mu) / sd / k
def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g

def cv_3block(X1, M2, X2, Y):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y)
    rf = np.zeros((N_SPLITS, A_MAX, B_MAX, C_MAX))
    P = np.zeros((n, A_MAX, B_MAX, C_MAX))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muX1, sdX1, k1 = _scale_fit(X1[tr]); x1t, x1e = _scale_apply(X1[tr],muX1,sdX1,k1), _scale_apply(X1[te],muX1,sdX1,k1)
        muM2, sdM2, k2 = _scale_fit(M2[tr]); m2t0, m2e0 = _scale_apply(M2[tr],muM2,sdM2,k2), _scale_apply(M2[te],muM2,sdM2,k2)
        muX2, sdX2, k3 = _scale_fit(X2[tr]); x2t0, x2e0 = _scale_apply(X2[tr],muX2,sdX2,k3), _scale_apply(X2[te],muX2,sdX2,k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(x1t, Ytr0, A_MAX); T1e = p1.transform(x1e)
        for a in range(1, A_MAX+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            m2t, gM = _resid(Tat, m2t0); m2e = m2e0 - Tae @ gM
            x2t_a, gX = _resid(Tat, x2t0); x2e_a = x2e0 - Tae @ gX
            Ytr_a, gY = _resid(Tat, Ytr0)
            p2, T2t, Q2, bf = _fit(m2t, Ytr_a, B_MAX); T2e = p2.transform(m2e)
            for b in range(1, B_MAX+1):
                be = min(b, bf); Tbt, Tbe = T2t[:,:be], T2e[:,:be]
                yh_ab = yh_a + Tbe @ Q2[:,:be].T
                x2t_b, gX2 = _resid(Tbt, x2t_a); x2e_b = x2e_a - Tbe @ gX2
                Ytr_b, gY2 = _resid(Tbt, Ytr_a)
                p3, T3t, Q3, cf = _fit(x2t_b, Ytr_b, C_MAX); T3e = p3.transform(x2e_b)
                for c in range(1, C_MAX+1):
                    ce = min(c, cf)
                    yh = (yh_ab + T3e[:,:ce] @ Q3[:,:ce].T)*sdy+muY
                    P[te,a-1,b-1,c-1] = yh.ravel()
                    rf[f,a-1,b-1,c-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    qabc = 1 - ((P-Y[:,:,None,None])**2).sum(0)/sst
    return rf.mean(0), qabc

print("Running blind-argmin CV search (real data, FIXED nominal M2 params)...")
rmsecv, qabc = cv_3block(X1a, M2, X2flat, Ya)
i_best = np.unravel_index(rmsecv.argmin(), rmsecv.shape)
best = (i_best[0]+1, i_best[1]+1, i_best[2]+1)
print(f"  best LV = {best}  RMSECV={rmsecv[i_best]:.4f}  Q2={qabc[i_best]:.4f}")
print(f"  paper's LV(4,6,2): RMSECV={rmsecv[3,5,1]:.4f}  Q2={qabc[3,5,1]:.4f}")
