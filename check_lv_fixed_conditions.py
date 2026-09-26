"""
Build full X1/M2/X2/Y blocks for case (ii) using the REAL, exact operating
conditions (fed in, not drawn fresh) -- the approach just validated to give
much better correlation (0.74-0.89) with the real X2_mis.xlsx/Y_mis.xlsx than
any previous fresh-random-condition attempt. Then run the SAME validated
blind-argmin SO-PLS CV search and compare the found LV allocation to the
paper's {2,3,6}. Also re-confirms case (i)'s allocation (already validated
with fresh seed=10, which exactly matches real conditions for case i).
"""
import numpy as np
import pandas as pd
import sdeint
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

R = 8.314
V1, V2 = 0.1, 0.2
A1, A2, A3, A4 = 28, 40, 10, 20
Ea1_m, Ea2_m, Ea3_m, Ea4_m = 20000, 30000, 50000, 55000
CV = 0.01
sigma1 = [0.75, 0.75, 0.25]
sigma2 = [0.25, 0.75, 0.25, 0.75, 0.75, 0.25]
species = ['A', 'B', 'C', 'D', 'E', 'F']
A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42

def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    return np.array([-k1*CA, k1*CA - k2*CB, k2*CB])
def stage1_noise(y, t): return np.diag(sigma1)
def stage2_drift(y, t, k3, k4):
    CA, CB, CC, CD, CE, CF = y
    return np.array([0, -k3*CB*CD**2, -k4*CC*CD**2, -2*k3*CB*CD**2-2*k4*CC*CD**2, k3*CB*CD**2, k4*CC*CD**2])
def stage2_noise(y, t): return np.diag(sigma2)

# =====================================================================
# STEP 1: case (ii) -- regenerate using REAL fixed conditions
# =====================================================================
X1m = pd.read_excel('X1_mis.xlsx'); M2m_real = pd.read_excel('M2_mis.xlsx')
N = len(X1m)

print("Regenerating case (ii) with REAL fixed conditions + fresh SDE noise...")
np.random.seed(10)
all_s1, all_s2, conds = [], [], []
for i in range(N):
    CA0s, T1, t1min = X1m['CA0'].iloc[i], X1m['T1'].iloc[i], X1m['t1'].iloc[i]
    T2, CD0s, t2min = M2m_real['T2'].iloc[i], M2m_real['D0'].iloc[i], M2m_real['t2'].iloc[i]
    t1s, t2s = t1min*60, t2min*60
    Ea1 = np.random.normal(Ea1_m, CV*Ea1_m); Ea2 = np.random.normal(Ea2_m, CV*Ea2_m)
    Ea3 = np.random.normal(Ea3_m, CV*Ea3_m); Ea4 = np.random.normal(Ea4_m, CV*Ea4_m)
    k1 = A1*np.exp(-Ea1/(R*T1)); k2 = A2*np.exp(-Ea2/(R*T1))
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))

    t1v = np.linspace(0, t1s, 1000)
    sol1 = sdeint.itoint(lambda y,t: stage1_drift(y,t,k1,k2), stage1_noise, np.array([CA0s,0.,0.]), t1v)
    CA1f, CB1f, CC1f = sol1[-1]
    d = V1/V2
    y0 = np.array([CA1f*d, CB1f*d, CC1f*d, CD0s, 0., 0.])
    t2v = np.linspace(0, t2s, 1000)
    sol2 = sdeint.itoint(lambda y,t: stage2_drift(y,t,k3,k4), stage2_noise, y0, t2v)
    all_s1.append((t1v/60, sol1)); all_s2.append((t2v/60, sol2))
    conds.append(dict(CA0=CA0s, T1=T1, t1=t1min, T2=T2, D0=CD0s, t2=t2min))

def discretise(all_s1, all_s2, noise_pct=0.0035, dt=0.25):
    mf1 = np.mean([s1[-1] for _, s1 in all_s1], axis=0); n1 = noise_pct*mf1
    mf2 = np.mean([s2[-1] for _, s2 in all_s2], axis=0); n2 = noise_pct*mf2
    d1, d2 = [], []
    for t1v, s1 in all_s1:
        tk1 = np.arange(0, t1v[-1], dt)
        out = np.column_stack([np.maximum(0, np.interp(tk1, t1v, s1[:, j]) + np.random.normal(0, n1[j], len(tk1))) for j in range(3)])
        d1.append((tk1, out))
    for t2v, s2 in all_s2:
        tk2 = np.arange(0, t2v[-1], dt)
        out = np.column_stack([np.maximum(0, np.interp(tk2, t2v, s2[:, j]) + np.random.normal(0, n2[j], len(tk2))) for j in range(6)])
        d2.append((tk2, out))
    return d1, d2

disc1, disc2 = discretise(all_s1, all_s2)
min_pts = min(len(d[0]) for d in disc2)
print(f"  min_pts = {min_pts}")

X1_ii = np.array([[conds[i]['CA0'], disc1[i][1][-1,0], disc1[i][1][-1,1], disc1[i][1][-1,2],
                    conds[i]['T1'], conds[i]['t1']] for i in range(N)])
X2_ii = np.array([disc2[i][1][:min_pts, :].T.flatten() for i in range(N)])
final = np.array([disc2[i][1][-1] for i in range(N)])   # own true final point
Y_ii = (final[:, 4] / final.sum(axis=1) * 100).reshape(-1, 1)

def stage2_ode_mis(y, t, T2, Ea3, Ea4, A3f, A4f):
    CA, CB, CC, CD, CE, CF = y
    k3 = A3f * np.exp(-Ea3 / (R * T2))
    return [0, -k3*CB*CD, 0, -2*k3*CB*CD, k3*CB*CD, 0]

tgrid = np.arange(min_pts) * 15.0
data_dict_list = [dict(T2=conds[i]['T2'], conc=disc2[i][1][:min_pts]) for i in range(N)]
def model_flat(_, Ea3, Ea4, A3f, A4f):
    preds = []
    for d in data_dict_list:
        sol = odeint(stage2_ode_mis, d['conc'][0], tgrid, args=(d['T2'], Ea3, Ea4, A3f, A4f))
        preds.append(sol.ravel())
    return np.concatenate(preds)
y_obs = np.concatenate([d['conc'].ravel() for d in data_dict_list])
fitted, _ = curve_fit(model_flat, np.zeros_like(y_obs), y_obs, p0=[50000, 55000, 10, 20],
                       maxfev=10000, bounds=([0,0,0,0],[np.inf,np.inf,100,100]))
print(f"  fitted [Ea3,Ea4,A3,A4] = {np.round(fitted,3)}")

M2_ii = np.zeros((N, 6, min_pts))
for i in range(N):
    sol = odeint(stage2_ode_mis, disc2[i][1][0], tgrid, args=(conds[i]['T2'], *fitted))
    M2_ii[i] = sol.T
M2_ii = M2_ii.reshape(N, -1)
meta = np.array([[conds[i]['t2'], conds[i]['T2'], conds[i]['D0']] for i in range(N)])
M2_ii = np.hstack([M2_ii, meta])

# quick sanity vs real
X2m = pd.read_excel('X2_mis.xlsx'); Ym = pd.read_excel('Y_mis.xlsx')
print(f"\nY: real {Ym['E_pur'].values.mean():.2f}+/-{Ym['E_pur'].values.std(ddof=1):.2f}   "
      f"gen {Y_ii.mean():.2f}+/-{Y_ii.std(ddof=1):.2f}   corr={np.corrcoef(Ym['E_pur'].values, Y_ii.ravel())[0,1]:.3f}")

# =====================================================================
# validated blind-argmin CV search (identical to verify_paper_LV.py)
# =====================================================================
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

def cv_surface(X1, M2, X2, Y):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y)
    P_abc = np.zeros((n, A_MAX, B_MAX, C_MAX))
    rf = np.zeros((N_SPLITS, A_MAX, B_MAX, C_MAX))
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
                    P_abc[te,a-1,b-1,c-1] = yh.ravel()
                    rf[f,a-1,b-1,c-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    qabc = 1 - ((P_abc-Y[:,:,None,None])**2).sum(0)/sst
    return rf.mean(0), qabc

print("\nRunning blind-argmin CV search, case (ii), fixed-conditions data...")
rmsecv, qabc = cv_surface(X1_ii, M2_ii, X2_ii, Y_ii)
i_best = np.unravel_index(rmsecv.argmin(), rmsecv.shape)
best = (i_best[0]+1, i_best[1]+1, i_best[2]+1)
print(f"  best LV = {best}  RMSECV={rmsecv[i_best]:.4f}  Q2={qabc[i_best]:.4f}")
print(f"  paper's LV(2,3,6): RMSECV={rmsecv[1,2,5]:.4f}  Q2={qabc[1,2,5]:.4f}")
print(f"  your LV(3,3,6):    RMSECV={rmsecv[2,2,5]:.4f}  Q2={qabc[2,2,5]:.4f}")
