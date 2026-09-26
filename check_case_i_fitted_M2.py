"""
Case (i) currently uses M2 built from the TRUE nominal parameters
(Ea3=50000, Ea4=55000, A3=10, A4=20) directly, no fitting -- matching the
paper's stated "Case (i): a model that exactly matches the true dynamics,
using the true parameters from Table 3." But the residual LV_X1/LV_M2 gap
vs the paper's {4,6,2} (found (6,14,2) instead) suggests testing the
alternative: refit Ea3/Ea4/A3/A4 via least_squares on the 100 training
batches (same procedure as case ii), even though the model is structurally
correct, and see whether that changes the LV allocation found.
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
A1, A2, A3_true, A4_true = 28, 40, 10, 20
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

def stage2_ode_correct(y, t, T2, Ea3, Ea4, A3v, A4v):
    CA, CB, CC, CD, CE, CF = y
    k3 = A3v*np.exp(-Ea3/(R*T2)); k4 = A4v*np.exp(-Ea4/(R*T2))
    return [0, -k3*CB*CD**2, -k4*CC*CD**2, -2*k3*CB*CD**2-2*k4*CC*CD**2, k3*CB*CD**2, k4*CC*CD**2]

# =====================================================================
# Regenerate case (i), seed=10 -- exact match to real X1.xlsx/X2.xlsx (validated earlier)
# =====================================================================
N = 100
print("Regenerating case (i), seed=10 (exact-match conditions)...")
np.random.seed(10)
all_s1, all_s2, params = [], [], []
for i in range(N):
    Ea1 = np.random.normal(Ea1_m, CV*Ea1_m); Ea2 = np.random.normal(Ea2_m, CV*Ea2_m)
    Ea3 = np.random.normal(Ea3_m, CV*Ea3_m); Ea4 = np.random.normal(Ea4_m, CV*Ea4_m)
    T1 = np.random.uniform(290, 310); T2 = np.random.uniform(330, 370)
    t1s = np.random.uniform(350, 650); t2s = np.random.uniform(100, 400)
    CA0s = np.random.normal(2000.0, 0.05*2000.0); CD0s = np.random.normal(1900.0, 0.05*1900.0)
    k1 = A1*np.exp(-Ea1/(R*T1)); k2 = A2*np.exp(-Ea2/(R*T1))
    k3 = A3_true*np.exp(-Ea3/(R*T2)); k4 = A4_true*np.exp(-Ea4/(R*T2))
    t1v = np.linspace(0, t1s, 1000)
    sol1 = sdeint.itoint(lambda y,t: stage1_drift(y,t,k1,k2), stage1_noise, np.array([CA0s,0.,0.]), t1v)
    CA1f, CB1f, CC1f = sol1[-1]; d = V1/V2
    y0 = np.array([CA1f*d, CB1f*d, CC1f*d, CD0s, 0., 0.])
    t2v = np.linspace(0, t2s, 1000)
    sol2 = sdeint.itoint(lambda y,t: stage2_drift(y,t,k3,k4), stage2_noise, y0, t2v)
    all_s1.append((t1v/60, sol1)); all_s2.append((t2v/60, sol2))
    params.append(dict(CA0=CA0s, T1=T1, t1=t1s/60, T2=T2, D0=CD0s, t2=t2s/60))

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

X1_i = np.array([[params[i]['CA0'], disc1[i][1][-1,0], disc1[i][1][-1,1], disc1[i][1][-1,2],
                   params[i]['T1'], params[i]['t1']] for i in range(N)])
X2_i = np.array([disc2[i][1][:min_pts, :].T.flatten() for i in range(N)])
final = np.array([disc2[i][1][-1] for i in range(N)])
Y_i = (final[:, 4] / final.sum(axis=1) * 100).reshape(-1, 1)

# sanity vs real
X2a = pd.read_excel('X2.xlsx'); Ya = pd.read_excel('Y.xlsx')
n_pts_a = X2a.shape[1]//6
print("Sanity vs real (final pt):")
for j, sp in enumerate(species):
    a = X2a[f'{sp}_{n_pts_a}'].values
    print(f"  {sp}: real {a.mean():8.2f}  gen {final[:,j].mean():8.2f}  corr={np.corrcoef(a,final[:,j])[0,1]:.3f}")
print(f"  Y: real {Ya['E_pur'].values.mean():.2f}  gen {Y_i.mean():.2f}  corr={np.corrcoef(Ya['E_pur'].values, Y_i.ravel())[0,1]:.3f}")

# =====================================================================
# M2, variant A: TRUE nominal params (what we did before)
# =====================================================================
tgrid = np.arange(min_pts) * 15.0
M2_true = np.zeros((N, 6, min_pts))
for i in range(N):
    sol = odeint(stage2_ode_correct, disc2[i][1][0], tgrid, args=(params[i]['T2'], Ea3_m, Ea4_m, A3_true, A4_true))
    M2_true[i] = sol.T
M2_true = M2_true.reshape(N, -1)

# =====================================================================
# M2, variant B: FIT Ea3,Ea4,A3,A4 via least_squares on all 100 (correct structure)
# =====================================================================
data_dict_list = [dict(T2=params[i]['T2'], conc=disc2[i][1][:min_pts]) for i in range(N)]
def model_flat(_, Ea3, Ea4, A3v, A4v):
    preds = []
    for d in data_dict_list:
        sol = odeint(stage2_ode_correct, d['conc'][0], tgrid, args=(d['T2'], Ea3, Ea4, A3v, A4v))
        preds.append(sol.ravel())
    return np.concatenate(preds)
y_obs = np.concatenate([d['conc'].ravel() for d in data_dict_list])
fitted, _ = curve_fit(model_flat, np.zeros_like(y_obs), y_obs, p0=[50000, 55000, 10, 20],
                       maxfev=10000, bounds=([0,0,0,0],[np.inf,np.inf,100,100]))
print(f"\nFitted [Ea3,Ea4,A3,A4] for case (i) (correct structure): {np.round(fitted,3)}  (true=[50000,55000,10,20])")

M2_fit = np.zeros((N, 6, min_pts))
for i in range(N):
    sol = odeint(stage2_ode_correct, disc2[i][1][0], tgrid, args=(params[i]['T2'], *fitted))
    M2_fit[i] = sol.T
M2_fit = M2_fit.reshape(N, -1)

meta = np.array([[params[i]['t2'], params[i]['T2'], params[i]['D0']] for i in range(N)])
M2_true = np.hstack([M2_true, meta])
M2_fit = np.hstack([M2_fit, meta])

# =====================================================================
# validated blind-argmin CV search
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

for label, M2 in [('TRUE nominal params (no fit)', M2_true), ('FITTED params', M2_fit)]:
    print(f"\nRunning blind-argmin CV search, case (i), M2 = {label}...")
    rmsecv, qabc = cv_surface(X1_i, M2, X2_i, Y_i)
    i_best = np.unravel_index(rmsecv.argmin(), rmsecv.shape)
    best = (i_best[0]+1, i_best[1]+1, i_best[2]+1)
    print(f"  best LV = {best}  RMSECV={rmsecv[i_best]:.4f}  Q2={qabc[i_best]:.4f}")
    print(f"  at paper's LV(4,6,2): RMSECV={rmsecv[3,5,1]:.4f}  Q2={qabc[3,5,1]:.4f}")
