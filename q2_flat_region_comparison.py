"""
How does Q2 change when the LV allocation is chosen by a 1-/2-/3-SE rule
instead of by plain argmin RMSECV?

Two separate questions, reported separately:

  (A) SELECTION COST.  Each SE rule selects the most PARSIMONIOUS allocation
      inside the flat region (minimum a+b+c; ties broken by lower RMSECV) --
      the standard use of the rule. Reported against the plain-argmin model:
      LV allocation, RMSECV, Q2, and the change in each.

  (B) SPREAD WITHIN THE REGION.  The distribution of Q2_full over all
      allocations admitted by each rule (median, p10, p90, min, max), which
      says how much Q2 is actually at stake inside the region regardless of
      which member you pick.

Flat region at level k:  RMSECV <= RMSECV_min + k*SE, with SE the across-fold
standard error of the mean RMSE at the argmin allocation.

Conventions unchanged: KFold(10, shuffle, seed=42), block scaling 1/sqrt(K),
RMSECV = mean of per-fold RMSE, Q2 = 1 - pooled PRESS/SST.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings, time
warnings.filterwarnings('ignore')

R = 8.314
species = ['A', 'B', 'C', 'D', 'E', 'F']
MAX_X1, MAX_M2, MAX_X2 = 6, 15, 15
N_SPLITS, SEED = 10, 42
KS = [1, 2, 3]

def ode_correct(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB*CD**2; r4 = k4*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_no_side(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, _, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); r3 = k3*CB*CD**2
    return [0, -r3, 0, -2*r3, r3, 0]
def ode_order1_D(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB*CD; r4 = k4*CC*CD
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_wrong_Ea4(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea3/(R*T2))
    r3 = k3*CB*CD**2; r4 = k4*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_lumped(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); r3 = k3*CB*CD**2; r4 = k3*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_no_D(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, _, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB; r4 = k4*CC
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_author_mis(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, _, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); r3 = k3*CB*CD
    return [0, -r3, 0, -2*r3, r3, 0]

MODELS = {'M0_correct': ode_correct, 'M1_no_side': ode_no_side, 'M2_order1_D': ode_order1_D,
          'M3_wrong_Ea4': ode_wrong_Ea4, 'M4_lumped_EF': ode_lumped, 'M5_no_D': ode_no_D,
          'M6_author_mis': ode_author_mis}
FITTED = {
    'M0_correct':    [4.9380789e+04, 5.6481086e+04, 8.023, 33.615],
    'M1_no_side':    [5.0088694e+04, 5.5000000e+04, 11.377, 20.0],
    'M2_order1_D':   [3.1123942e+04, 4.3435688e+04, 6.397, 145.759],
    'M3_wrong_Ea4':  [4.99379e+04, 5.50000e+04, 9.75, 3.506],
    'M4_lumped_EF':  [5.0039241e+04, 5.5000000e+04, 9.883, 20.0],
    'M5_no_D':       [2577.581, 658929.499, 9735.492, 9822.344],
    'M6_author_mis': [3.5882909e+04, 5.5000000e+04, 38.713, 20.0],
}

X1a = pd.read_excel('X1.xlsx')[['CA0','T1','t1','A','B','C']].values
X2a = pd.read_excel('X2.xlsx'); Ya = pd.read_excel('Y.xlsx')['E_pur'].values.reshape(-1,1)
M2a_meta = pd.read_excel('M2.xlsx')[['t2','T2','D0']]
n_pts_a = X2a.shape[1]//6
X1m = pd.read_excel('X1_mis.xlsx')[['CA0','T1','t1','A','B','C']].values
X2m = pd.read_excel('X2_mis.xlsx'); Ym = pd.read_excel('Y_mis.xlsx')['E_pur'].values.reshape(-1,1)
M2m_meta = pd.read_excel('M2_mis.xlsx')[['t2','T2','D0']]
n_pts_m = X2m.shape[1]//6
N = len(X1a)

def build_ic_and_X2(X2df, n_pts):
    IC = np.array([[X2df[f'{s}_1'].iloc[i] for s in species] for i in range(len(X2df))])
    X2flat = np.array([[X2df[f'{s}_{k+1}'].iloc[i] for s in species for k in range(n_pts)]
                        for i in range(len(X2df))])
    return IC, X2flat
IC_a, X2flat_a = build_ic_and_X2(X2a, n_pts_a)
IC_m, X2flat_m = build_ic_and_X2(X2m, n_pts_m)

def build_M2(rhs, IC, T2v, n_pts, tgrid, meta, params):
    M2 = np.zeros((N, 6, n_pts))
    for i in range(N):
        M2[i] = odeint(rhs, IC[i], tgrid, args=(T2v[i], *params)).T
    return np.hstack([M2.reshape(N, -1), meta.values])

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n
def _scale_fit(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, np.sqrt(B.shape[1])
def _scale_apply(B, mu, sd, k): return (B - mu)/sd/k
def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g

def cv_3(B1, B2, B3, Y, m1, m2, m3):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _scale_fit(B1[tr]); b1t, b1e = _scale_apply(B1[tr],mu1,sd1,k1), _scale_apply(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _scale_fit(B2[tr]); b2t0, b2e0 = _scale_apply(B2[tr],mu2,sd2,k2), _scale_apply(B2[te],mu2,sd2,k2)
        mu3, sd3, k3 = _scale_fit(B3[tr]); b3t0, b3e0 = _scale_apply(B3[tr],mu3,sd3,k3), _scale_apply(B3[te],mu3,sd3,k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Ytr0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            b2t, gM = _resid(Tat, b2t0); b2e = b2e0 - Tae @ gM
            b3t_a, gX = _resid(Tat, b3t0); b3e_a = b3e0 - Tae @ gX
            Ytr_a, _ = _resid(Tat, Ytr0)
            p2, T2t, Q2, bf = _fit(b2t, Ytr_a, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf); Tbt, Tbe = T2t[:,:be], T2e[:,:be]
                yh_ab = yh_a + Tbe @ Q2[:,:be].T
                b3t_b, gX2 = _resid(Tbt, b3t_a); b3e_b = b3e_a - Tbe @ gX2
                Ytr_b, _ = _resid(Tbt, Ytr_a)
                p3, T3t, Q3, cf = _fit(b3t_b, Ytr_b, m3); T3e = p3.transform(b3e_b)
                for c in range(1, m3+1):
                    ce = min(c, cf)
                    yh = ((yh_ab + T3e[:,:ce] @ Q3[:,:ce].T)*sdy + muY).ravel()
                    P[te, a-1, b-1, c-1] = yh
                    rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None,None])**2).sum(0)/sst, rf

selrows, sprows = [], []
t0 = time.time()
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1m, M2m_meta, Ym, n_pts_m, IC_m, X2flat_m
    else:
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1a, M2a_meta, Ya, n_pts_a, IC_a, X2flat_a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    qF, rf = cv_3(X1_, M2_, X2_, Y_, MAX_X1, MAX_M2, MAX_X2)
    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin]); qopt = float(qF[imin])
    se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    a0, b0, c0 = imin[0]+1, imin[1]+1, imin[2]+1

    regions = [('argmin (full)', np.zeros_like(rmse, dtype=bool))]
    regions[0][1][imin] = True
    for k in KS:
        regions.append((f'{k}-SE', rmse <= rmin + k*se))

    for lab, mask in regions:
        idx = np.argwhere(mask)
        tot = idx.sum(1) + 3                      # a+b+c  (indices are 0-based)
        # most parsimonious member; ties -> lowest RMSECV
        cand = idx[tot == tot.min()]
        rc = np.array([rmse[tuple(u)] for u in cand])
        pick = cand[rc.argmin()]
        a, b, c = pick[0]+1, pick[1]+1, pick[2]+1
        qsel = float(qF[tuple(pick)]); rsel = float(rmse[tuple(pick)])
        selrows.append(dict(model=name, rule=lab, n_region=int(mask.sum()),
            LV=f"({a},{b},{c})", total_LV=int(a+b+c),
            RMSECV=rsel, Q2=qsel,
            dQ2_pp=100*(qsel-qopt), dRMSECV=rsel-rmin,
            opt_LV=f"({a0},{b0},{c0})", opt_total_LV=a0+b0+c0,
            opt_RMSECV=rmin, opt_Q2=qopt, SE=se))
        if lab != 'argmin (full)':
            v = qF[mask]
            sprows.append(dict(model=name, region=lab, n=int(mask.sum()),
                Q2_median=float(np.median(v)), Q2_p10=float(np.percentile(v, 10)),
                Q2_p90=float(np.percentile(v, 90)), Q2_min=float(v.min()), Q2_max=float(v.max()),
                Q2_range=float(v.max()-v.min()), Q2_opt=qopt,
                dQ2_worst_pp=100*(float(v.min())-qopt)))
    v = qF.ravel()
    sprows.append(dict(model=name, region='full grid', n=v.size,
        Q2_median=float(np.median(v)), Q2_p10=float(np.percentile(v, 10)),
        Q2_p90=float(np.percentile(v, 90)), Q2_min=float(v.min()), Q2_max=float(v.max()),
        Q2_range=float(v.max()-v.min()), Q2_opt=qopt, dQ2_worst_pp=100*(float(v.min())-qopt)))
    print(f"{name:14s} argmin=({a0},{b0},{c0}) RMSECV={rmin:.4f} Q2={qopt:.4f} SE={se:.4f}", flush=True)

sel = pd.DataFrame(selrows); spr = pd.DataFrame(sprows)
pd.set_option('display.width', 320, 'display.max_columns', 60)

print("\n=== (A) MODEL SELECTED BY EACH RULE (most parsimonious inside the region) ===")
print(sel[['model','rule','n_region','LV','total_LV','RMSECV','Q2','dRMSECV','dQ2_pp']].round(4).to_string(index=False))

print("\n--- Q2 of the selected model, by rule ---")
print(sel.pivot(index='model', columns='rule', values='Q2')[['argmin (full)','1-SE','2-SE','3-SE']].round(4).to_string())
print("\n--- change in Q2 vs plain argmin, percentage points ---")
print(sel.pivot(index='model', columns='rule', values='dQ2_pp')[['argmin (full)','1-SE','2-SE','3-SE']].round(2).to_string())
print("\n--- LV allocation selected, by rule ---")
print(sel.pivot(index='model', columns='rule', values='LV')[['argmin (full)','1-SE','2-SE','3-SE']].to_string())
print("\n--- total LVs, by rule ---")
print(sel.pivot(index='model', columns='rule', values='total_LV')[['argmin (full)','1-SE','2-SE','3-SE']].to_string())
print("\n--- RMSECV of the selected model, by rule ---")
print(sel.pivot(index='model', columns='rule', values='RMSECV')[['argmin (full)','1-SE','2-SE','3-SE']].round(4).to_string())

print("\n=== (B) SPREAD OF Q2 ACROSS EACH REGION ===")
print(spr[['model','region','n','Q2_opt','Q2_median','Q2_p10','Q2_p90','Q2_min','Q2_max','Q2_range','dQ2_worst_pp']].round(4).to_string(index=False))
print("\n--- median Q2 inside the region ---")
print(spr.pivot(index='model', columns='region', values='Q2_median')[['1-SE','2-SE','3-SE','full grid']].round(4).to_string())
print("\n--- worst Q2 admitted by the region (pp below the argmin Q2) ---")
print(spr.pivot(index='model', columns='region', values='dQ2_worst_pp')[['1-SE','2-SE','3-SE','full grid']].round(2).to_string())

with pd.ExcelWriter('q2_flat_region_comparison.xlsx') as w:
    sel.to_excel(w, sheet_name='selected_model', index=False)
    spr.to_excel(w, sheet_name='Q2_spread', index=False)
print(f"\nSaved q2_flat_region_comparison.xlsx   ({time.time()-t0:.0f}s)")
