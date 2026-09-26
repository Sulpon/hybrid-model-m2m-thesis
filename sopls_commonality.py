"""
Commonality decomposition of the SO-PLS predictive performance.

The user's proposal:
    contribution of M2  =  full(X1+M2+X2)  -  DD(X1+X2)
    contribution of X2  =  full(X1+M2+X2)  -  KD(X1+M2)

These are the UNIQUE (incremental / semipartial) contributions. They do not sum
to the joint contribution of the two blocks, because whatever M2 and X2 both
carry is credited to neither. The missing piece needs the X1-only baseline:

    Joint    = Q2_full - Q2_X1only          (everything M2 and X2 add over X1)
    Unique_M = Q2_full - Q2_X1X2            (user's first quantity)
    Unique_X = Q2_full - Q2_X1M2            (user's second quantity)
    Shared   = Joint - Unique_M - Unique_X  (redundant, unattributable)

This script computes all four, under TWO allocation protocols, because the
choice changes the numbers materially:

    protocol "reopt"  -- every sub-model gets its own blind argmin RMSECV
                         allocation (consistent with sopls_two_block.py)
    protocol "fixed"  -- every sub-model is evaluated at the FULL model's LVs
                         for the blocks it retains (no re-optimisation)

Nothing else changes: KFold(10, shuffle, seed=42), block scaling 1/sqrt(K),
RMSECV = mean of per-fold RMSE, Q2 = 1 - pooled PRESS/SST. The full 3-block
results are read back from sopls_two_orderings.xlsx (same protocol, same seed).
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

def cv_surface_1(B1, Y, m1):
    """One-block SO-PLS (i.e. plain PLS on the scaled block). Returns RMSECV, Q2."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1)); rf = np.zeros((N_SPLITS, m1))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _scale_fit(B1[tr])
        b1t, b1e = _scale_apply(B1[tr],mu1,sd1,k1), _scale_apply(B1[te],mu1,sd1,k1)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Ytr0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af)
            yh = (T1e[:,:ae] @ Q1[:,:ae].T)*sdy + muY
            P[te, a-1] = yh.ravel()
            rf[f, a-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return rf.mean(0), 1 - ((P - Y)**2).sum(0)/sst

def cv_surface_2(B1, B2, Y, m1, m2):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2)); rf = np.zeros((N_SPLITS, m1, m2))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _scale_fit(B1[tr]); b1t, b1e = _scale_apply(B1[tr],mu1,sd1,k1), _scale_apply(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _scale_fit(B2[tr]); b2t0, b2e0 = _scale_apply(B2[tr],mu2,sd2,k2), _scale_apply(B2[te],mu2,sd2,k2)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Ytr0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            b2t, g2 = _resid(Tat, b2t0); b2e = b2e0 - Tae @ g2
            Ytr_a, _ = _resid(Tat, Ytr0)
            p2, T2t, Q2, bf = _fit(b2t, Ytr_a, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf)
                yh = (yh_a + T2e[:,:be] @ Q2[:,:be].T)*sdy + muY
                P[te, a-1, b-1] = yh.ravel()
                rf[f, a-1, b-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return rf.mean(0), 1 - ((P - Y[:,:,None])**2).sum(0)/sst

# full 3-block results from the run already done under the identical protocol
full = pd.read_excel('sopls_two_orderings.xlsx').set_index('model')

rows = []
t0 = time.time()
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1m, M2m_meta, Ym, n_pts_m, IC_m, X2flat_m
    else:
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1a, M2a_meta, Ya, n_pts_a, IC_a, X2flat_a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    fr = full.loc[name]
    a_F, b_F, c_F = int(fr.fwd_LV_X1), int(fr.fwd_LV_M2), int(fr.fwd_LV_X2)
    Q_full, R_full = float(fr.fwd_Q2), float(fr.fwd_RMSECV)

    rm1, q1   = cv_surface_1(X1_, Y_, MAX_X1)                      # X1 only
    rmM, qM   = cv_surface_2(X1_, M2_, Y_, MAX_X1, MAX_M2)         # X1 -> M2  (KD)
    rmX, qX   = cv_surface_2(X1_, X2_, Y_, MAX_X1, MAX_X2)         # X1 -> X2  (DD)

    # ---- protocol "reopt": each sub-model at its own blind argmin ----
    i1 = int(rm1.argmin())
    iM = np.unravel_index(rmM.argmin(), rmM.shape)
    iX = np.unravel_index(rmX.argmin(), rmX.shape)
    ro = dict(Q_X1=q1[i1], Q_KD=qM[iM], Q_DD=qX[iX],
              R_X1=rm1[i1], R_KD=rmM[iM], R_DD=rmX[iX],
              LV_X1=i1+1, LV_KD=f"({iM[0]+1},{iM[1]+1})", LV_DD=f"({iX[0]+1},{iX[1]+1})")
    # ---- protocol "fixed": sub-models at the FULL model's LVs ----
    fx = dict(Q_X1=q1[a_F-1], Q_KD=qM[a_F-1, b_F-1], Q_DD=qX[a_F-1, c_F-1],
              R_X1=rm1[a_F-1], R_KD=rmM[a_F-1, b_F-1], R_DD=rmX[a_F-1, c_F-1],
              LV_X1=a_F, LV_KD=f"({a_F},{b_F})", LV_DD=f"({a_F},{c_F})")

    r = dict(model=name, LV_full=f"({a_F},{b_F},{c_F})", Q_full=Q_full, R_full=R_full)
    for tagp, d in [('ro', ro), ('fx', fx)]:
        UM = Q_full - d['Q_DD']          # unique M2   (user's quantity 1)
        UX = Q_full - d['Q_KD']          # unique X2   (user's quantity 2)
        JT = Q_full - d['Q_X1']          # joint effect of M2 and X2 over X1
        r.update({
            f'{tagp}_LV_X1':  d['LV_X1'], f'{tagp}_LV_KD': d['LV_KD'], f'{tagp}_LV_DD': d['LV_DD'],
            f'{tagp}_Q_X1':   d['Q_X1'],  f'{tagp}_Q_KD':  d['Q_KD'],  f'{tagp}_Q_DD':  d['Q_DD'],
            f'{tagp}_R_X1':   d['R_X1'],  f'{tagp}_R_KD':  d['R_KD'],  f'{tagp}_R_DD':  d['R_DD'],
            f'{tagp}_UniqueM': UM, f'{tagp}_UniqueX': UX, f'{tagp}_Joint': JT,
            f'{tagp}_Shared': JT - UM - UX,
            f'{tagp}_sum_uniques': UM + UX,
            f'{tagp}_dR_M2': d['R_DD'] - R_full,     # same idea in RMSECV units
            f'{tagp}_dR_X2': d['R_KD'] - R_full,
        })
    rows.append(r)
    print(f"{name:14s} full Q2={Q_full:.4f}@{r['LV_full']:9s} | reopt: Q_X1={ro['Q_X1']:.4f} "
          f"Q_KD={ro['Q_KD']:.4f} Q_DD={ro['Q_DD']:.4f} -> UM={r['ro_UniqueM']:+.4f} "
          f"UX={r['ro_UniqueX']:+.4f} Joint={r['ro_Joint']:+.4f} Shared={r['ro_Shared']:+.4f}", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 320, 'display.max_columns', 80)

print("\n=== PROTOCOL 'reopt' : every sub-model re-optimised (blind argmin RMSECV) ===")
print(df[['model','LV_full','ro_LV_X1','ro_LV_KD','ro_LV_DD','Q_full','ro_Q_X1','ro_Q_KD','ro_Q_DD',
          'ro_UniqueM','ro_UniqueX','ro_sum_uniques','ro_Joint','ro_Shared']].round(4).to_string(index=False))
print("\n=== PROTOCOL 'fixed' : sub-models held at the FULL model's LVs ===")
print(df[['model','LV_full','fx_LV_KD','fx_LV_DD','Q_full','fx_Q_X1','fx_Q_KD','fx_Q_DD',
          'fx_UniqueM','fx_UniqueX','fx_sum_uniques','fx_Joint','fx_Shared']].round(4).to_string(index=False))
print("\n=== same contributions expressed in RMSECV units (reduction achieved by adding the block) ===")
print(df[['model','R_full','ro_R_X1','ro_R_KD','ro_R_DD','ro_dR_M2','ro_dR_X2',
          'fx_R_KD','fx_R_DD','fx_dR_M2','fx_dR_X2']].round(4).to_string(index=False))

df.to_excel('sopls_commonality.xlsx', index=False)
print(f"\nSaved sopls_commonality.xlsx   ({time.time()-t0:.0f}s)")
