"""
TWO-BLOCK SO-PLS runs, each with its OWN blind-argmin RMSECV-optimal allocation
and the sequential in-sample SS decomposition at that allocation.

  Run A:  X1 -> X2      (purely data-driven; contains NO mechanistic block, so it
                         is model-independent -- only two distinct fits exist,
                         one on the main dataset and one on the *_mis dataset
                         that M6 uses)
  Run B:  X1 -> M2      (one fit per candidate model M0..M6)

Every convention is held identical to sopls_two_orderings.py: mean-centre +
unit-variance per column, block-scale by 1/sqrt(K), KFold(10, shuffle, seed=42),
RMSECV = mean over folds of per-fold RMSE, Q2 = 1 - pooled PRESS/SST, SS in
standardised-Y units (SST = N-1 = 99). Per-block LV maxima are the same:
X1 <= 6, M2 <= 15, X2 <= 15. Kinetic parameters are the same fitted values.
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

# ---------------------------------------------------------------- mechanisms
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

# ---------------------------------------------------------------- data
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

# ---------------------------------------------------------------- core helpers
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

def cv_surface_2(B1, B2, Y, m1, m2):
    """CV grid for the TWO-block sequence B1 -> B2. Returns RMSECV, Q2 arrays."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y)
    P = np.zeros((n, m1, m2)); rf = np.zeros((N_SPLITS, m1, m2))
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
    q2 = 1 - ((P - Y[:,:,None])**2).sum(0)/sst
    return rf.mean(0), q2

def seq_SS_2(B1, B2, Y, n1, n2):
    """In-sample sequential SS for the TWO-block sequence B1(n1) -> B2(n2)."""
    mu1, sd1, k1 = _scale_fit(B1); b1s = _scale_apply(B1, mu1, sd1, k1)
    mu2, sd2, k2 = _scale_fit(B2); b2s = _scale_apply(B2, mu2, sd2, k2)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    sst = float(np.sum(ys**2))
    p1, T1, Q1, af = _fit(b1s, ys, n1); ae = min(n1, af); T1, Q1 = T1[:,:ae], Q1[:,:ae]
    ss1 = float(np.sum((T1 @ Q1.T)**2))
    b2r, _ = _resid(T1, b2s); ytr_a, _ = _resid(T1, ys)
    p2, T2v, Q2, bf = _fit(b2r, ytr_a, n2); be = min(n2, bf); T2v, Q2 = T2v[:,:be], Q2[:,:be]
    ss2 = float(np.sum((T2v @ Q2.T)**2))
    return ss1, ss2, sst - ss1 - ss2, sst

# ================================================================ RUN A: X1 -> X2
t0 = time.time()
print("=== RUN A: X1 -> X2   (no mechanistic block; model-independent) ===", flush=True)
rowsA = []
for tag, X1_, X2_, Y_ in [('main dataset (M0-M5)', X1a, X2flat_a, Ya),
                          ('mis dataset (M6)',     X1m, X2flat_m, Ym)]:
    rm, q = cv_surface_2(X1_, X2_, Y_, MAX_X1, MAX_X2)
    i = np.unravel_index(rm.argmin(), rm.shape); a, c = i[0]+1, i[1]+1
    ss1, ss2, ssF, sst = seq_SS_2(X1_, X2_, Y_, a, c)
    rowsA.append(dict(dataset=tag, LV_X1=a, LV_X2=c, RMSECV=rm[i], Q2=q[i],
                      SS_X1=ss1, SS_X2=ss2, SS_F=ssF, SS_Y=sst))
    print(f"  {tag:22s} LV=(X1={a}, X2={c})  RMSECV={rm[i]:.4f}  Q2={q[i]:.4f}  "
          f"SS: X1={ss1:.3f} X2={ss2:.3f} F={ssF:.3f}  (SS_Y={sst:.1f})", flush=True)
dfA = pd.DataFrame(rowsA)

# ================================================================ RUN B: X1 -> M2
print("\n=== RUN B: X1 -> M2   (one fit per candidate model) ===", flush=True)
rowsB = []
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_ = X1m, M2m_meta, Ym, n_pts_m, IC_m
    else:
        X1_, meta_, Y_, n_pts_, IC_ = X1a, M2a_meta, Ya, n_pts_a, IC_a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    rm, q = cv_surface_2(X1_, M2_, Y_, MAX_X1, MAX_M2)
    i = np.unravel_index(rm.argmin(), rm.shape); a, b = i[0]+1, i[1]+1
    ss1, ss2, ssF, sst = seq_SS_2(X1_, M2_, Y_, a, b)
    rowsB.append(dict(model=name, LV_X1=a, LV_M2=b, RMSECV=rm[i], Q2=q[i],
                      SS_X1=ss1, SS_M2=ss2, SS_F=ssF, SS_Y=sst))
    print(f"  {name:14s} LV=(X1={a}, M2={b})  RMSECV={rm[i]:.4f}  Q2={q[i]:.4f}  "
          f"SS: X1={ss1:.3f} M2={ss2:.3f} F={ssF:.3f}", flush=True)
dfB = pd.DataFrame(rowsB)

dfA['SS_closure'] = dfA.SS_X1+dfA.SS_X2+dfA.SS_F-dfA.SS_Y
dfB['SS_closure'] = dfB.SS_X1+dfB.SS_M2+dfB.SS_F-dfB.SS_Y

pd.set_option('display.width', 250, 'display.max_columns', 60)
print("\n--- RUN A table (X1 -> X2) ---")
print(dfA.round(4).to_string(index=False))
print("\n--- RUN B table (X1 -> M2) ---")
print(dfB.round(4).to_string(index=False))

with pd.ExcelWriter('sopls_two_block.xlsx') as w:
    dfA.to_excel(w, sheet_name='X1_X2', index=False)
    dfB.to_excel(w, sheet_name='X1_M2', index=False)
print(f"\nSaved sopls_two_block.xlsx   ({time.time()-t0:.0f}s)")
