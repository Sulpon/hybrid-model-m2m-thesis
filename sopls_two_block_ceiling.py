"""
Is the reported two-block RMSECV the best these configurations can predict?

Three separate questions, answered separately -- no new methodology, no
thresholds beyond the ones already used elsewhere in this project (1%, 1-SE):

  (1) BOUNDARY   -- was the argmin pressed against the LV cap? Re-run the same
                    grids with the caps raised (X1 to its full column rank,
                    M2 and X2 to 30) and report whether RMSECV improves.
  (2) FLATNESS   -- how sharply defined is the minimum? Count grid points
                    within 1% of it, and report the across-fold SE of the
                    minimum plus the simplest allocation inside the 1-SE band.
  (3) SEED       -- how much of the reported number is CV-partition luck?
                    Repeat over 10 KFold seeds. For each seed record both the
                    seed's own argmin RMSECV (what a blind search would report)
                    and the RMSECV at the seed-42 allocation (honest re-use of
                    a fixed allocation). The gap between the two is the
                    selection optimism of picking the min of a noisy surface.

Conventions otherwise identical to sopls_two_block.py.
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
CAP_M2_EXT, CAP_X2_EXT = 30, 30      # raised caps for the boundary test
BASE_M2, BASE_X2 = 15, 15            # the caps used in sopls_two_block.py
N_SPLITS = 10
SEED = 42
SEEDS = [1, 7, 13, 42, 99, 123, 777, 2024, 31337, 5]

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

def cv_folds_2(B1, B2, Y, m1, m2, seed):
    """Per-fold RMSE cube (n_splits, m1, m2) for B1 -> B2."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=seed)
    rf = np.zeros((N_SPLITS, m1, m2))
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
                rf[f, a-1, b-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    return rf

# ------------------------------------------------------------------ the runs
RUNS = []
rank_X1a = np.linalg.matrix_rank(X1a); rank_X1m = np.linalg.matrix_rank(X1m)
RUNS.append(('A: X1->X2  main', X1a, X2flat_a, Ya, rank_X1a, CAP_X2_EXT, BASE_X2, 'X2'))
RUNS.append(('A: X1->X2  mis',  X1m, X2flat_m, Ym, rank_X1m, CAP_X2_EXT, BASE_X2, 'X2'))
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, rk = X1m, M2m_meta, Ym, n_pts_m, IC_m, rank_X1m
    else:
        X1_, meta_, Y_, n_pts_, IC_, rk = X1a, M2a_meta, Ya, n_pts_a, IC_a, rank_X1a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])
    RUNS.append((f'B: X1->M2  {name}', X1_, M2_, Y_, rk, CAP_M2_EXT, BASE_M2, 'M2'))

print(f"rank(X1 main)={rank_X1a}  rank(X1 mis)={rank_X1m}   "
      f"ncol(X2 main)={X2flat_a.shape[1]}  ncol(X2 mis)={X2flat_m.shape[1]}\n", flush=True)

rows = []
t0 = time.time()
for tag, B1, B2, Y, m1, cap_ext, cap_base, b2name in RUNS:
    # ---------- (1)+(2) on the EXTENDED grid at the reference seed ----------
    rf = cv_folds_2(B1, B2, Y, m1, cap_ext, SEED)
    rm = rf.mean(0)

    # baseline result: restrict to the original cap
    rm_base = rm[:, :cap_base]
    i_b = np.unravel_index(rm_base.argmin(), rm_base.shape); a_b, b_b = i_b[0]+1, i_b[1]+1
    r_base = rm_base[i_b]
    # extended result
    i_e = np.unravel_index(rm.argmin(), rm.shape); a_e, b_e = i_e[0]+1, i_e[1]+1
    r_ext = rm[i_e]

    # flatness at the extended optimum
    n_within1 = int((rm <= r_ext*1.01).sum())
    se = rf[:, i_e[0], i_e[1]].std(ddof=1)/np.sqrt(N_SPLITS)
    inside = np.argwhere(rm <= r_ext + se)
    simplest = inside[(inside[:,0]+inside[:,1]).argmin()]
    a_s, b_s = int(simplest[0])+1, int(simplest[1])+1
    r_s = rm[simplest[0], simplest[1]]

    # ---------- (3) seed sensitivity, on the ORIGINAL grid ----------
    own, fixed = [], []
    for s in SEEDS:
        rfs = cv_folds_2(B1, B2, Y, m1, cap_base, s).mean(0)
        own.append(rfs.min())
        fixed.append(rfs[a_b-1, b_b-1])
    own, fixed = np.array(own), np.array(fixed)

    rows.append(dict(run=tag,
        base_LV=f"({a_b},{b_b})", base_RMSECV=r_base,
        ext_LV=f"({a_e},{b_e})", ext_RMSECV=r_ext,
        cap_bound=(b_b == cap_base), ext_gain=r_base-r_ext,
        n_within_1pct=n_within1, fold_SE=se,
        oneSE_LV=f"({a_s},{b_s})", oneSE_RMSECV=r_s,
        seed_own_mean=own.mean(), seed_own_sd=own.std(ddof=1),
        seed_fixed_mean=fixed.mean(), seed_fixed_sd=fixed.std(ddof=1),
        seed_fixed_min=fixed.min(), seed_fixed_max=fixed.max(),
        optimism=fixed.mean()-own.mean()))
    r = rows[-1]
    print(f"{tag:26s} base={r_base:.4f}@{r['base_LV']:7s} ext={r_ext:.4f}@{r['ext_LV']:7s} "
          f"gain={r_base-r_ext:+.4f}  flat1%={n_within1:4d}  SE={se:.4f}  "
          f"1SE@{r['oneSE_LV']:7s}={r_s:.4f}  seedfix={fixed.mean():.4f}+/-{fixed.std(ddof=1):.4f} "
          f"[{fixed.min():.4f},{fixed.max():.4f}]  optimism={r['optimism']:+.4f}", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 300, 'display.max_columns', 60)
print("\n--- (1) BOUNDARY: does raising the LV cap help? ---")
print(df[['run','base_LV','base_RMSECV','cap_bound','ext_LV','ext_RMSECV','ext_gain']].round(4).to_string(index=False))
print("\n--- (2) FLATNESS around the optimum ---")
print(df[['run','ext_RMSECV','n_within_1pct','fold_SE','oneSE_LV','oneSE_RMSECV']].round(4).to_string(index=False))
print("\n--- (3) SEED sensitivity over 10 KFold partitions (original grid) ---")
print(df[['run','base_RMSECV','seed_fixed_mean','seed_fixed_sd','seed_fixed_min','seed_fixed_max',
          'seed_own_mean','optimism']].round(4).to_string(index=False))
df.to_excel('sopls_two_block_ceiling.xlsx', index=False)
print(f"\nSaved sopls_two_block_ceiling.xlsx   ({time.time()-t0:.0f}s)")
