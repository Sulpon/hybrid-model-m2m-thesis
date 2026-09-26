"""
Two independent SO-PLS runs per model, each with its OWN RMSECV-optimal LV
allocation (blind argmin over the full grid), plus the sequential in-sample
SS decomposition at that allocation.

  Run 1:  X1 -> M2 -> X2
  Run 2:  X1 -> X2 -> M2      (re-optimised from scratch, NOT run 1's LV)

Regime: real calibration data (X1.xlsx/X2.xlsx/Y.xlsx; M6 on the *_mis files),
per-model kinetic parameters fitted by least_squares. The fitted parameters are
reused verbatim from the established ablation run -- least_squares is
deterministic given the same data/p0/bounds, so re-fitting reproduces the same
numbers at large cost. Nothing else is reused: both LV searches are run fresh.

Conventions held identical to every prior script: mean-centre + unit-variance
per column, block-scale by 1/sqrt(K), KFold(10, shuffle, seed=42),
RMSECV = mean over folds of per-fold RMSE, Q2 = 1 - pooled PRESS/SST,
SS in standardised-Y units (SST = N-1 = 99).
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
MAX_X1, MAX_M2, MAX_X2 = 6, 15, 15          # per-BLOCK maxima, independent of position
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

def cv_surface_ordered(B1, B2, B3, Y, m1, m2, m3):
    """CV grid for B1 -> B2 -> B3 (B1 always first). Returns RMSECV, Q2 arrays."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y)
    P = np.zeros((n, m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
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
                    yh = (yh_ab + T3e[:,:ce] @ Q3[:,:ce].T)*sdy + muY
                    P[te, a-1, b-1, c-1] = yh.ravel()
                    rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    q2 = 1 - ((P - Y[:,:,None,None])**2).sum(0)/sst
    return rf.mean(0), q2

def seq_SS_ordered(B1, B2, B3, Y, n1, n2, n3):
    """In-sample sequential SS for B1(n1) -> B2(n2) -> B3(n3)."""
    mu1, sd1, k1 = _scale_fit(B1); b1s = _scale_apply(B1, mu1, sd1, k1)
    mu2, sd2, k2 = _scale_fit(B2); b2s = _scale_apply(B2, mu2, sd2, k2)
    mu3, sd3, k3 = _scale_fit(B3); b3s = _scale_apply(B3, mu3, sd3, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    sst = float(np.sum(ys**2))
    p1, T1, Q1, af = _fit(b1s, ys, n1); ae = min(n1, af); T1, Q1 = T1[:,:ae], Q1[:,:ae]
    ss1 = float(np.sum((T1 @ Q1.T)**2))
    b2r, _ = _resid(T1, b2s); b3_a, _ = _resid(T1, b3s); ytr_a, _ = _resid(T1, ys)
    p2, T2v, Q2, bf = _fit(b2r, ytr_a, n2); be = min(n2, bf); T2v, Q2 = T2v[:,:be], Q2[:,:be]
    ss2 = float(np.sum((T2v @ Q2.T)**2))
    b3_b, _ = _resid(T2v, b3_a); ytr_b, _ = _resid(T2v, ytr_a)
    p3, T3, Q3, cf = _fit(b3_b, ytr_b, n3); ce = min(n3, cf); T3, Q3 = T3[:,:ce], Q3[:,:ce]
    ss3 = float(np.sum((T3 @ Q3.T)**2))
    return ss1, ss2, ss3, sst - ss1 - ss2 - ss3, sst

# ---------------------------------------------------------------- run
rows = []
t0 = time.time()
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, X2flat_ = X1m, M2m_meta, Ym, n_pts_m, IC_m, X2flat_m
    else:
        X1_, meta_, Y_, n_pts_, IC_, X2flat_ = X1a, M2a_meta, Ya, n_pts_a, IC_a, X2flat_a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    # ---- Run 1: X1 -> M2 -> X2 ----
    rm_f, q_f = cv_surface_ordered(X1_, M2_, X2flat_, Y_, MAX_X1, MAX_M2, MAX_X2)
    i_f = np.unravel_index(rm_f.argmin(), rm_f.shape)
    a_f, b_f, c_f = i_f[0]+1, i_f[1]+1, i_f[2]+1
    ssx1_f, ssM_f, ssX_f, ssF_f, sst = seq_SS_ordered(X1_, M2_, X2flat_, Y_, a_f, b_f, c_f)

    # ---- Run 2: X1 -> X2 -> M2  (independent argmin) ----
    rm_r, q_r = cv_surface_ordered(X1_, X2flat_, M2_, Y_, MAX_X1, MAX_X2, MAX_M2)
    i_r = np.unravel_index(rm_r.argmin(), rm_r.shape)
    a_r, c_r, b_r = i_r[0]+1, i_r[1]+1, i_r[2]+1     # 2nd position = X2, 3rd = M2
    ssx1_r, ssX_r, ssM_r, ssF_r, _ = seq_SS_ordered(X1_, X2flat_, M2_, Y_, a_r, c_r, b_r)

    rows.append(dict(model=name,
        fwd_LV_X1=a_f, fwd_LV_M2=b_f, fwd_LV_X2=c_f,
        fwd_RMSECV=rm_f[i_f], fwd_Q2=q_f[i_f],
        fwd_SS_X1=ssx1_f, fwd_SS_M2=ssM_f, fwd_SS_X2=ssX_f, fwd_SS_F=ssF_f,
        fwd_M2M=ssM_f/ssX_f if ssX_f > 1e-9 else np.nan,
        rev_LV_X1=a_r, rev_LV_X2=c_r, rev_LV_M2=b_r,
        rev_RMSECV=rm_r[i_r], rev_Q2=q_r[i_r],
        rev_SS_X1=ssx1_r, rev_SS_X2=ssX_r, rev_SS_M2=ssM_r, rev_SS_F=ssF_r,
        rev_M2M=ssM_r/ssX_r if ssX_r > 1e-9 else np.nan,
        SS_Y=sst))
    r = rows[-1]
    print(f"{name}")
    print(f"   FWD X1->M2->X2  LV=({a_f},{b_f},{c_f})  RMSECV={rm_f[i_f]:.4f}  Q2={q_f[i_f]:.4f}  "
          f"SS: X1={ssx1_f:.3f} M2={ssM_f:.3f} X2={ssX_f:.3f} F={ssF_f:.3f}")
    print(f"   REV X1->X2->M2  LV=({a_r},{c_r},{b_r})  RMSECV={rm_r[i_r]:.4f}  Q2={q_r[i_r]:.4f}  "
          f"SS: X1={ssx1_r:.3f} X2={ssX_r:.3f} M2={ssM_r:.3f} F={ssF_r:.3f}", flush=True)

df = pd.DataFrame(rows)
df['fwd_SS_closure'] = df.fwd_SS_X1+df.fwd_SS_M2+df.fwd_SS_X2+df.fwd_SS_F-df.SS_Y
df['rev_SS_closure'] = df.rev_SS_X1+df.rev_SS_X2+df.rev_SS_M2+df.rev_SS_F-df.SS_Y
df['joint_fwd'] = df.fwd_SS_M2+df.fwd_SS_X2
df['joint_rev'] = df.rev_SS_M2+df.rev_SS_X2

pd.set_option('display.width', 250, 'display.max_columns', 60)
print("\n=== RUN 1: X1 -> M2 -> X2 ===")
print(df[['model','fwd_LV_X1','fwd_LV_M2','fwd_LV_X2','fwd_RMSECV','fwd_Q2',
          'fwd_SS_X1','fwd_SS_M2','fwd_SS_X2','fwd_SS_F','fwd_M2M']].round(4).to_string(index=False))
print("\n=== RUN 2: X1 -> X2 -> M2 (independently re-optimised) ===")
print(df[['model','rev_LV_X1','rev_LV_X2','rev_LV_M2','rev_RMSECV','rev_Q2',
          'rev_SS_X1','rev_SS_X2','rev_SS_M2','rev_SS_F','rev_M2M']].round(4).to_string(index=False))
print("\n=== checks ===")
print(df[['model','SS_Y','fwd_SS_closure','rev_SS_closure','joint_fwd','joint_rev']].round(6).to_string(index=False))
df.to_excel('sopls_two_orderings.xlsx', index=False)
print(f"\nSaved sopls_two_orderings.xlsx   ({time.time()-t0:.0f}s)")
