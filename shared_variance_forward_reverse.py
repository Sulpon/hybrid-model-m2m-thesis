"""
Rigorous test of whether shared M2/X2 variance is identifiable from the
SO-PLS SS decomposition, using forward (X1->M2->X2) and reverse
(X1->X2->M2) fits, for all 7 models, on the REAL calibration data with
each model's ALREADY-ESTABLISHED LV allocation and fitted kinetic
parameters (reused as-is, not re-derived -- deterministic outputs of the
prior validated run).

Mathematical basis (commonality analysis, Newton & Rudestam 1999 /
Seibold & McPhee 1979 -- a standard technique for decomposing variance
shared between two correlated predictor sets, both regressed on a third,
common covariate X1 first):

  SS_M_forward = SSR(M2 alone | X1)            -- M2 fit BEFORE X2 enters
  SS_X_reverse = SSR(X2 alone | X1)             -- X2 fit BEFORE M2 enters
  SS_X_forward = SSR(X2 | X1, M2) = unique_X    -- X2's contribution AFTER M2
  SS_M_reverse = SSR(M2 | X1, X2) = unique_M    -- M2's contribution AFTER X2

  Common(M,X) = SSR(M|X1) + SSR(X|X1) - SSR(M,X|X1)
              = SS_M_forward - SS_M_reverse     (one route)
              = SS_X_reverse - SS_X_forward     (the other route)

These two routes are ALGEBRAICALLY IDENTICAL if and only if the "joint
invariance" condition holds:
  SS_M_forward + SS_X_forward  ==  SS_M_reverse + SS_X_reverse
(i.e. the two orderings explain the same TOTAL joint variance from
{M2,X2} combined, after X1). This is guaranteed exactly for saturated
(full-rank) OLS regression, but is NOT guaranteed for SO-PLS with a
FINITE number of latent variables per block -- checked empirically below,
not assumed.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

R = 8.314
species = ['A', 'B', 'C', 'D', 'E', 'F']

# ---------------------------------------------------------------------------
# 7 candidate mechanisms (identical to established scripts)
# ---------------------------------------------------------------------------
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

MODELS = {
    'M0_correct':    ode_correct,
    'M1_no_side':    ode_no_side,
    'M2_order1_D':   ode_order1_D,
    'M3_wrong_Ea4':  ode_wrong_Ea4,
    'M4_lumped_EF':  ode_lumped,
    'M5_no_D':       ode_no_D,
    'M6_author_mis': ode_author_mis,
}
LV = {
    'M0_correct':    (4, 6, 2),
    'M1_no_side':    (1, 13, 4),
    'M2_order1_D':   (4, 6, 4),
    'M3_wrong_Ea4':  (4, 6, 2),
    'M4_lumped_EF':  (4, 8, 2),
    'M5_no_D':       (3, 7, 4),
    'M6_author_mis': (3, 3, 6),
}
FITTED = {
    'M0_correct':    [4.9380789e+04, 5.6481086e+04, 8.023, 33.615],
    'M1_no_side':    [5.0088694e+04, 5.5000000e+04, 11.377, 20.0],
    'M2_order1_D':   [3.1123942e+04, 4.3435688e+04, 6.397, 145.759],
    'M3_wrong_Ea4':  [4.99379e+04, 5.50000e+04, 9.75, 3.506],
    'M4_lumped_EF':  [5.0039241e+04, 5.5000000e+04, 9.883, 20.0],
    'M5_no_D':       [2577.581, 658929.499, 9735.492, 9822.344],
    'M6_author_mis': [3.5882909e+04, 5.5000000e+04, 38.713, 20.0],
}

X1a = pd.read_excel('X1.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
X2a = pd.read_excel('X2.xlsx')
Ya = pd.read_excel('Y.xlsx')['E_pur'].values.reshape(-1, 1)
M2a_meta = pd.read_excel('M2.xlsx')[['t2', 'T2', 'D0']]
n_pts_a = X2a.shape[1] // 6

X1m = pd.read_excel('X1_mis.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
X2m = pd.read_excel('X2_mis.xlsx')
Ym = pd.read_excel('Y_mis.xlsx')['E_pur'].values.reshape(-1, 1)
M2m_meta = pd.read_excel('M2_mis.xlsx')[['t2', 'T2', 'D0']]
n_pts_m = X2m.shape[1] // 6

N = len(X1a)
tgrid_a = np.arange(n_pts_a) * 15.0
tgrid_m = np.arange(n_pts_m) * 15.0

def build_ic_and_X2(X2df, n_pts):
    IC = np.array([[X2df[f'{s}_1'].iloc[i] for s in species] for i in range(len(X2df))])
    X2flat = np.array([[X2df[f'{s}_{k+1}'].iloc[i] for s in species for k in range(n_pts)] for i in range(len(X2df))])
    return IC, X2flat

IC_a, X2flat_a = build_ic_and_X2(X2a, n_pts_a)
IC_m, X2flat_m = build_ic_and_X2(X2m, n_pts_m)

def build_M2(rhs, IC, T2v, n_pts, tgrid, meta, params):
    M2 = np.zeros((N, 6, n_pts))
    for i in range(N):
        sol = odeint(rhs, IC[i], tgrid, args=(T2v[i], *params))
        M2[i] = sol.T
    return np.hstack([M2.reshape(N, -1), meta.values])

# ---------------------------------------------------------------------------
# Core PLS helpers
# ---------------------------------------------------------------------------
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

def three_block_decomposition(X1, block2, n2, block3, n3, Y, n1):
    """Generic X1(n1) -> block2(n2) -> block3(n3) SS decomposition."""
    muX1, sdX1, k1 = _scale_fit(X1); x1s = _scale_apply(X1, muX1, sdX1, k1)
    mu2, sd2, k2 = _scale_fit(block2); b2s = _scale_apply(block2, mu2, sd2, k2)
    mu3, sd3, k3 = _scale_fit(block3); b3s = _scale_apply(block3, mu3, sd3, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y - muY) / sdy
    sst = float(np.sum(ys ** 2))
    p1, T1, Q1, af = _fit(x1s, ys, n1); ae = min(n1, af); T1, Q1 = T1[:, :ae], Q1[:, :ae]
    ssx1 = float(np.sum((T1 @ Q1.T) ** 2))
    b2r, g2 = _resid(T1, b2s); b3_a, g3 = _resid(T1, b3s); ytr_a, gY = _resid(T1, ys)
    p2, T2v, Q2, bf = _fit(b2r, ytr_a, n2); be = min(n2, bf); T2v, Q2 = T2v[:, :be], Q2[:, :be]
    ss2 = float(np.sum((T2v @ Q2.T) ** 2))
    b3_b, g3b = _resid(T2v, b3_a); ytr_b, gYb = _resid(T2v, ytr_a)
    p3, T3, Q3, cf = _fit(b3_b, ytr_b, n3); ce = min(n3, cf); T3, Q3 = T3[:, :ce], Q3[:, :ce]
    ss3 = float(np.sum((T3 @ Q3.T) ** 2))
    ssf = sst - ssx1 - ss2 - ss3
    return ssx1, ss2, ss3, ssf, sst


rows = []
for name, rhs in MODELS.items():
    print(f"=== {name} ===")
    a, b, c = LV[name]
    params = FITTED[name]
    if name == 'M6_author_mis':
        X1_, T2v_, n_pts_, tgrid_, meta_, Y_ = X1m, M2m_meta['T2'].values, n_pts_m, tgrid_m, M2m_meta, Ym
        X2flat_ = X2flat_m; IC_ = IC_m
    else:
        X1_, T2v_, n_pts_, tgrid_, meta_, Y_ = X1a, M2a_meta['T2'].values, n_pts_a, tgrid_a, M2a_meta, Ya
        X2flat_ = X2flat_a; IC_ = IC_a

    M2_ = build_M2(rhs, IC_, T2v_, n_pts_, tgrid_, meta_, params)

    # FORWARD: X1(a) -> M2(b) -> X2(c)
    ssx1_f, ssM_f, ssX_f, ssF_f, sst = three_block_decomposition(X1_, M2_, b, X2flat_, c, Y_, a)
    # REVERSE: X1(a) -> X2(c) -> M2(b)
    ssx1_r, ssX_r, ssM_r, ssF_r, _ = three_block_decomposition(X1_, X2flat_, c, M2_, b, Y_, a)

    joint_f = ssM_f + ssX_f
    joint_r = ssM_r + ssX_r
    joint_diff = joint_f - joint_r

    shared_from_M = ssM_f - ssM_r
    shared_from_X = ssX_r - ssX_f
    shared_discrepancy = shared_from_M - shared_from_X
    shared_est = 0.5 * (shared_from_M + shared_from_X)

    unique_M = ssM_r
    unique_X = ssX_f

    m2m_forward = ssM_f / ssX_f if abs(ssX_f) > 1e-9 else np.nan
    m2m_reverse = ssM_r / ssX_r if abs(ssX_r) > 1e-9 else np.nan  # same SS_M/SS_X ratio, using the reverse-order fit's own shares

    # verify SS_Y = SS_X1 + unique_M + unique_X + shared + SS_F (both routes)
    recon_f = ssx1_f + unique_M + unique_X + shared_est + ssF_f
    recon_r = ssx1_r + unique_M + unique_X + shared_est + ssF_r

    rows.append(dict(model=name, LV_X1=a, LV_M2=b, LV_X2=c, SS_Y=sst, SS_X1=ssx1_f,
                      SS_M_forward=ssM_f, SS_X_forward=ssX_f, SS_F_forward=ssF_f,
                      SS_M_reverse=ssM_r, SS_X_reverse=ssX_r, SS_F_reverse=ssF_r,
                      joint_forward=joint_f, joint_reverse=joint_r, joint_diff=joint_diff,
                      shared_from_M=shared_from_M, shared_from_X=shared_from_X,
                      shared_discrepancy=shared_discrepancy, estimated_shared_SS=shared_est,
                      unique_M_SS=unique_M, unique_X_SS=unique_X,
                      Shared_pct_of_SSY=shared_est/sst*100,
                      M2M_forward=m2m_forward, M2M_reverse=m2m_reverse,
                      recon_check_forward=recon_f, recon_check_reverse=recon_r))
    print(f"  LV=({a},{b},{c})  SS_M_fwd={ssM_f:.3f} SS_X_fwd={ssX_f:.3f}  SS_X_rev={ssX_r:.3f} SS_M_rev={ssM_r:.3f}")
    print(f"  joint_fwd={joint_f:.3f}  joint_rev={joint_r:.3f}  diff={joint_diff:.3f}")
    print(f"  shared_from_M={shared_from_M:.3f}  shared_from_X={shared_from_X:.3f}  discrepancy={shared_discrepancy:.3f}")
    print(f"  estimated_shared_SS={shared_est:.3f} ({shared_est/sst*100:.1f}% of SS_Y)  unique_M={unique_M:.3f}  unique_X={unique_X:.3f}")

df = pd.DataFrame(rows)
print("\n===== FULL TABLE =====")
cols = ['model','SS_Y','SS_M_forward','SS_X_forward','SS_M_reverse','SS_X_reverse',
        'estimated_shared_SS','unique_M_SS','unique_X_SS','SS_F_forward','M2M_forward','M2M_reverse']
print(df[cols].to_string(index=False))

print("\n===== JOINT INVARIANCE CHECK (should be ~0 if shared variance is uniquely identifiable) =====")
print(df[['model','joint_forward','joint_reverse','joint_diff']].to_string(index=False))

print("\n===== SHARED-ESTIMATE DISCREPANCY (shared_from_M vs shared_from_X) =====")
print(df[['model','shared_from_M','shared_from_X','shared_discrepancy']].to_string(index=False))

df.to_excel('shared_variance_forward_reverse.xlsx', index=False)
print("\nSaved shared_variance_forward_reverse.xlsx")
