"""
Ablation + AMR for all 7 candidate models (M0-M6), using the REAL data
directly (no SDE regeneration needed -- we already have the ground truth):
  - M0-M5: real X1.xlsx/X2.xlsx/Y.xlsx (100 batches, shared)
  - M6:    real X1_mis.xlsx/X2_mis.xlsx/Y_mis.xlsx (100 batches, its own)

For each model, M2 is built by re-integrating that model's mechanism from
each real batch's own X2(t=0) and real T2, with [Ea3,Ea4,A3,A4] FIT via
least_squares against that model's own real X2 data (matching the
validated approach for M6/case ii, and faithful_gen.py's convention of
fitting every candidate rather than only the misspecified ones). M2 has
t2/T2/D0 appended, matching the real M2.xlsx/M2_mis.xlsx structure exactly.

Ablation structures: FULL (X1+M2+X2), NO_M2 (X1+X2), NO_X2 (X1+M2),
M2_ONLY, X1_ONLY. FULL's own LV is the blind-argmin over the full grid;
NO_M2/NO_X2/M2_ONLY are evaluated at the SAME LV_X1/LV_M2/LV_X2 as FULL
(no independent re-optimization), matching the corrected, validated
methodology from ablation_recalc_paper_consistent.py.

AMR = |Q2_FULL - Q2_NO_M2| / |Q2_FULL - Q2_NO_X2|
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

R = 8.314
A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42
species = ['A', 'B', 'C', 'D', 'E', 'F']

# ---------------------------------------------------------------------------
# 7 candidate mechanisms (verified against model_definitions.md / faithful_gen.py)
# ---------------------------------------------------------------------------
def ode_correct(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB*CD**2; r4 = k4*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]

def ode_no_side(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, _, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2))
    r3 = k3*CB*CD**2
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
    k3 = A3*np.exp(-Ea3/(R*T2))
    r3 = k3*CB*CD**2; r4 = k3*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]

def ode_no_D(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, _, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB; r4 = k4*CC
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]

def ode_author_mis(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, _, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2))
    r3 = k3*CB*CD
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

# ---------------------------------------------------------------------------
# Load real data
# ---------------------------------------------------------------------------
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


def fit_and_build_M2(rhs, IC, T2v, X2df, n_pts, tgrid, meta):
    """Fit [Ea3,Ea4,A3,A4] via least_squares against the real X2 trajectories,
    then build M2 (species blocks + t2,T2,D0), matching the real M2.xlsx structure."""
    def resid(p):
        Ea3, Ea4, A3, A4 = p
        out = []
        for i in range(N):
            sol = odeint(rhs, IC[i], tgrid, args=(T2v[i], Ea3, Ea4, A3, A4))
            obs = np.array([[X2df[f'{s}_{k+1}'].iloc[i] for s in species] for k in range(n_pts)])
            out.append((sol - obs).ravel())
        return np.concatenate(out)
    r = least_squares(resid, [50000., 55000., 10., 20.],
                       bounds=([1e3, 1e3, 1e-4, 1e-4], [1e6, 1e6, 1e4, 1e4]),
                       xtol=1e-8, ftol=1e-8, max_nfev=60)
    p = r.x
    M2 = np.zeros((N, 6, n_pts))
    for i in range(N):
        sol = odeint(rhs, IC[i], tgrid, args=(T2v[i], *p))
        M2[i] = sol.T
    M2flat = M2.reshape(N, -1)
    M2flat = np.hstack([M2flat, meta.values])
    return p, M2flat


# ---------------------------------------------------------------------------
# Core SO-PLS / ablation machinery (identical to prior validated scripts)
# ---------------------------------------------------------------------------
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

def cv_1block(X, Y, MAXn):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, MAXn)); rf = np.zeros((N_SPLITS, MAXn))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muX, sdX, k = _scale_fit(X[tr]); xt, xe = _scale_apply(X[tr], muX, sdX, k), _scale_apply(X[te], muX, sdX, k)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr = (Y[tr]-muY)/sdy
        p, Tt, Q, af = _fit(xt, ytr, MAXn); Te = p.transform(xe)
        for a in range(1, MAXn+1):
            ae = min(a, af)
            yh = (Te[:, :ae] @ Q[:, :ae].T) * sdy + muY
            P[te, a-1] = yh.ravel()
            rf[f, a-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    q2 = 1 - ((P - Y)**2).sum(0)/sst
    return rf.mean(0), q2

def cv_2block(Xa, Xb, Y, MAXa, MAXb):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, MAXa, MAXb)); rf = np.zeros((N_SPLITS, MAXa, MAXb))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muXa, sdXa, ka = _scale_fit(Xa[tr]); xat, xae = _scale_apply(Xa[tr],muXa,sdXa,ka), _scale_apply(Xa[te],muXa,sdXa,ka)
        muXb, sdXb, kb = _scale_fit(Xb[tr]); xbt0, xbe0 = _scale_apply(Xb[tr],muXb,sdXb,kb), _scale_apply(Xb[te],muXb,sdXb,kb)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr0 = (Y[tr]-muY)/sdy
        p1, Tat, Qa, af = _fit(xat, ytr0, MAXa); Tae = p1.transform(xae)
        for a in range(1, MAXa+1):
            ae = min(a, af); Ta, Tae_ = Tat[:,:ae], Tae[:,:ae]
            yh_a = Tae_ @ Qa[:,:ae].T
            xbt, gXb = _resid(Ta, xbt0); xbe = xbe0 - Tae_ @ gXb
            ytr_a, gY = _resid(Ta, ytr0)
            p2, Tbt, Qb, bf = _fit(xbt, ytr_a, MAXb); Tbe = p2.transform(xbe)
            for b in range(1, MAXb+1):
                be = min(b, bf)
                yh = (yh_a + Tbe[:,:be] @ Qb[:,:be].T)*sdy+muY
                P[te,a-1,b-1] = yh.ravel()
                rf[f,a-1,b-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    q2 = 1 - ((P - Y[:,:,None])**2).sum(0)/sst
    return rf.mean(0), q2

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


# ---------------------------------------------------------------------------
# Run for all 7 models
# ---------------------------------------------------------------------------
print("X1_ONLY (shared across all models)...")
rmsecv_X1o, q2_X1o = cv_1block(X1a, Ya, A_MAX)  # X1 is the same real block for M0-M5; M6 uses X1m separately below

rows = []
for name, rhs in MODELS.items():
    print(f"\n=== {name} ===")
    if name == 'M6_author_mis':
        X1_, IC_, X2df_, n_pts_, tgrid_, meta_, Y_ = X1m, IC_m, X2m, n_pts_m, tgrid_m, M2m_meta, Ym
        T2v_ = meta_['T2'].values
        X2flat_ = X2flat_m
    else:
        X1_, IC_, X2df_, n_pts_, tgrid_, meta_, Y_ = X1a, IC_a, X2a, n_pts_a, tgrid_a, M2a_meta, Ya
        T2v_ = meta_['T2'].values
        X2flat_ = X2flat_a

    p, M2_ = fit_and_build_M2(rhs, IC_, T2v_, X2df_, n_pts_, tgrid_, meta_)
    print(f"  fitted [Ea3,Ea4,A3,A4] = {np.round(p,3)}")

    rmsecv_full, q2_full = cv_3block(X1_, M2_, X2flat_, Y_)
    i_best = np.unravel_index(rmsecv_full.argmin(), rmsecv_full.shape)
    a, b, c = i_best[0]+1, i_best[1]+1, i_best[2]+1
    RF, QF = rmsecv_full[i_best], q2_full[i_best]

    rmsecv_nm2, q2_nm2 = cv_2block(X1_, X2flat_, Y_, A_MAX, C_MAX)
    rmsecv_nx2, q2_nx2 = cv_2block(X1_, M2_, Y_, A_MAX, B_MAX)
    rmsecv_m2o, q2_m2o = cv_1block(M2_, Y_, B_MAX)

    if name == 'M6_author_mis':
        RX1O, QX1O = cv_1block(X1_, Y_, A_MAX)
        RX1O, QX1O = RX1O[a-1], QX1O[a-1]
    else:
        RX1O, QX1O = rmsecv_X1o[a-1], q2_X1o[a-1]

    RNM2, QNM2 = rmsecv_nm2[a-1, c-1], q2_nm2[a-1, c-1]
    RNX2, QNX2 = rmsecv_nx2[a-1, b-1], q2_nx2[a-1, b-1]
    RM2O, QM2O = rmsecv_m2o[b-1], q2_m2o[b-1]

    d_m2_pp = (QF - QNM2) * 100
    d_x2_pp = (QF - QNX2) * 100
    amr = abs(d_m2_pp) / abs(d_x2_pp) if abs(d_x2_pp) > 1e-9 else np.nan
    umc = (RNM2 - RF) / RNM2
    uxc = (RNX2 - RF) / RNX2
    mir = umc / (umc + uxc) if abs(umc + uxc) > 1e-9 else np.nan
    mar = RF / RM2O

    rows.append(dict(model=name, LV_X1=a, LV_M2=b, LV_X2=c, RMSECV_FULL=RF, Q2_FULL=QF,
                      Q2_NO_M2=QNM2, Q2_NO_X2=QNX2, Q2_M2_ONLY=QM2O, Q2_X1_ONLY=QX1O,
                      dQ2_drop_M2_pp=d_m2_pp, dQ2_drop_X2_pp=d_x2_pp, AMR=amr,
                      UMC=umc, UXC=uxc, MIR=mir, MAR=mar))
    print(f"  LV=({a},{b},{c})  Q2_FULL={QF:.4f}  Q2_NO_M2={QNM2:.4f}  Q2_NO_X2={QNX2:.4f}  "
          f"dQ2_M2={d_m2_pp:.2f}pp  dQ2_X2={d_x2_pp:.2f}pp  AMR={amr:.3f}  MAR={mar:.3f}")

result_df = pd.DataFrame(rows)
print("\n===== FINAL TABLE: ablation + AMR, all 7 models, real data =====")
print(result_df.to_string(index=False))
result_df.to_excel('ablation_AMR_all_models_real_data.xlsx', index=False)
print("\nSaved ablation_AMR_all_models_real_data.xlsx")
