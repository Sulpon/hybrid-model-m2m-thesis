"""
Extracts T_shared exactly as computed by the existing SMB-PLS algorithm
(sopls_r.py's per-LV `tk` at M2's stage, BEFORE fusion into the super-score
tT), for all 7 models. Verified (previous turn, M0): tk is exactly
proportional to ts at every LV (rank-1 Xhat_k, since X2 is the only block
after M2) -- so T_shared spans the identical subspace as T_M. This is a
structural property of the 3-block layout, not re-derived per model, but
T_shared/T_M are still computed via the ACTUAL NIPALS loop (not shortcut)
for every model, since the loadings/exact values are what's needed here.

Then, per the user's explicit specification:
  M2_shared = T_shared @ P_M'   (P_M = M2's own loadings on T_shared)
  M2_unique = M2 - M2_shared    (M2's own residual w.r.t. T_shared)
  X2_shared = T_shared @ P_X'
  X2_unique = X2 - X2_shared
Both M2_unique and X2_unique are exact OLS/projection residuals w.r.t.
T_shared, hence exactly orthogonal to T_shared (standard projection
theorem) -- verified numerically below.

Sequential fit, exactly as specified: X1 -> M2_unique -> Shared -> X2_unique -> Y.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

R = 8.314
species = ['A', 'B', 'C', 'D', 'E', 'F']

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
    'M0_correct': ode_correct, 'M1_no_side': ode_no_side, 'M2_order1_D': ode_order1_D,
    'M3_wrong_Ea4': ode_wrong_Ea4, 'M4_lumped_EF': ode_lumped, 'M5_no_D': ode_no_D,
    'M6_author_mis': ode_author_mis,
}
LV = {
    'M0_correct': (4, 6, 2), 'M1_no_side': (1, 13, 4), 'M2_order1_D': (4, 6, 4),
    'M3_wrong_Ea4': (4, 6, 2), 'M4_lumped_EF': (4, 8, 2), 'M5_no_D': (3, 7, 4),
    'M6_author_mis': (3, 3, 6),
}
FITTED = {
    'M0_correct': [4.9380789e+04, 5.6481086e+04, 8.023, 33.615],
    'M1_no_side': [5.0088694e+04, 5.5000000e+04, 11.377, 20.0],
    'M2_order1_D': [3.1123942e+04, 4.3435688e+04, 6.397, 145.759],
    'M3_wrong_Ea4': [4.99379e+04, 5.50000e+04, 9.75, 3.506],
    'M4_lumped_EF': [5.0039241e+04, 5.5000000e+04, 9.883, 20.0],
    'M5_no_D': [2577.581, 658929.499, 9735.492, 9822.344],
    'M6_author_mis': [3.5882909e+04, 5.5000000e+04, 38.713, 20.0],
}
AMR_PREV = {'M0_correct': 3.554, 'M1_no_side': 1.033, 'M2_order1_D': 2.994, 'M3_wrong_Ea4': 3.643,
            'M4_lumped_EF': 3.466, 'M5_no_D': 0.751, 'M6_author_mis': 1.110}
M2M_UNIQUE_PREV = {'M0_correct': 24.707, 'M1_no_side': 7.665, 'M2_order1_D': 15.335, 'M3_wrong_Ea4': 27.511,
                   'M4_lumped_EF': 25.867, 'M5_no_D': 6.811, 'M6_author_mis': 1.230}
SOPLS_SHARED_PREV = {'M0_correct': 50.109, 'M1_no_side': 56.208, 'M2_order1_D': 55.392, 'M3_wrong_Ea4': 50.075,
                     'M4_lumped_EF': 50.123, 'M5_no_D': 39.076, 'M6_author_mis': 54.864}
SOPLS_UNIQUE_M_PREV = {'M0_correct': 21.505, 'M1_no_side': 16.757, 'M2_order1_D': 16.568, 'M3_wrong_Ea4': 21.493,
                       'M4_lumped_EF': 21.781, 'M5_no_D': 14.739, 'M6_author_mis': 11.354}
M2M_ORIGINAL_PREV = {'M0_correct': 12.569, 'M1_no_side': 6.543, 'M2_order1_D': 11.619, 'M3_wrong_Ea4': 12.509,
                     'M4_lumped_EF': 12.571, 'M5_no_D': 2.369, 'M6_author_mis': 5.335}

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
# EXACT sopls_r.py preprocessing
# ---------------------------------------------------------------------------
def fit_scaler(X):
    mean = X.mean(0); std = X.std(0, ddof=1); std[std == 0] = 1
    return mean, std
def apply_scaler(X, mean, std): return (X - mean) / std
def scale_full(X):
    m, s = fit_scaler(X); return apply_scaler(X, m, s)

def project_onto(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return T @ g

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0] - 1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n
def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g

def run_smbpls_extract_Tshared(X1n, M2n, X2n, Y_s, a, b):
    """EXACT NIPALS loop from sopls_r.py, X1 stage (a LVs) then M2 stage (b LVs),
    capturing ts (->T_M) and tk (->T_shared) per LV at the M2 stage."""
    X_current = [X1n.copy(), M2n.copy(), X2n.copy()]
    Y_current = Y_s.copy()
    # X1 stage
    for _a in range(a):
        u = Y_current.copy(); t_old = None
        for _ in range(500):
            Xs = X_current[0]
            ws = Xs.T @ u / (u.T @ u); ws = ws / np.linalg.norm(ws); ts = Xs @ ws
            block_scores = [ts]
            for k in range(1, 3):
                Xk = X_current[k]
                Xhat_k = ts @ np.linalg.pinv(ts.T @ ts) @ ts.T @ Xk
                wk = Xhat_k.T @ u / (u.T @ u); wk = wk / np.linalg.norm(wk); tk = Xhat_k @ wk
                block_scores.append(tk)
            T = np.hstack(block_scores)
            wT = T.T @ u / (u.T @ u); wT = wT / np.linalg.norm(wT); tT = T @ wT
            q = Y_current.T @ tT / (tT.T @ tT); u_new = Y_current @ q / (q.T @ q)
            if t_old is not None and np.linalg.norm(tT - t_old) < 1e-10: break
            t_old = tT.copy(); u = u_new.copy()
        Yhat_a = tT @ q.T
        for k in range(3):
            Xk = X_current[k]; pk = Xk.T @ tT / (tT.T @ tT); X_current[k] = Xk - tT @ pk.T
        Y_current = Y_current - Yhat_a

    # M2 stage: capture ts, tk explicitly
    ts_list, tk_list = [], []
    for _b in range(b):
        u = Y_current.copy(); t_old = None
        for _ in range(500):
            Xs = X_current[1]
            ws = Xs.T @ u / (u.T @ u); ws = ws / np.linalg.norm(ws); ts = Xs @ ws
            Xk = X_current[2]
            Xhat_k = ts @ np.linalg.pinv(ts.T @ ts) @ ts.T @ Xk
            wk = Xhat_k.T @ u / (u.T @ u); wk = wk / np.linalg.norm(wk); tk = Xhat_k @ wk
            T = np.hstack([ts, tk])
            wT = T.T @ u / (u.T @ u); wT = wT / np.linalg.norm(wT); tT = T @ wT
            q = Y_current.T @ tT / (tT.T @ tT); u_new = Y_current @ q / (q.T @ q)
            if t_old is not None and np.linalg.norm(tT - t_old) < 1e-10: break
            t_old = tT.copy(); u = u_new.copy()
        ts_list.append(ts.copy()); tk_list.append(tk.copy())
        Yhat_a = tT @ q.T
        for k in range(3):
            Xk = X_current[k]; pk = Xk.T @ tT / (tT.T @ tT); X_current[k] = Xk - tT @ pk.T
        Y_current = Y_current - Yhat_a

    T_M = np.hstack(ts_list)
    T_shared = np.hstack(tk_list)
    return T_M, T_shared


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

    M2raw = build_M2(rhs, IC_, T2v_, n_pts_, tgrid_, meta_, params)
    X1s = scale_full(X1_); M2s = scale_full(M2raw); X2s = scale_full(X2flat_); Ys = scale_full(Y_)
    X1n = X1s / np.sqrt(X1s.shape[1]); M2n = M2s / np.sqrt(M2s.shape[1]); X2n = X2s / np.sqrt(X2s.shape[1])

    T_M, T_shared = run_smbpls_extract_Tshared(X1n, M2n, X2n, Ys, a, b)
    print(f"  rank T_M={np.linalg.matrix_rank(T_M)}  rank T_shared={np.linalg.matrix_rank(T_shared)}  "
          f"combined rank={np.linalg.matrix_rank(np.hstack([T_M,T_shared]))}")

    # ---- need M2/X2 AFTER removing X1 (for computing M2_shared, X2_shared in their OWN normalized space) ----
    p1, T1, Q1, af = _fit(X1n, Ys, a); ae = min(a, af); T1 = T1[:, :ae]
    m2_r1, _ = _resid(T1, M2n); x2_r1, _ = _resid(T1, X2n); ytr1, _ = _resid(T1, Ys)

    M2_shared = project_onto(T_shared, m2_r1)
    M2_unique = m2_r1 - M2_shared
    X2_shared = project_onto(T_shared, x2_r1)
    X2_unique = x2_r1 - X2_shared

    # orthogonality check (should be ~0, exact by projection theorem)
    orth_check_M2 = float(np.max(np.abs(T_shared.T @ M2_unique)))
    orth_check_X2 = float(np.max(np.abs(T_shared.T @ X2_unique)))

    # ---- sequential fit: X1 -> M2_unique -> Shared(T_shared) -> X2_unique -> Y ----
    sst = float(np.sum(ytr1 ** 2)) + float(np.sum((T1 @ Q1[:, :ae].T) ** 2))  # SST total (=99, sanity)
    ssx1 = float(np.sum((T1 @ Q1[:, :ae].T) ** 2))

    pB, TB, QB, bf = _fit(M2_unique, ytr1, b); be = min(b, bf); TB, QB = TB[:, :be], QB[:, :be]
    ss_m_unique = float(np.sum((TB @ QB.T) ** 2))
    shared_r, _ = _resid(TB, T_shared); x2u_r, _ = _resid(TB, X2_unique); ytr_b, _ = _resid(TB, ytr1)

    # check dilution: does residualizing T_shared against TB change it?
    dilution = float(np.max(np.abs(shared_r - T_shared))) / (float(np.max(np.abs(T_shared))) + 1e-12)

    pC, TC, QC, cf = _fit(shared_r, ytr_b, b); ce = min(b, cf); TC, QC = TC[:, :ce], QC[:, :ce]
    ss_shared = float(np.sum((TC @ QC.T) ** 2))
    x2u_r2, _ = _resid(TC, x2u_r); ytr_c, _ = _resid(TC, ytr_b)

    pD, TD, QD, df_ = _fit(x2u_r2, ytr_c, c); ce2 = min(c, df_); TD, QD = TD[:, :ce2], QD[:, :ce2]
    ss_x_unique = float(np.sum((TD @ QD.T) ** 2))

    ss_f = sst - ssx1 - ss_m_unique - ss_shared - ss_x_unique

    m2m_unique_new = ss_m_unique / ss_x_unique if abs(ss_x_unique) > 1e-9 else np.nan

    row = dict(model=name, LV_X1=a, LV_M2=b, LV_X2=c, SS_X1=ssx1,
               SS_M_unique=ss_m_unique, SS_shared=ss_shared, SS_X_unique=ss_x_unique, SS_F=ss_f, SS_Y=sst,
               M_unique_pct=ss_m_unique/sst*100, Shared_pct=ss_shared/sst*100,
               X_unique_pct=ss_x_unique/sst*100, Residual_pct=ss_f/sst*100,
               M2M_Tshared=m2m_unique_new,
               dilution_of_shared_pct=dilution*100, orth_check_M2=orth_check_M2, orth_check_X2=orth_check_X2,
               AMR=AMR_PREV[name], M2M_original=M2M_ORIGINAL_PREV[name],
               M2M_unique_reverse_pass=M2M_UNIQUE_PREV[name],
               SOPLS_shared_reverse=SOPLS_SHARED_PREV[name], SOPLS_unique_M_reverse=SOPLS_UNIQUE_M_PREV[name])
    rows.append(row)
    print(f"  SS_M_unique={ss_m_unique:.3f}  SS_shared={ss_shared:.3f}  SS_X_unique={ss_x_unique:.3f}  SS_F={ss_f:.3f}")
    print(f"  dilution of Shared from sequencing after M2_unique: {dilution*100:.4f}%  (orth checks: {orth_check_M2:.2e}, {orth_check_X2:.2e})")
    print(f"  M2M_Tshared={m2m_unique_new:.3f}   [compare: M2M_original={M2M_ORIGINAL_PREV[name]:.3f}, "
          f"M2M_unique(reverse-pass)={M2M_UNIQUE_PREV[name]:.3f}, AMR={AMR_PREV[name]:.3f}]")

df = pd.DataFrame(rows)
print("\n===== FULL TABLE =====")
print(df[['model','SS_X1','SS_M_unique','SS_shared','SS_X_unique','SS_F','SS_Y',
          'M_unique_pct','Shared_pct','X_unique_pct','Residual_pct']].to_string(index=False))
print("\n===== COMPARISON: M2M_Tshared vs M2M_original vs M2M_unique(reverse) vs AMR =====")
print(df[['model','M2M_Tshared','M2M_original','M2M_unique_reverse_pass','AMR']].to_string(index=False))
print("\n===== SS_M_unique (this method) vs SS_M_unique (reverse-pass SO-PLS) =====")
print(df[['model','SS_M_unique','SOPLS_unique_M_reverse']].to_string(index=False))

df.to_excel('SMBPLS_Tshared_explicit.xlsx', index=False)
print("\nSaved SMBPLS_Tshared_explicit.xlsx")
