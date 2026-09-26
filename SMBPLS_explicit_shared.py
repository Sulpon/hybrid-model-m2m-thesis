"""
Extends the EXISTING SMB-PLS mechanism from sopls_r.py (the Xhat_k
projection: Xhat_k = ts @ pinv(ts'ts) @ ts' @ Xk, which is what currently
fuses shared M2/X2 information into the earlier block's super-score) to
produce explicit M2_unique / Shared / X2_unique blocks, instead of letting
that information get absorbed.

IMPORTANT FINDING (established before writing any numerics -- see the
STEP 4/8 discussion in the final report): a SINGLE forward pass (M2 first,
matching sopls_r.py's fixed X1->M2->X2 order) can only purify the LATER
block (X2_unique = X2 - Shared, where Shared = projection of X2 onto M2's
own PLS scores). It CANNOT purify M2 itself -- M2 is used whole in a
forward-only pass, exactly as found for plain SO-PLS. Producing a genuine
M2_unique requires the reverse pass (X2 first), which is NOT part of
sopls_r.py's implementation (hardcoded X1->M2->X2 order) and is added
here as the necessary, symmetric complement -- not a new algorithm, the
SAME Xhat_k-style projection, just applied in the other direction.

This means "Shared" ends up defined asymmetrically: Shared_onto_X2 (living
in X2's column space, from the forward/M2-first pass) and Shared_onto_M2
(living in M2's column space, from the reverse/X2-first pass) are
DIFFERENT matrices with DIFFERENT shapes (X2 has 42 columns, M2 has 45 --
species*timepoints plus t2/T2/D0). They cannot be added to the same
"M2_unique" or reconciled into one universal Shared matrix. This is
checked and reported explicitly in Step 4, not assumed away.
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
AMR_PREV = {  # from the established ablation run, for Step 7 comparison
    'M0_correct': 3.554, 'M1_no_side': 1.033, 'M2_order1_D': 2.994, 'M3_wrong_Ea4': 3.643,
    'M4_lumped_EF': 3.466, 'M5_no_D': 0.751, 'M6_author_mis': 1.110,
}
SO_PLS_PREV = {  # from the forward/reverse SO-PLS run last turn
    'M0_correct': dict(unique_M=21.505, shared=50.109, unique_X=5.706),
    'M1_no_side': dict(unique_M=16.757, shared=56.208, unique_X=11.267),
    'M2_order1_D': dict(unique_M=16.568, shared=55.392, unique_X=6.194),
    'M3_wrong_Ea4': dict(unique_M=21.493, shared=50.075, unique_X=5.729),
    'M4_lumped_EF': dict(unique_M=21.781, shared=50.123, unique_X=5.731),
    'M5_no_D': dict(unique_M=14.739, shared=39.076, unique_X=22.965),
    'M6_author_mis': dict(unique_M=11.354, shared=54.864, unique_X=12.366),
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
# Core helpers (same conventions as the established SO-PLS scripts:
# mean-center/unit-variance scale, block-scale by 1/sqrt(K))
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

def project_onto(T, target):
    """The Xhat_k = T @ pinv(T'T) @ T' @ target operation from sopls_r.py,
    generalized to a full (already-fit) score matrix T instead of one LV
    at a time -- the batch equivalent of the existing per-LV loop."""
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return T @ g


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

    # ---- scale (same convention as established scripts) ----
    muX1, sdX1, k1 = _scale_fit(X1_); x1s = _scale_apply(X1_, muX1, sdX1, k1)
    muM2, sdM2, k2 = _scale_fit(M2_); m2s = _scale_apply(M2_, muM2, sdM2, k2)
    muX2, sdX2, k3s = _scale_fit(X2flat_); x2s = _scale_apply(X2flat_, muX2, sdX2, k3s)
    muY, sdy = Y_.mean(0), Y_.std(0, ddof=1); ys = (Y_ - muY) / sdy
    sst = float(np.sum(ys ** 2))

    # ---- remove X1 first (same for both directions) ----
    p1, T1, Q1, af = _fit(x1s, ys, a); ae = min(a, af); T1, Q1 = T1[:, :ae], Q1[:, :ae]
    ssx1 = float(np.sum((T1 @ Q1.T) ** 2))
    m2_r1, _ = _resid(T1, m2s)     # M2 after removing X1
    x2_r1, _ = _resid(T1, x2s)     # X2 after removing X1
    ytr1, _ = _resid(T1, ys)

    # ---- FORWARD (M2-first, matches sopls_r.py's fixed order): T_M, Shared_onto_X2, X2_unique ----
    p_m, T_M, Q_m, bf = _fit(m2_r1, ytr1, b); be = min(b, bf); T_M = T_M[:, :be]
    Shared_onto_X2 = project_onto(T_M, x2_r1)          # lives in X2's column space (42 cols)
    X2_unique = x2_r1 - Shared_onto_X2                  # exact by construction

    # ---- REVERSE (X2-first, the necessary complement -- NOT in sopls_r.py's fixed order,
    #      added here as the same Xhat_k-style projection applied the other way): T_X, Shared_onto_M2, M2_unique ----
    p_x, T_X, Q_x, cf = _fit(x2_r1, ytr1, c); ce = min(c, cf); T_X = T_X[:, :ce]
    Shared_onto_M2 = project_onto(T_X, m2_r1)          # lives in M2's column space (45 cols)
    M2_unique = m2_r1 - Shared_onto_M2                  # exact by construction

    # ---- STEP 4: reconstruction checks ----
    recon_X2_err = float(np.max(np.abs((Shared_onto_X2 + X2_unique) - x2_r1)))
    recon_M2_err = float(np.max(np.abs((Shared_onto_M2 + M2_unique) - m2_r1)))
    shapes_match = (Shared_onto_X2.shape == M2_unique.shape)  # can Shared+M2_unique even be added?
    cross_recon_possible = shapes_match

    # ---- STEP 3: 4-block sequential fit X1 -> M2_unique -> Shared(X2-side) -> X2_unique -> Y ----
    # M2_unique and Shared_onto_X2 both used with LV_M2 components (their natural rank: M2_unique's
    # residual rank is <=b after removing a b-dim projection; Shared_onto_X2 has rank exactly <=b
    # since it's a projection onto a b-dim score space); X2_unique uses LV_X2 components.
    ytr_after_x1 = ytr1
    pB, TB, QB, bf2 = _fit(M2_unique, ytr_after_x1, b); be2 = min(b, bf2); TB, QB = TB[:, :be2], QB[:, :be2]
    ss_m_unique = float(np.sum((TB @ QB.T) ** 2))
    shared_r, _ = _resid(TB, Shared_onto_X2); x2u_r, _ = _resid(TB, X2_unique); ytr_after_b, _ = _resid(TB, ytr_after_x1)

    pC, TC, QC, cf2 = _fit(shared_r, ytr_after_b, b); ce2 = min(b, cf2); TC, QC = TC[:, :ce2], QC[:, :ce2]
    ss_shared = float(np.sum((TC @ QC.T) ** 2))
    x2u_r2, _ = _resid(TC, x2u_r); ytr_after_c, _ = _resid(TC, ytr_after_b)

    pD, TD, QD, df2 = _fit(x2u_r2, ytr_after_c, c); de2 = min(c, df2); TD, QD = TD[:, :de2], QD[:, :de2]
    ss_x_unique = float(np.sum((TD @ QD.T) ** 2))

    ss_f = sst - ssx1 - ss_m_unique - ss_shared - ss_x_unique

    m2m_original = ssx1 and None  # placeholder not used
    ssM_forward_orig = float(np.sum((T_M @ np.linalg.pinv(T_M.T@T_M)@T_M.T@ytr1)**2)) if False else None
    # M2M_original = SS_M_forward / SS_X_forward, from the plain forward SO-PLS (M2 whole, not split)
    ssM_forward_whole = float(np.sum((T_M @ (np.linalg.pinv(T_M.T @ T_M) @ (T_M.T @ ytr1))) ** 2))
    ssX_forward_whole, _g = None, None
    x2_after_m_whole, _ = _resid(T_M, x2_r1); ytr_after_m_whole, _ = _resid(T_M, ytr1)
    pXw, TXw, QXw, cfw = _fit(x2_after_m_whole, ytr_after_m_whole, c); cew = min(c, cfw); TXw, QXw = TXw[:, :cew], QXw[:, :cew]
    ssX_forward_whole = float(np.sum((TXw @ QXw.T) ** 2))
    m2m_original = ssM_forward_whole / ssX_forward_whole if abs(ssX_forward_whole) > 1e-9 else np.nan
    m2m_unique = ss_m_unique / ss_x_unique if abs(ss_x_unique) > 1e-9 else np.nan

    row = dict(model=name, LV_X1=a, LV_M2=b, LV_X2=c,
               SS_X1=ssx1, SS_M_unique=ss_m_unique, SS_shared=ss_shared, SS_X_unique=ss_x_unique,
               SS_F=ss_f, SS_Y=sst,
               M_unique_pct=ss_m_unique/sst*100, Shared_pct=ss_shared/sst*100,
               X_unique_pct=ss_x_unique/sst*100, Residual_pct=ss_f/sst*100,
               M2M_original=m2m_original, M2M_unique=m2m_unique,
               recon_X2_err=recon_X2_err, recon_M2_err=recon_M2_err,
               shapes_match_for_cross_recon=shapes_match,
               X2_ncols=Shared_onto_X2.shape[1], M2_ncols=M2_unique.shape[1],
               AMR=AMR_PREV[name],
               SOPLS_unique_M=SO_PLS_PREV[name]['unique_M'], SOPLS_shared=SO_PLS_PREV[name]['shared'],
               SOPLS_unique_X=SO_PLS_PREV[name]['unique_X'])
    rows.append(row)
    print(f"  LV=({a},{b},{c})  SS_M_unique={ss_m_unique:.3f}  SS_shared={ss_shared:.3f}  SS_X_unique={ss_x_unique:.3f}  SS_F={ss_f:.3f}")
    print(f"  M2M_original={m2m_original:.3f}  M2M_unique={m2m_unique:.3f}")
    print(f"  recon check: X2_unique+Shared(X2side) vs X2 residual, max abs err={recon_X2_err:.2e} (trivial, by construction)")
    print(f"  recon check: M2_unique+Shared(M2side) vs M2 residual, max abs err={recon_M2_err:.2e} (trivial, by construction)")
    print(f"  CAN M2_unique + Shared(X2-side block used in the 4-block model) reconstruct M2_original? "
          f"shapes: Shared={Shared_onto_X2.shape[1]} cols vs M2={M2_unique.shape[1]} cols -> {'SAME' if shapes_match else 'DIFFERENT, NOT ADDABLE'}")

df = pd.DataFrame(rows)
print("\n===== SUMMARY TABLE =====")
print(df[['model','SS_X1','SS_M_unique','SS_shared','SS_X_unique','SS_F','SS_Y',
          'M_unique_pct','Shared_pct','X_unique_pct','Residual_pct','M2M_original','M2M_unique']].to_string(index=False))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with pd.ExcelWriter('SMBPLS_explicit_shared.xlsx') as w:
    df[['model','LV_X1','LV_M2','LV_X2','SS_X1','SS_M_unique','SS_shared','SS_X_unique','SS_F','SS_Y',
        'M_unique_pct','Shared_pct','X_unique_pct','Residual_pct','M2M_original','M2M_unique']].to_excel(w, sheet_name='summary', index=False)
    df[['model','SS_X1','SS_M_unique','SS_shared','SS_X_unique','SS_F']].to_excel(w, sheet_name='block_contributions', index=False)
    df[['model','SS_M_unique','SS_shared','SS_X_unique','SOPLS_unique_M','SOPLS_shared','SOPLS_unique_X']].to_excel(w, sheet_name='comparison_SO_PLS', index=False)
    df[['model','M2M_unique','AMR']].to_excel(w, sheet_name='comparison_AMR', index=False)
    df[['model','recon_X2_err','recon_M2_err','shapes_match_for_cross_recon','X2_ncols','M2_ncols']].to_excel(w, sheet_name='reconstruction_checks', index=False)
print("\nSaved SMBPLS_explicit_shared.xlsx")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

order = df['model'].tolist()
fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(order))
for col, color, label in [('SS_M_unique', '#3E7C8C', 'M2_unique'), ('SS_shared', '#D97E3A', 'Shared'),
                           ('SS_X_unique', '#B5533C', 'X2_unique'), ('SS_F', '#888888', 'Residual')]:
    vals = df[col].values
    ax.bar(order, vals, bottom=bottom, label=label, color=color)
    bottom += vals
ax.set_ylabel('SS (in-sample, calibration)'); ax.set_title('Explicit SMB-PLS decomposition: M2_unique / Shared / X2_unique / Residual')
ax.legend(); ax.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig('SMBPLS_explicit_shared_contributions.png', dpi=140, facecolor='white')
print("Saved SMBPLS_explicit_shared_contributions.png")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df['AMR'], df['M2M_unique'], s=70, color='#16323F')
for _, r in df.iterrows():
    ax.annotate(r['model'].replace('_',' '), (r['AMR'], r['M2M_unique']), fontsize=8)
ax.set_xlabel('AMR (cross-validated)'); ax.set_ylabel('M2M_unique (in-sample, SS_M_unique/SS_X_unique)')
ax.set_title('M2M_unique vs AMR'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('SMBPLS_M2unique_vs_AMR.png', dpi=140, facecolor='white')
print("Saved SMBPLS_M2unique_vs_AMR.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(order, df['Shared_pct'], color='#D97E3A')
ax.set_ylabel('Shared % of SS_Y'); ax.set_title('Shared M2/X2 contribution, all 7 models')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig('SMBPLS_shared_contribution.png', dpi=140, facecolor='white')
print("Saved SMBPLS_shared_contribution.png")
