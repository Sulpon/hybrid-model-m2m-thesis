"""
Direct comparison of SO-PLS variance decomposition (M2M) vs ablation
(AMR), plus a shared/complementary-information investigation for both
approaches, for all 7 candidate models. Uses the REAL data (X1.xlsx/
X2.xlsx/Y.xlsx for M0-M5, X1_mis.xlsx/X2_mis.xlsx/Y_mis.xlsx for M6) and
the EXISTING established results: the FULL-model LV allocation and fitted
kinetic parameters are reused as-is from ablation_AMR_all_models_real_data.py
(both are deterministic outputs of that already-run, validated procedure --
re-deriving them here would only reproduce the same numbers at the cost of
re-running the full 1350-point CV grid search and curve_fit for each model).
Only the new quantities (in-sample SS decomposition, X1-only Q2 at the
matching LV, and the ablation overlap) are computed here, all at the SAME
fixed (a,b,c) per model -- no independent re-optimization of any block.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

R = 8.314
N_SPLITS, SEED = 10, 42
species = ['A', 'B', 'C', 'D', 'E', 'F']

# ---------------------------------------------------------------------------
# 7 candidate mechanisms (identical to ablation_AMR_all_models_real_data.py)
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

# Reused as-is from the established ablation_AMR_all_models_real_data.py run
# (blind-argmin LV search + least_squares fit -- both deterministic; not re-derived here)
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


def build_M2(rhs, IC, T2v, n_pts, tgrid, meta, params):
    M2 = np.zeros((N, 6, n_pts))
    for i in range(N):
        sol = odeint(rhs, IC[i], tgrid, args=(T2v[i], *params))
        M2[i] = sol.T
    M2flat = M2.reshape(N, -1)
    return np.hstack([M2flat, meta.values])


# ---------------------------------------------------------------------------
# Core PLS helpers (identical to every prior validated script)
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


def full_fit_decomposition(X1, M2, X2, Y, a, b, c):
    """In-sample (no CV) SS decomposition at a SPECIFIC (a,b,c) -- SSX1,SSM,SSX,SSF,SST."""
    muX1, sdX1, k1 = _scale_fit(X1); x1s = _scale_apply(X1, muX1, sdX1, k1)
    muM2, sdM2, k2 = _scale_fit(M2); m2s = _scale_apply(M2, muM2, sdM2, k2)
    muX2, sdX2, k3 = _scale_fit(X2); x2s = _scale_apply(X2, muX2, sdX2, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y - muY) / sdy
    sst = float(np.sum(ys ** 2))
    p1, T1, Q1, af = _fit(x1s, ys, a); ae = min(a, af); T1, Q1 = T1[:, :ae], Q1[:, :ae]
    ssx1 = float(np.sum((T1 @ Q1.T) ** 2))
    m2r, gM = _resid(T1, m2s); x2_a, gX = _resid(T1, x2s); ytr_a, gY = _resid(T1, ys)
    p2, T2v, Q2, bf = _fit(m2r, ytr_a, b); be = min(b, bf); T2v, Q2 = T2v[:, :be], Q2[:, :be]
    ssm = float(np.sum((T2v @ Q2.T) ** 2))
    x2_b, gX2 = _resid(T2v, x2_a); ytr_b, gY2 = _resid(T2v, ytr_a)
    p3, T3, Q3, cf = _fit(x2_b, ytr_b, c); ce = min(c, cf); T3, Q3 = T3[:, :ce], Q3[:, :ce]
    ssx = float(np.sum((T3 @ Q3.T) ** 2))
    ssf = sst - ssx1 - ssm - ssx
    return ssx1, ssm, ssx, ssf, sst


def cv_q2_3block_at(X1, M2, X2, Y, a, b, c):
    """CV Q2 of FULL (X1(a)->M2(b)->X2(c)) at a SPECIFIC point (no grid scan)."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, 1))
    for tr, te in cv.split(Y):
        muX1, sdX1, k1 = _scale_fit(X1[tr]); x1t, x1e = _scale_apply(X1[tr],muX1,sdX1,k1), _scale_apply(X1[te],muX1,sdX1,k1)
        muM2, sdM2, k2 = _scale_fit(M2[tr]); m2t0, m2e0 = _scale_apply(M2[tr],muM2,sdM2,k2), _scale_apply(M2[te],muM2,sdM2,k2)
        muX2, sdX2, k3 = _scale_fit(X2[tr]); x2t0, x2e0 = _scale_apply(X2[tr],muX2,sdX2,k3), _scale_apply(X2[te],muX2,sdX2,k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(x1t, ytr0, a); ae = min(a, af); T1t, T1e = T1t[:,:ae], p1.transform(x1e)[:,:ae]
        yh_a = T1e @ Q1[:,:ae].T
        m2t, gM = _resid(T1t, m2t0); m2e = m2e0 - T1e @ gM
        x2t_a, gX = _resid(T1t, x2t0); x2e_a = x2e0 - T1e @ gX
        ytr_a, gY = _resid(T1t, ytr0)
        p2, T2t, Q2, bf = _fit(m2t, ytr_a, b); be = min(b, bf); T2t, T2e = T2t[:,:be], p2.transform(m2e)[:,:be]
        yh_ab = yh_a + T2e @ Q2[:,:be].T
        x2t_b, gX2 = _resid(T2t, x2t_a); x2e_b = x2e_a - T2e @ gX2
        ytr_b, gY2 = _resid(T2t, ytr_a)
        p3, T3t, Q3, cf = _fit(x2t_b, ytr_b, c); ce = min(c, cf); T3e = p3.transform(x2e_b)[:,:ce]
        yh = (yh_ab + T3e @ Q3[:,:ce].T) * sdy + muY
        P[te] = yh
    sst = np.sum((Y - Y.mean(0))**2)
    return float(1 - ((P - Y)**2).sum() / sst)


def cv_q2_2block_at(Xa, Xb, Y, na, nb):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, 1))
    for tr, te in cv.split(Y):
        muXa, sdXa, ka = _scale_fit(Xa[tr]); xat, xae = _scale_apply(Xa[tr],muXa,sdXa,ka), _scale_apply(Xa[te],muXa,sdXa,ka)
        muXb, sdXb, kb = _scale_fit(Xb[tr]); xbt0, xbe0 = _scale_apply(Xb[tr],muXb,sdXb,kb), _scale_apply(Xb[te],muXb,sdXb,kb)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr0 = (Y[tr]-muY)/sdy
        p1, Tat, Qa, af = _fit(xat, ytr0, na); ae = min(na, af); Tat, Tae = Tat[:,:ae], p1.transform(xae)[:,:ae]
        yh_a = Tae @ Qa[:,:ae].T
        xbt, gXb = _resid(Tat, xbt0); xbe = xbe0 - Tae @ gXb
        ytr_a, gY = _resid(Tat, ytr0)
        p2, Tbt, Qb, bf = _fit(xbt, ytr_a, nb); be = min(nb, bf); Tbe = p2.transform(xbe)[:,:be]
        yh = (yh_a + Tbe @ Qb[:,:be].T) * sdy + muY
        P[te] = yh
    sst = np.sum((Y-Y.mean(0))**2)
    return float(1 - ((P-Y)**2).sum()/sst)


def cv_q2_1block_at(X, Y, n):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    nn = len(Y); P = np.zeros((nn, 1))
    for tr, te in cv.split(Y):
        muX, sdX, k = _scale_fit(X[tr]); xt, xe = _scale_apply(X[tr],muX,sdX,k), _scale_apply(X[te],muX,sdX,k)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr = (Y[tr]-muY)/sdy
        p, Tt, Q, af = _fit(xt, ytr, n); ae = min(n, af); Te = p.transform(xe)[:,:ae]
        yh = (Te @ Q[:,:ae].T)*sdy+muY
        P[te] = yh
    sst = np.sum((Y-Y.mean(0))**2)
    return float(1 - ((P-Y)**2).sum()/sst)


# ---------------------------------------------------------------------------
# Run for all 7 models
# ---------------------------------------------------------------------------
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

    # PART 1: SO-PLS in-sample SS decomposition at the FULL model's LV
    ssx1, ssm, ssx, ssf, sst = full_fit_decomposition(X1_, M2_, X2flat_, Y_, a, b, c)
    m2m = ssm / ssx if abs(ssx) > 1e-9 else np.nan

    # PART 2: ablation, all at the SAME (a,b,c) -- no independent re-optimization
    q2_full = cv_q2_3block_at(X1_, M2_, X2flat_, Y_, a, b, c)
    q2_no_m2 = cv_q2_2block_at(X1_, X2flat_, Y_, a, c)   # TD = X1(a)+X2(c)
    q2_no_x2 = cv_q2_2block_at(X1_, M2_, Y_, a, b)        # KD = X1(a)+M2(b)
    q2_x1_only = cv_q2_1block_at(X1_, Y_, a)

    d_q2_m = q2_full - q2_no_m2
    d_q2_x = q2_full - q2_no_x2
    amr = d_q2_m / d_q2_x if abs(d_q2_x) > 1e-9 else np.nan

    # ablation overlap/complementarity: (KD-X1only)+(TD-X1only) vs (FULL-X1only)
    overlap = q2_no_x2 + q2_no_m2 - q2_x1_only - q2_full

    rows.append(dict(model=name, LV_X1=a, LV_M2=b, LV_X2=c,
                      SS_M=ssm, SS_X=ssx, SS_F=ssf, SS_Y=sst, M2M=m2m,
                      Q2_FULL=q2_full, Q2_NO_M2=q2_no_m2, Q2_NO_X2=q2_no_x2, Q2_X1_ONLY=q2_x1_only,
                      DeltaQ2_M=d_q2_m, DeltaQ2_X=d_q2_x, AMR=amr,
                      Ablation_overlap=overlap,
                      Ablation_interpretation=('redundant/shared' if overlap > 0 else 'complementary/non-additive')))
    print(f"  LV=({a},{b},{c})  M2M={m2m:.3f}  Q2_FULL={q2_full:.4f}  AMR={amr:.3f}  overlap={overlap:.4f}")

df = pd.DataFrame(rows)
print("\n===== COMPARISON TABLE =====")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------
m2m_rank = df[['model', 'M2M']].sort_values('M2M', ascending=False).reset_index(drop=True)
m2m_rank['rank'] = m2m_rank.index + 1
amr_rank = df[['model', 'AMR']].sort_values('AMR', ascending=False).reset_index(drop=True)
amr_rank['rank'] = amr_rank.index + 1
print("\nM2M ranking:\n", m2m_rank.to_string(index=False))
print("\nAMR ranking:\n", amr_rank.to_string(index=False))

# ---------------------------------------------------------------------------
# Save both requested workbooks
# ---------------------------------------------------------------------------
comparison_cols = ['model','LV_X1','LV_M2','LV_X2','SS_M','SS_X','SS_F','SS_Y','M2M',
                    'Q2_FULL','Q2_NO_M2','Q2_NO_X2','DeltaQ2_M','DeltaQ2_X','AMR']
with pd.ExcelWriter('M2M_vs_AMR_comparison.xlsx') as w:
    df[comparison_cols].to_excel(w, sheet_name='comparison_table', index=False)
    m2m_rank.to_excel(w, sheet_name='M2M_ranking', index=False)
    amr_rank.to_excel(w, sheet_name='AMR_ranking', index=False)
print("\nSaved M2M_vs_AMR_comparison.xlsx")

so_pls_shared = pd.DataFrame([{'model': m, 'shared_information': 'NOT DERIVABLE',
    'reason': ('Sequential orthogonalization (X1->M2->X2) forces M2 to absorb ALL variance it shares '
               'with X2 (M2 is fit first, before any residualization); X2 is then fit only on the '
               'orthogonal complement, so SS_X is already X2-unique-beyond-M2 by construction. There is '
               'no further split of SS_M into "M2-alone" vs "shared-with-X2" available from this single '
               'FULL-model fit -- doing so would require a second, reversed-order (X1->X2->M2) fit, which '
               'is outside the FULL model as specified and was not computed here.')}
    for m in MODELS])
ablation_shared = df[['model','Q2_FULL','Q2_NO_M2','Q2_NO_X2','Q2_X1_ONLY','Ablation_overlap','Ablation_interpretation']]

with pd.ExcelWriter('M2M_AMR_shared_comparison.xlsx') as w:
    df[comparison_cols + ['Q2_X1_ONLY','Ablation_overlap','Ablation_interpretation']].to_excel(w, sheet_name='comparison_table', index=False)
    so_pls_shared.to_excel(w, sheet_name='SO_PLS_shared', index=False)
    ablation_shared.to_excel(w, sheet_name='ablation_shared', index=False)
    pd.concat([m2m_rank.assign(metric='M2M'), amr_rank.rename(columns={'AMR':'M2M'}).assign(metric='AMR')]).to_excel(w, sheet_name='rankings', index=False)
print("Saved M2M_AMR_shared_comparison.xlsx")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order = df['model'].tolist()
axes[0].bar(order, df['M2M'], color='#3E7C8C')
axes[0].set_title('M2M = SS_M / SS_X'); axes[0].set_ylabel('M2M'); axes[0].tick_params(axis='x', rotation=30)
axes[1].bar(order, df['AMR'], color='#B5533C')
axes[1].axhline(1.0, color='black', linestyle=':', linewidth=1)
axes[1].set_title('AMR = DeltaQ2_M / DeltaQ2_X'); axes[1].set_ylabel('AMR'); axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('M2M_vs_AMR_comparison.png', dpi=140, facecolor='white')
print("Saved M2M_vs_AMR_comparison.png")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(order)); w = 0.2
ax.bar(x - 1.5*w, df['SS_M'], width=w, label='SS_M (mechanistic)', color='#3E7C8C')
ax.bar(x - 0.5*w, df['SS_X'], width=w, label='SS_X (measured, unique)', color='#D97E3A')
ax2 = ax.twinx()
ax2.bar(x + 0.5*w, df['DeltaQ2_M'], width=w, label='DeltaQ2_M (ablation)', color='#16323F')
ax2.bar(x + 1.5*w, df['DeltaQ2_X'], width=w, label='DeltaQ2_X (ablation)', color='#B5533C')
ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha='right')
ax.set_ylabel('SS (SO-PLS, in-sample)'); ax2.set_ylabel('DeltaQ2 (ablation, CV)')
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right')
ax.set_title('Mechanistic vs measured contribution: SO-PLS (SS) vs ablation (DeltaQ2)')
plt.tight_layout()
plt.savefig('M2M_AMR_shared_contributions.png', dpi=140, facecolor='white')
print("Saved M2M_AMR_shared_contributions.png")
