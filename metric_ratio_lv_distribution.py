"""
Distribution of the proposed unique-ratio metric over ALL LV allocations.

For every allocation (a,b,c) on the grid X1<=6, M2<=15, X2<=15 both estimators
are evaluated with the sub-models held CONSISTENT with that allocation, so the
ratio is a function of (a,b,c) alone:

  SS version   unique_X(a,b,c) = SS of X2 entering last  in X1(a)->M2(b)->X2(c)
               unique_M(a,b,c) = SS of M2 entering last  in X1(a)->X2(c)->M2(b)
               ratio_SS = unique_M / unique_X
               (same per-block LV counts on both sides -- the "SS-match" variant)

  Q2 version   unique_M(a,b,c) = Q2_full(a,b,c) - Q2_DD(a,c)
               unique_X(a,b,c) = Q2_full(a,b,c) - Q2_KD(a,b)
               ratio_Q2 = unique_M / unique_X

The Q2 denominators can be <= 0 (adding X2 does not help out-of-sample at that
allocation); those cells are counted and reported, never silently dropped.

Conventions unchanged: mean-centre + unit-variance, block scaling 1/sqrt(K),
KFold(10, shuffle, seed=42), Q2 = 1 - pooled PRESS/SST, SS on standardised Y.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def ss_surface_3(B1, B2, B3, Y, m1, m2, m3):
    """In-sample SS of the THIRD block over the whole (a,b,c) grid."""
    mu, sd, k = _scale_fit(B1); b1 = _scale_apply(B1, mu, sd, k)
    mu, sd, k = _scale_fit(B2); b2_0 = _scale_apply(B2, mu, sd, k)
    mu, sd, k = _scale_fit(B3); b3_0 = _scale_apply(B3, mu, sd, k)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    out = np.zeros((m1, m2, m3))
    p1, T1, Q1, af = _fit(b1, ys, m1)
    for a in range(1, m1+1):
        ae = min(a, af); Ta = T1[:, :ae]
        b2_a, _ = _resid(Ta, b2_0); b3_a, _ = _resid(Ta, b3_0); y_a, _ = _resid(Ta, ys)
        p2, T2v, Q2, bf = _fit(b2_a, y_a, m2)
        for b in range(1, m2+1):
            be = min(b, bf); Tb = T2v[:, :be]
            b3_b, _ = _resid(Tb, b3_a); y_b, _ = _resid(Tb, y_a)
            p3, T3, Q3, cf = _fit(b3_b, y_b, m3)
            for c in range(1, m3+1):
                ce = min(c, cf)
                out[a-1, b-1, c-1] = float(np.sum((T3[:, :ce] @ Q3[:, :ce].T)**2))
    return out

def cv_q2_3(B1, B2, B3, Y, m1, m2, m3):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3))
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
                    P[te, a-1, b-1, c-1] = ((yh_ab + T3e[:,:ce] @ Q3[:,:ce].T)*sdy + muY).ravel()
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None,None])**2).sum(0)/sst

def cv_q2_2(B1, B2, Y, m1, m2):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2))
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
                P[te, a-1, b-1] = ((yh_a + T2e[:,:be] @ Q2[:,:be].T)*sdy + muY).ravel()
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None])**2).sum(0)/sst

opt = pd.read_excel('sopls_two_orderings.xlsx').set_index('model')
rows, store, longrows = [], {}, []
t0 = time.time()
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1m, M2m_meta, Ym, n_pts_m, IC_m, X2flat_m
    else:
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1a, M2a_meta, Ya, n_pts_a, IC_a, X2flat_a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    # ---- SS surfaces ----
    ssX = ss_surface_3(X1_, M2_, X2_, Y_, MAX_X1, MAX_M2, MAX_X2)      # [a,b,c] -> unique_X
    ssM = ss_surface_3(X1_, X2_, M2_, Y_, MAX_X1, MAX_X2, MAX_M2)      # [a,c,b] -> unique_M
    uM_ss = np.transpose(ssM, (0, 2, 1))                               # -> [a,b,c]
    ratio_ss = uM_ss/ssX

    # ---- Q2 surfaces ----
    qF = cv_q2_3(X1_, M2_, X2_, Y_, MAX_X1, MAX_M2, MAX_X2)            # [a,b,c]
    qKD = cv_q2_2(X1_, M2_, Y_, MAX_X1, MAX_M2)                        # [a,b]
    qDD = cv_q2_2(X1_, X2_, Y_, MAX_X1, MAX_X2)                        # [a,c]
    uM_q = qF - qDD[:, None, :]
    uX_q = qF - qKD[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_q = np.where(uX_q > 1e-9, uM_q/uX_q, np.nan)

    fr = opt.loc[name]
    a, b, c = int(fr.fwd_LV_X1), int(fr.fwd_LV_M2), int(fr.fwd_LV_X2)
    store[name] = dict(ratio_ss=ratio_ss, ratio_q=ratio_q, opt=(a, b, c),
                       n_bad=int((uX_q <= 1e-9).sum()))

    rs, rq = ratio_ss.ravel(), ratio_q.ravel()
    rq_ok = rq[np.isfinite(rq)]
    rows.append(dict(model=name, opt_LV=f"({a},{b},{c})", n_cells=rs.size,
        ss_at_opt=ratio_ss[a-1, b-1, c-1], ss_median=np.median(rs),
        ss_p10=np.percentile(rs, 10), ss_p90=np.percentile(rs, 90),
        ss_min=rs.min(), ss_max=rs.max(), ss_frac_gt1=float((rs > 1).mean()),
        q_at_opt=ratio_q[a-1, b-1, c-1], q_median=np.median(rq_ok),
        q_p10=np.percentile(rq_ok, 10), q_p90=np.percentile(rq_ok, 90),
        q_min=rq_ok.min(), q_max=rq_ok.max(), q_frac_gt1=float((rq_ok > 1).mean()),
        q_n_undefined=int(np.isnan(rq).sum()),
        q_frac_defined=float(np.isfinite(rq).mean())))
    for ai in range(MAX_X1):
        for bi in range(MAX_M2):
            for ci in range(MAX_X2):
                longrows.append((name, ai+1, bi+1, ci+1, ratio_ss[ai,bi,ci], ratio_q[ai,bi,ci],
                                 uM_ss[ai,bi,ci], ssX[ai,bi,ci], uM_q[ai,bi,ci], uX_q[ai,bi,ci]))
    r = rows[-1]
    print(f"{name:14s} SS: med={r['ss_median']:7.3f} p10={r['ss_p10']:6.3f} p90={r['ss_p90']:8.3f} "
          f"P(>1)={r['ss_frac_gt1']:.3f} opt={r['ss_at_opt']:6.3f} | "
          f"Q2: med={r['q_median']:6.3f} p10={r['q_p10']:6.3f} p90={r['q_p90']:8.3f} "
          f"P(>1)={r['q_frac_gt1']:.3f} opt={r['q_at_opt']:6.3f} undef={r['q_n_undefined']:4d}", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 300, 'display.max_columns', 60)
print("\n=== SS version over all 1350 allocations ===")
print(df[['model','opt_LV','ss_at_opt','ss_median','ss_p10','ss_p90','ss_min','ss_max','ss_frac_gt1']].round(4).to_string(index=False))
print("\n=== Q2 version over all 1350 allocations ===")
print(df[['model','opt_LV','q_at_opt','q_median','q_p10','q_p90','q_min','q_max','q_frac_gt1','q_n_undefined']].round(4).to_string(index=False))

lg = pd.DataFrame(longrows, columns=['model','LV_X1','LV_M2','LV_X2','ratio_SS','ratio_Q2',
                                     'uM_SS','uX_SS','uM_Q2','uX_Q2'])
with pd.ExcelWriter('metric_ratio_lv_distribution.xlsx') as w:
    df.to_excel(w, sheet_name='summary', index=False)
    lg.to_excel(w, sheet_name='all_allocations', index=False)

# ------------------------------------------------------------------ figure
names = list(MODELS)
fig, axes = plt.subplots(len(names), 2, figsize=(11, 2.05*len(names)), sharex='col')
for i, nm in enumerate(names):
    d = store[nm]
    for j, (key, lab, col) in enumerate([('ratio_ss', 'SS version', '#4C72B0'),
                                         ('ratio_q',  'Q$^2$ version', '#C44E52')]):
        ax = axes[i, j]
        v = d[key].ravel(); v = v[np.isfinite(v) & (v > 0)]
        bins = np.logspace(np.log10(max(v.min(), 1e-3)), np.log10(v.max()), 45)
        ax.hist(v, bins=bins, color=col, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.set_xscale('log')
        ax.axvline(1.0, color='k', lw=1.6, ls='-', zorder=5)
        a, b, c = d['opt']
        ax.axvline(d[key][a-1, b-1, c-1], color='darkgreen', lw=1.6, ls='--', zorder=6)
        frac = float((v > 1).mean())
        ax.text(0.015, 0.86, f"{nm}   P(>1)={frac:.2f}", transform=ax.transAxes,
                fontsize=8, va='top')
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(f"{lab}   (solid = reference 1, dashed = RMSECV-optimal LV)", fontsize=9)
        if i == len(names)-1:
            ax.set_xlabel('unique$_{M2}$ / unique$_{X2}$   (log scale)', fontsize=9)
fig.suptitle('Distribution of the unique-contribution ratio over all 1350 LV allocations '
             '(X1$\\leq$6, M2$\\leq$15, X2$\\leq$15)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig('metric_ratio_lv_distribution.png', dpi=160)
print(f"\nSaved metric_ratio_lv_distribution.xlsx and .png   ({time.time()-t0:.0f}s)")
