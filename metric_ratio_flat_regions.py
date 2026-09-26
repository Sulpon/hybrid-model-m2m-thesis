"""
Same unique-ratio distributions as metric_ratio_lv_distribution.py, but
restricted to the RMSECV flat regions instead of the whole grid.

Flat region at level k:   { (a,b,c) : RMSECV(a,b,c) <= RMSECV_min + k*SE }

SE is the across-fold standard error of the mean RMSE AT THE ARGMIN allocation
(the usual 1-SE-rule convention), SE = sd(fold RMSEs at argmin)/sqrt(10).
k = 1, 2, 3 are compared against the full grid.

Both estimators are as before, with sub-models held consistent with each
candidate allocation:
    ratio_SS(a,b,c) = SS_M2[X1(a)->X2(c)->M2(b)] / SS_X2[X1(a)->M2(b)->X2(c)]
    ratio_Q2(a,b,c) = (Q2_full(a,b,c) - Q2_DD(a,c)) / (Q2_full(a,b,c) - Q2_KD(a,b))

Conventions unchanged: KFold(10, shuffle, seed=42), block scaling 1/sqrt(K),
RMSECV = mean of per-fold RMSE, Q2 = 1 - pooled PRESS/SST, SS on standardised Y.
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
KS = [1, 2, 3]

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

def cv_3(B1, B2, B3, Y, m1, m2, m3):
    """Returns Q2 surface and the per-fold RMSE cube."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
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
                    yh = ((yh_ab + T3e[:,:ce] @ Q3[:,:ce].T)*sdy + muY).ravel()
                    P[te, a-1, b-1, c-1] = yh
                    rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None,None])**2).sum(0)/sst, rf

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

def describe(v, label, n_total, n_undef=0):
    ok = v[np.isfinite(v)]
    return dict(region=label, n_cells=int(n_total), n_defined=int(ok.size), n_undefined=int(n_undef),
                median=float(np.median(ok)) if ok.size else np.nan,
                p10=float(np.percentile(ok, 10)) if ok.size else np.nan,
                p90=float(np.percentile(ok, 90)) if ok.size else np.nan,
                vmin=float(ok.min()) if ok.size else np.nan,
                vmax=float(ok.max()) if ok.size else np.nan,
                frac_gt1=float((ok > 1).mean()) if ok.size else np.nan)

rows, store = [], {}
t0 = time.time()
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1m, M2m_meta, Ym, n_pts_m, IC_m, X2flat_m
    else:
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1a, M2a_meta, Ya, n_pts_a, IC_a, X2flat_a
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    ssX = ss_surface_3(X1_, M2_, X2_, Y_, MAX_X1, MAX_M2, MAX_X2)
    ssM = ss_surface_3(X1_, X2_, M2_, Y_, MAX_X1, MAX_X2, MAX_M2)
    ratio_ss = np.transpose(ssM, (0, 2, 1))/ssX

    qF, rf = cv_3(X1_, M2_, X2_, Y_, MAX_X1, MAX_M2, MAX_X2)
    qKD = cv_q2_2(X1_, M2_, Y_, MAX_X1, MAX_M2)
    qDD = cv_q2_2(X1_, X2_, Y_, MAX_X1, MAX_X2)
    uM_q = qF - qDD[:, None, :]; uX_q = qF - qKD[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_q = np.where(uX_q > 1e-9, uM_q/uX_q, np.nan)

    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = rmse[imin]
    se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    a, b, c = imin[0]+1, imin[1]+1, imin[2]+1

    masks = {'full grid': np.ones_like(rmse, dtype=bool)}
    for k in KS:
        masks[f'{k}-SE'] = rmse <= rmin + k*se
    store[name] = dict(ratio_ss=ratio_ss, ratio_q=ratio_q, masks=masks, opt=(a, b, c),
                       rmin=rmin, se=se)

    for lab, m in masks.items():
        vs = ratio_ss[m]
        vq_all = ratio_q[m]
        n_und = int(np.isnan(vq_all).sum())
        d_ss = describe(vs, lab, m.sum())
        d_q  = describe(vq_all, lab, m.sum(), n_und)
        rows.append(dict(model=name, region=lab, n_cells=int(m.sum()),
            RMSECV_min=rmin, SE=se, opt_LV=f"({a},{b},{c})",
            ss_median=d_ss['median'], ss_p10=d_ss['p10'], ss_p90=d_ss['p90'],
            ss_min=d_ss['vmin'], ss_max=d_ss['vmax'], ss_frac_gt1=d_ss['frac_gt1'],
            ss_at_opt=ratio_ss[a-1, b-1, c-1],
            q_median=d_q['median'], q_p10=d_q['p10'], q_p90=d_q['p90'],
            q_min=d_q['vmin'], q_max=d_q['vmax'], q_frac_gt1=d_q['frac_gt1'],
            q_n_undefined=n_und, q_frac_undefined=n_und/m.sum(),
            q_at_opt=ratio_q[a-1, b-1, c-1]))
    sizes = {lab: int(m.sum()) for lab, m in masks.items()}
    print(f"{name:14s} opt=({a},{b},{c}) RMSECV={rmin:.4f} SE={se:.4f}  sizes={sizes}", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 320, 'display.max_columns', 60)
for reg in ['full grid'] + [f'{k}-SE' for k in KS]:
    s = df[df.region == reg]
    print(f"\n=== {reg} ===")
    print(s[['model','n_cells','ss_at_opt','ss_median','ss_p10','ss_p90','ss_frac_gt1',
             'q_at_opt','q_median','q_p10','q_p90','q_frac_gt1','q_frac_undefined']].round(4).to_string(index=False))

print("\n=== P(ratio>1) by region ===")
for est, col in [('SS', 'ss_frac_gt1'), ('Q2', 'q_frac_gt1')]:
    piv = df.pivot(index='model', columns='region', values=col)
    piv = piv[['1-SE', '2-SE', '3-SE', 'full grid']]
    print(f"\n{est}:"); print(piv.round(3).to_string())
print("\n=== median ratio by region ===")
for est, col in [('SS', 'ss_median'), ('Q2', 'q_median')]:
    piv = df.pivot(index='model', columns='region', values=col)[['1-SE','2-SE','3-SE','full grid']]
    print(f"\n{est}:"); print(piv.round(3).to_string())

df.to_excel('metric_ratio_flat_regions.xlsx', index=False)

# ------------------------------------------------------------------ figure
names = list(MODELS)
cols = {'full grid': ('0.75', 1.0), '3-SE': ('#55A868', 1.4), '2-SE': ('#DD8452', 1.6), '1-SE': ('#C44E52', 2.0)}
fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.1*len(names)), sharex='col')
for i, nm in enumerate(names):
    d = store[nm]
    for j, key in enumerate(['ratio_ss', 'ratio_q']):
        ax = axes[i, j]
        allv = d[key][np.isfinite(d[key]) & (d[key] > 0)]
        bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 45)
        for lab, (col, lw) in cols.items():
            v = d[key][d['masks'][lab]]
            v = v[np.isfinite(v) & (v > 0)]
            if v.size == 0:
                continue
            if lab == 'full grid':
                ax.hist(v, bins=bins, density=True, color=col, alpha=0.55,
                        edgecolor='none', label=f"full ({v.size})")
            else:
                ax.hist(v, bins=bins, density=True, histtype='step', color=col,
                        lw=lw, label=f"{lab} ({v.size})")
        ax.set_xscale('log')
        ax.axvline(1.0, color='k', lw=1.5, zorder=6)
        a, b, c = d['opt']
        ax.axvline(d[key][a-1, b-1, c-1], color='darkgreen', lw=1.4, ls='--', zorder=7)
        ax.text(0.015, 0.92, nm, transform=ax.transAxes, fontsize=8, va='top')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.85)
        ax.tick_params(labelsize=7); ax.set_yticks([])
        if i == 0:
            ax.set_title(('SS version' if j == 0 else 'Q$^2$ version') +
                         '   (black = reference 1, dashed = RMSECV optimum)', fontsize=9)
        if i == len(names)-1:
            ax.set_xlabel('unique$_{M2}$ / unique$_{X2}$   (log scale)', fontsize=9)
fig.suptitle('Unique-contribution ratio within RMSECV flat regions (1-, 2-, 3-SE) vs the full grid',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig('metric_ratio_flat_regions.png', dpi=160)
print(f"\nSaved metric_ratio_flat_regions.xlsx and .png   ({time.time()-t0:.0f}s)")
