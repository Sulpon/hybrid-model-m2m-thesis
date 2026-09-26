"""
Extrapolation performance of EVERY allocation inside the 1-SE flat region.

All members of a 1-SE region are, by construction, statistically
indistinguishable on calibration. This asks what they do OUTSIDE the
calibration domain, and reports the full distribution rather than the single
value at the selected allocation.

Also tested, inside the region only: does spending latent variables on M2
rather than X2 improve extrapolation? (Spearman of each block's LV count
against RMSEP_extrap, within the region.)

Same regime caveats as extrapolation_test.py -- DG_* generated data, M2 blocks
from gen_all.py with FIXED NOMINAL kinetic parameters, and the "within-domain"
set actually spans only T2 362-370 (the top 20% of the calibration range), so
it is reported for completeness but not used for any conclusion.

Benchmark: the data-driven X1+X2 model contains no M2 block and is therefore
identical for all seven models; its extrapolation RMSEP is a fixed reference.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, time
warnings.filterwarnings('ignore')

MAX_X1, MAX_M2, MAX_X2 = 6, 15, 15
N_SPLITS, SEED = 10, 42
MODELS = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
          'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

def load(fn, sheet): return pd.read_excel(fn, sheet_name=sheet).values.astype(float)
X1c = load('DG_calibration.xlsx', 'X1'); X2c = load('DG_calibration.xlsx', 'X2')
Yc  = load('DG_calibration.xlsx', 'Y').reshape(-1, 1)
X1w = load('DG_test_within_domain_large.xlsx', 'X1'); X2w = load('DG_test_within_domain_large.xlsx', 'X2')
Yw  = load('DG_test_within_domain_large.xlsx', 'Y').reshape(-1, 1)
X1e = load('DG_test_extrapolation_large.xlsx', 'X1'); X2e = load('DG_test_extrapolation_large.xlsx', 'X2')
Ye  = load('DG_test_extrapolation_large.xlsx', 'Y').reshape(-1, 1)

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n
def _scale_fit(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, np.sqrt(B.shape[1])
def _sa(B, mu, sd, k): return (B - mu)/sd/k
def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g

def cv_3(B1, B2, B3, Y, m1, m2, m3):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _scale_fit(B1[tr]); b1t, b1e = _sa(B1[tr],mu1,sd1,k1), _sa(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _scale_fit(B2[tr]); b2t0, b2e0 = _sa(B2[tr],mu2,sd2,k2), _sa(B2[te],mu2,sd2,k2)
        mu3, sd3, k3 = _scale_fit(B3[tr]); b3t0, b3e0 = _sa(B3[tr],mu3,sd3,k3), _sa(B3[te],mu3,sd3,k3)
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

def rmsep_surface_3(B1, B2, B3, Y, T1b, T2b, T3b, Yt, m1, m2, m3):
    mu1, sd1, k1 = _scale_fit(B1); b1t, b1e = _sa(B1,mu1,sd1,k1), _sa(T1b,mu1,sd1,k1)
    mu2, sd2, k2 = _scale_fit(B2); b2t0, b2e0 = _sa(B2,mu2,sd2,k2), _sa(T2b,mu2,sd2,k2)
    mu3, sd3, k3 = _scale_fit(B3); b3t0, b3e0 = _sa(B3,mu3,sd3,k3), _sa(T3b,mu3,sd3,k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Ytr0 = (Y-muY)/sdy
    out = np.zeros((m1, m2, m3))
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
                out[a-1, b-1, c-1] = np.sqrt(np.mean((yh - Yt.ravel())**2))
    return out

def rmsep_2(B1, B2, Y, T1b, T2b, Yt, n1, n2):
    mu1, sd1, k1 = _scale_fit(B1); b1t, b1e = _sa(B1,mu1,sd1,k1), _sa(T1b,mu1,sd1,k1)
    mu2, sd2, k2 = _scale_fit(B2); b2t0, b2e0 = _sa(B2,mu2,sd2,k2), _sa(T2b,mu2,sd2,k2)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Ytr0 = (Y-muY)/sdy
    p1, T1t, Q1, af = _fit(b1t, Ytr0, n1); T1e = p1.transform(b1e)
    ae = min(n1, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
    yh = Tae @ Q1[:,:ae].T
    b2t, g2 = _resid(Tat, b2t0); b2e = b2e0 - Tae @ g2
    Ytr_a, _ = _resid(Tat, Ytr0)
    p2, T2t, Q2, bf = _fit(b2t, Ytr_a, n2); T2e = p2.transform(b2e)
    be = min(n2, bf)
    pred = ((yh + T2e[:,:be] @ Q2[:,:be].T)*sdy + muY).ravel()
    return float(np.sqrt(np.mean((pred - Yt.ravel())**2)))

# data-driven benchmark (no M2 block -> identical for every model)
from itertools import product
q_dd_best, dd_lv = -9, None
cvk = KFold(N_SPLITS, shuffle=True, random_state=SEED)
def cv_q2_2(B1, B2, Y, m1, m2):
    P = np.zeros((len(Y), m1, m2))
    for f, (tr, te) in enumerate(cvk.split(Y)):
        mu1, sd1, k1 = _scale_fit(B1[tr]); b1t, b1e = _sa(B1[tr],mu1,sd1,k1), _sa(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _scale_fit(B2[tr]); b2t0, b2e0 = _sa(B2[tr],mu2,sd2,k2), _sa(B2[te],mu2,sd2,k2)
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
qDD = cv_q2_2(X1c, X2c, Yc, MAX_X1, MAX_X2)
iDD = np.unravel_index(qDD.argmax(), qDD.shape)
DD_BENCH = rmsep_2(X1c, X2c, Yc, X1e, X2e, Ye, iDD[0]+1, iDD[1]+1)
print(f"data-driven X1+X2 benchmark: LV=({iDD[0]+1},{iDD[1]+1})  RMSEP_extrap = {DD_BENCH:.4f}\n", flush=True)

rows, longrows, store = [], [], {}
t0 = time.time()
for name in MODELS:
    M2c = load(f'M2_{name}.xlsx', 'calibration')
    M2w = load(f'M2_{name}_large.xlsx', 'within_large')
    M2e = load(f'M2_{name}_large.xlsx', 'extrap_large')

    qF, rf = cv_3(X1c, M2c, X2c, Yc, MAX_X1, MAX_M2, MAX_X2)
    rmseE = rmsep_surface_3(X1c, M2c, X2c, Yc, X1e, M2e, X2e, Ye, MAX_X1, MAX_M2, MAX_X2)
    rmseW = rmsep_surface_3(X1c, M2c, X2c, Yc, X1w, M2w, X2w, Yw, MAX_X1, MAX_M2, MAX_X2)

    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin]); se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    mask = rmse <= rmin + se
    idx = np.argwhere(mask); tot = idx.sum(1)+3
    cand = idx[tot == tot.min()]; pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])

    v = rmseE[mask]; vw = rmseW[mask]
    lvs = np.argwhere(mask) + 1
    store[name] = dict(v=v, lvs=lvs, opt=float(rmseE[imin]), sel=float(rmseE[pick]),
                       opt_lv=tuple(np.array(imin)+1), sel_lv=tuple(np.array(pick)+1))
    rows.append(dict(model=name, n_1SE=int(mask.sum()),
        RMSECV_min=rmin, SE=se,
        argmin_LV=str(tuple(np.array(imin)+1)), RMSEP_at_argmin=float(rmseE[imin]),
        oneSE_LV=str(tuple(np.array(pick)+1)), RMSEP_at_1SE_pick=float(rmseE[pick]),
        RMSEP_min=float(v.min()), RMSEP_p10=float(np.percentile(v,10)),
        RMSEP_p25=float(np.percentile(v,25)), RMSEP_median=float(np.median(v)),
        RMSEP_p75=float(np.percentile(v,75)), RMSEP_p90=float(np.percentile(v,90)),
        RMSEP_max=float(v.max()), RMSEP_mean=float(v.mean()), RMSEP_sd=float(v.std(ddof=1)),
        RMSEP_IQR=float(np.percentile(v,75)-np.percentile(v,25)),
        RMSEP_range=float(v.max()-v.min()),
        frac_beat_DD=float((v < DD_BENCH).mean()),
        best_LV=str(tuple(np.argwhere(mask)[v.argmin()]+1)),
        worst_LV=str(tuple(np.argwhere(mask)[v.argmax()]+1)),
        rho_LVM2=spearmanr(lvs[:,1], v).correlation,
        rho_LVX2=spearmanr(lvs[:,2], v).correlation,
        rho_LVX1=spearmanr(lvs[:,0], v).correlation,
        RMSEP_within_median=float(np.median(vw))))
    for (a, b, c), e_, w_ in zip(lvs, v, vw):
        longrows.append((name, int(a), int(b), int(c), float(rmse[a-1,b-1,c-1]),
                         float(qF[a-1,b-1,c-1]), float(e_), float(w_)))
    r = rows[-1]
    print(f"{name:14s} n={r['n_1SE']:4d}  RMSEP_extrap  min={r['RMSEP_min']:.3f} "
          f"p25={r['RMSEP_p25']:.3f} med={r['RMSEP_median']:.3f} p75={r['RMSEP_p75']:.3f} "
          f"max={r['RMSEP_max']:.3f}  range={r['RMSEP_range']:.3f}  "
          f"beat_DD={r['frac_beat_DD']:.2f}  rho(LV_M2)={r['rho_LVM2']:+.2f} rho(LV_X2)={r['rho_LVX2']:+.2f}",
          flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 330, 'display.max_columns', 80)
print("\n=== distribution of extrapolation RMSEP across the 1-SE region ===")
print(df[['model','n_1SE','RMSEP_min','RMSEP_p10','RMSEP_p25','RMSEP_median','RMSEP_p75',
          'RMSEP_p90','RMSEP_max','RMSEP_sd','RMSEP_range']].round(4).to_string(index=False))
print("\n=== where the two conventional picks land inside that distribution ===")
print(df[['model','argmin_LV','RMSEP_at_argmin','oneSE_LV','RMSEP_at_1SE_pick','RMSEP_median',
          'best_LV','RMSEP_min','worst_LV','RMSEP_max','frac_beat_DD']].round(4).to_string(index=False))
print(f"\n(data-driven benchmark RMSEP_extrap = {DD_BENCH:.4f})")
print("\n=== inside the region, does spending LVs on M2 rather than X2 help extrapolation? ===")
print("    (Spearman of block LV count vs RMSEP_extrap; negative = more LVs improves extrapolation)")
print(df[['model','rho_LVX1','rho_LVM2','rho_LVX2']].round(3).to_string(index=False))

lg = pd.DataFrame(longrows, columns=['model','LV_X1','LV_M2','LV_X2','RMSECV','Q2_cal',
                                     'RMSEP_extrap','RMSEP_within'])
with pd.ExcelWriter('extrapolation_1se_distribution.xlsx') as w:
    df.to_excel(w, sheet_name='summary', index=False)
    lg.to_excel(w, sheet_name='all_1SE_allocations', index=False)

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.25)
lo = min(s['v'].min() for s in store.values()); hi = max(s['v'].max() for s in store.values())
bins = np.linspace(lo, hi, 40)
for i, nm in enumerate(MODELS):
    ax = fig.add_subplot(gs[i//3, i % 3]); s = store[nm]
    ax.hist(s['v'], bins=bins, color='#4C72B0', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(DD_BENCH, color='k', lw=1.8, label='data-driven')
    ax.axvline(s['opt'], color='darkgreen', lw=1.6, ls='--', label='argmin pick')
    ax.axvline(s['sel'], color='#C44E52', lw=1.6, ls=':', label='1-SE pick')
    ax.set_title(f"{nm}  (n={len(s['v'])})", fontsize=9)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=6.5)
    if i//3 == 2 or i >= 4:
        ax.set_xlabel('RMSEP extrapolation', fontsize=8)
ax = fig.add_subplot(gs[2, 1:])
ax.boxplot([store[m]['v'] for m in MODELS], labels=[m.split('_')[0] for m in MODELS],
           showfliers=False, medianprops=dict(color='#C44E52', lw=2))
ax.axhline(DD_BENCH, color='k', lw=1.6, ls='-')
ax.text(0.02, DD_BENCH, ' data-driven benchmark', transform=ax.get_yaxis_transform(),
        fontsize=8, va='bottom')
ax.set_ylabel('RMSEP extrapolation', fontsize=9); ax.tick_params(labelsize=8)
ax.set_title('all calibration-equivalent (1-SE) allocations, by model', fontsize=9)
fig.suptitle('Extrapolation performance of every allocation inside the 1-SE flat region', fontsize=12)
fig.savefig('extrapolation_1se_distribution.png', dpi=160, bbox_inches='tight')
print(f"\nSaved extrapolation_1se_distribution.xlsx and .png   ({time.time()-t0:.0f}s)")
