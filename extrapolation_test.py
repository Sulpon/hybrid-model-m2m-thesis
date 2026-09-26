"""
EXTRAPOLATION TEST of the calibration-only unique-ratio metric.

Question: does a metric computed ONLY on calibration data predict which
mechanistic model degrades least outside the calibration domain?

REGIME WARNING -- this is NOT the same data regime as the metric development.
All earlier metric work (metric_unique_m2m.py, metric_ratio_*.py) used the
author's real X1/X2/Y.xlsx with per-model least_squares-FITTED kinetic
parameters. No extrapolation set exists for that data. The only extrapolation
sets that exist are the DG_* generated family, whose M2 blocks were built by
gen_all.py with FIXED NOMINAL parameters (A3=10, A4=20, Ea3=50000, Ea4=55000).
So the absolute metric values here are NOT comparable to the earlier tables;
only the internal comparison (metric vs extrapolation, within this regime) is.

Data:
  calibration   DG_calibration.xlsx          T2 ~ U(330,370), n=100
  within-domain DG_test_within_domain_large  T2 <= 370,       n=100  (control)
  extrapolation DG_test_extrapolation_large  T2 in [370,382], n=100
  M2 blocks     M2_<model>.xlsx sheet 'calibration'
                M2_<model>_large.xlsx sheets 'within_large' / 'extrap_large'

Everything else follows the established conventions: mean-centre + unit
variance (fit on calibration only), block scaling 1/sqrt(K), KFold(10,
shuffle, seed=42) for the calibration CV, RMSECV = mean of per-fold RMSE,
Q2 = 1 - pooled PRESS/SST, SS on standardised Y, grid X1<=6, M2<=15, X2<=15.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
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

def cv_q2_2(B1, B2, Y, m1, m2):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2))
    for f, (tr, te) in enumerate(cv.split(Y)):
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

def ss_surface_3(B1, B2, B3, Y, m1, m2, m3):
    mu, sd, k = _scale_fit(B1); b1 = _sa(B1, mu, sd, k)
    mu, sd, k = _scale_fit(B2); b2_0 = _sa(B2, mu, sd, k)
    mu, sd, k = _scale_fit(B3); b3_0 = _sa(B3, mu, sd, k)
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

def rmsep_surface_3(B1, B2, B3, Y, T1b, T2b, T3b, Yt, m1, m2, m3):
    """Fit on the FULL calibration set, predict the test set, over the whole grid."""
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

SST_w = float(((Yw - Yw.mean())**2).sum()); SST_e = float(((Ye - Ye.mean())**2).sum())
nW, nE = len(Yw), len(Ye)

rows, store = [], {}
t0 = time.time()
for name in MODELS:
    M2c = load(f'M2_{name}.xlsx', 'calibration')
    M2w = load(f'M2_{name}_large.xlsx', 'within_large')
    M2e = load(f'M2_{name}_large.xlsx', 'extrap_large')

    qF, rf = cv_3(X1c, M2c, X2c, Yc, MAX_X1, MAX_M2, MAX_X2)
    qKD = cv_q2_2(X1c, M2c, Yc, MAX_X1, MAX_M2)
    qDD = cv_q2_2(X1c, X2c, Yc, MAX_X1, MAX_X2)
    uM_q = qF - qDD[:, None, :]; uX_q = qF - qKD[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_q = np.where(uX_q > 1e-9, uM_q/uX_q, np.nan)
    ssX = ss_surface_3(X1c, M2c, X2c, Yc, MAX_X1, MAX_M2, MAX_X2)
    ssM = ss_surface_3(X1c, X2c, M2c, Yc, MAX_X1, MAX_X2, MAX_M2)
    ratio_ss = np.transpose(ssM, (0, 2, 1))/ssX

    rmseW = rmsep_surface_3(X1c, M2c, X2c, Yc, X1w, M2w, X2w, Yw, MAX_X1, MAX_M2, MAX_X2)
    rmseE = rmsep_surface_3(X1c, M2c, X2c, Yc, X1e, M2e, X2e, Ye, MAX_X1, MAX_M2, MAX_X2)

    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin]); se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    a0, b0, c0 = imin[0]+1, imin[1]+1, imin[2]+1
    mask = rmse <= rmin + se
    idx = np.argwhere(mask); tot = idx.sum(1)+3
    cand = idx[tot == tot.min()]; pick = cand[np.array([rmse[tuple(u)] for u in cand]).argmin()]
    a1, b1, c1 = pick[0]+1, pick[1]+1, pick[2]+1
    i1 = tuple(pick)

    # reference sub-models at their own calibration optimum
    iKD = np.unravel_index(qKD.argmax(), qKD.shape); iDD = np.unravel_index(qDD.argmax(), qDD.shape)
    kdE = rmsep_2(X1c, M2c, Yc, X1e, M2e, Ye, iKD[0]+1, iKD[1]+1)
    ddE = rmsep_2(X1c, X2c, Yc, X1e, X2e, Ye, iDD[0]+1, iDD[1]+1)
    kdW = rmsep_2(X1c, M2c, Yc, X1w, M2w, Yw, iKD[0]+1, iKD[1]+1)
    ddW = rmsep_2(X1c, X2c, Yc, X1w, X2w, Yw, iDD[0]+1, iDD[1]+1)

    rq_flat = ratio_q[mask]; rs_flat = ratio_ss[mask]
    store[name] = dict(ratio_q=ratio_q, ratio_ss=ratio_ss, rmseE=rmseE, rmseW=rmseW,
                       mask=mask, rmse=rmse, opt=imin, one=i1)
    rows.append(dict(model=name,
        argmin_LV=f"({a0},{b0},{c0})", RMSECV=rmin, SE=se, Q2_cal=float(qF[imin]),
        oneSE_LV=f"({a1},{b1},{c1})", n_1SE=int(mask.sum()),
        RMSECV_1SE=float(rmse[i1]), Q2_cal_1SE=float(qF[i1]),
        # --- calibration-only metric ---
        ratioQ_at_argmin=float(ratio_q[imin]), ratioQ_at_1SE=float(ratio_q[i1]),
        ratioQ_med_1SE=float(np.nanmedian(rq_flat)),
        ratioSS_at_argmin=float(ratio_ss[imin]), ratioSS_at_1SE=float(ratio_ss[i1]),
        ratioSS_med_1SE=float(np.median(rs_flat)),
        # --- extrapolation outcome ---
        RMSEP_within_argmin=float(rmseW[imin]), RMSEP_extrap_argmin=float(rmseE[imin]),
        RMSEP_within_1SE=float(rmseW[i1]), RMSEP_extrap_1SE=float(rmseE[i1]),
        R2p_within_1SE=1-nW*float(rmseW[i1])**2/SST_w, R2p_extrap_1SE=1-nE*float(rmseE[i1])**2/SST_e,
        degrade_1SE=float(rmseE[i1])-float(rmseW[i1]),
        degrade_argmin=float(rmseE[imin])-float(rmseW[imin]),
        RMSEP_extrap_med_1SE=float(np.median(rmseE[mask])),
        # --- reference sub-models ---
        KD_LV=f"({iKD[0]+1},{iKD[1]+1})", RMSEP_extrap_KD=kdE, RMSEP_within_KD=kdW,
        DD_LV=f"({iDD[0]+1},{iDD[1]+1})", RMSEP_extrap_DD=ddE, RMSEP_within_DD=ddW))
    r = rows[-1]
    print(f"{name:14s} argmin=({a0},{b0},{c0}) 1SE=({a1},{b1},{c1})  "
          f"ratioQ(1SE med)={r['ratioQ_med_1SE']:6.3f}  "
          f"RMSEP within={r['RMSEP_within_1SE']:.3f} extrap={r['RMSEP_extrap_1SE']:.3f} "
          f"degrade={r['degrade_1SE']:+.3f} | KD_ext={kdE:.3f} DD_ext={ddE:.3f}", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 330, 'display.max_columns', 80)
print("\n=== calibration metric vs extrapolation (1-SE selected model) ===")
print(df[['model','oneSE_LV','Q2_cal_1SE','ratioQ_med_1SE','ratioSS_med_1SE',
          'RMSEP_within_1SE','RMSEP_extrap_1SE','degrade_1SE','R2p_extrap_1SE']].round(4).to_string(index=False))
print("\n=== hybrid vs mechanistic-only vs data-driven, on the extrapolation set ===")
print(df[['model','RMSEP_extrap_1SE','KD_LV','RMSEP_extrap_KD','DD_LV','RMSEP_extrap_DD',
          'RMSEP_within_1SE','RMSEP_within_KD','RMSEP_within_DD']].round(4).to_string(index=False))

print("\n=== ACROSS-MODEL association: does the calibration metric rank extrapolation? ===")
for mcol in ['ratioQ_med_1SE','ratioQ_at_1SE','ratioQ_at_argmin','ratioSS_med_1SE','Q2_cal_1SE','RMSECV']:
    for ocol in ['RMSEP_extrap_1SE','degrade_1SE','RMSEP_extrap_med_1SE']:
        rho, p = spearmanr(df[mcol], df[ocol])
        print(f"  Spearman({mcol:18s}, {ocol:22s}) = {rho:+.3f}  (p={p:.3f})")

print("\n=== WITHIN-MODEL association across the 1-SE region ===")
wm = []
for name in MODELS:
    d = store[name]; m = d['mask']
    rq = d['ratio_q'][m]; rs = d['ratio_ss'][m]; re_ = d['rmseE'][m]; rw = d['rmseW'][m]
    ok = np.isfinite(rq)
    wm.append(dict(model=name, n=int(m.sum()),
        rho_ratioQ_vs_extrap=spearmanr(rq[ok], re_[ok]).correlation,
        rho_ratioSS_vs_extrap=spearmanr(rs, re_).correlation,
        rho_ratioQ_vs_degrade=spearmanr(rq[ok], (re_-rw)[ok]).correlation))
wmd = pd.DataFrame(wm)
print(wmd.round(3).to_string(index=False))

with pd.ExcelWriter('extrapolation_test.xlsx') as w:
    df.to_excel(w, sheet_name='summary', index=False)
    wmd.to_excel(w, sheet_name='within_model', index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].scatter(df.ratioQ_med_1SE, df.RMSEP_extrap_1SE, s=70, c='#C44E52')
for _, r in df.iterrows():
    ax[0].annotate(r.model.split('_')[0], (r.ratioQ_med_1SE, r.RMSEP_extrap_1SE),
                   textcoords='offset points', xytext=(6, 4), fontsize=9)
ax[0].axvline(1, color='k', lw=1.2)
ax[0].set_xlabel('calibration metric: median ratio$_{Q^2}$ in 1-SE region')
ax[0].set_ylabel('RMSEP extrapolation'); ax[0].set_title('metric vs extrapolation error')

w_ = np.arange(len(df)); bw = 0.27
ax[1].bar(w_-bw, df.RMSEP_extrap_1SE, bw, label='hybrid X1+M2+X2')
ax[1].bar(w_,     df.RMSEP_extrap_KD, bw, label='mechanistic X1+M2')
ax[1].bar(w_+bw,  df.RMSEP_extrap_DD, bw, label='data-driven X1+X2')
ax[1].set_xticks(w_); ax[1].set_xticklabels([m.split('_')[0] for m in df.model], fontsize=9)
ax[1].set_ylabel('RMSEP extrapolation'); ax[1].legend(fontsize=8); ax[1].set_title('extrapolation by block set')

ax[2].scatter(df.RMSEP_within_1SE, df.RMSEP_extrap_1SE, s=70, c='#4C72B0')
lim = [min(df.RMSEP_within_1SE.min(), df.RMSEP_extrap_1SE.min())*0.95,
       max(df.RMSEP_within_1SE.max(), df.RMSEP_extrap_1SE.max())*1.05]
ax[2].plot(lim, lim, 'k--', lw=1)
for _, r in df.iterrows():
    ax[2].annotate(r.model.split('_')[0], (r.RMSEP_within_1SE, r.RMSEP_extrap_1SE),
                   textcoords='offset points', xytext=(6, 4), fontsize=9)
ax[2].set_xlabel('RMSEP within-domain'); ax[2].set_ylabel('RMSEP extrapolation')
ax[2].set_title('degradation outside the domain')
fig.tight_layout(); fig.savefig('extrapolation_test.png', dpi=160)
print(f"\nSaved extrapolation_test.xlsx and .png   ({time.time()-t0:.0f}s)")
