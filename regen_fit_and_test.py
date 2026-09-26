"""
End-to-end rerun on the regenerated (corrected-sigma) data.

  1. fit [Ea3,Ea4,A3,A4] per candidate mechanism by least_squares against the
     CALIBRATION X2 trajectories only (same setup as
     ablation_AMR_all_models_real_data.py: p0=[50000,55000,10,20],
     bounds [1e3,1e3,1e-4,1e-4]-[1e6,1e6,1e4,1e4], max_nfev=60);
  2. build M2 blocks for calibration / within / extrapolation with those fitted
     parameters (fitted on calibration only -- no leakage into either test set);
  3. calibration-only quantities: RMSECV/Q2 surfaces, 1-SE region,
     ratio_SS and ratio_Q2 over the whole grid;
  4. extrapolation and within-domain RMSEP for every allocation;
  5. does the calibration metric predict extrapolation?

Unlike the DG_* regime this puts M6 on the SAME data as the other six, so all
seven are numerically comparable for the first time.

Conventions unchanged: KFold(10, shuffle, seed=42), block scaling 1/sqrt(K),
RMSECV = mean of per-fold RMSE, Q2 = 1 - pooled PRESS/SST, SS on standardised
Y, grid X1<=6, M2<=15, X2<=15, 1-SE = RMSECV <= min + SE(argmin).
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, time, json
warnings.filterwarnings('ignore')

R = 8.314
SPECIES = ['A', 'B', 'C', 'D', 'E', 'F']
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

def read(fn):
    x1 = pd.read_excel(fn, sheet_name='X1')[['CA0','T1','t1','CA1_final','CB1_final','CC1_final']].values
    x2 = pd.read_excel(fn, sheet_name='X2')
    y  = pd.read_excel(fn, sheet_name='Y')['E_purity'].values.reshape(-1, 1)
    return x1, x2, y
X1c, X2dc, Yc = read('RG_calibration.xlsx')
X1w, X2dw, Yw = read('RG_within.xlsx')
X1e, X2de, Ye = read('RG_extrap.xlsx')
NPTS = (X2dc.shape[1]-3)//6
TGRID = np.arange(NPTS)*15.0
print(f"n_cal={len(Yc)} n_wit={len(Yw)} n_ext={len(Ye)}  timepoints={NPTS}\n", flush=True)

def parts(X2df):
    n = len(X2df)
    IC = np.array([[X2df[f'{s}_t1'].iloc[i] for s in SPECIES] for i in range(n)])
    flat = np.array([[X2df[f'{s}_t{k+1}'].iloc[i] for s in SPECIES for k in range(NPTS)] for i in range(n)])
    obs = np.array([[[X2df[f'{s}_t{k+1}'].iloc[i] for s in SPECIES] for k in range(NPTS)] for i in range(n)])
    meta = X2df[['t2_final','T2','CD0']].values
    return IC, flat, obs, meta, X2df['T2'].values
ICc, X2c, OBSc, METAc, T2c = parts(X2dc)
ICw, X2w, OBSw, METAw, T2w = parts(X2dw)
ICe, X2e, OBSe, METAe, T2e = parts(X2de)

def fit_params(rhs):
    def resid(p):
        Ea3, Ea4, A3, A4 = p
        out = []
        for i in range(len(ICc)):
            sol = odeint(rhs, ICc[i], TGRID, args=(T2c[i], Ea3, Ea4, A3, A4))
            out.append((sol - OBSc[i]).ravel())
        return np.concatenate(out)
    r = least_squares(resid, [50000., 55000., 10., 20.],
                      bounds=([1e3, 1e3, 1e-4, 1e-4], [1e6, 1e6, 1e4, 1e4]),
                      xtol=1e-8, ftol=1e-8, max_nfev=60)
    return r.x, float(r.cost)

def build_M2(rhs, IC, T2v, meta, p):
    n = len(IC)
    M = np.zeros((n, 6, NPTS))
    for i in range(n):
        M[i] = odeint(rhs, IC[i], TGRID, args=(T2v[i], *p)).T
    return np.hstack([M.reshape(n, -1), meta])

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n
def _sf(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, np.sqrt(B.shape[1])
def _sa(B, mu, sd, k): return (B - mu)/sd/k
def _rs(T, tgt):
    g = np.linalg.pinv(T.T @ T) @ T.T @ tgt
    return tgt - T @ g, g

def cv_3(B1, B2, B3, Y, m1, m2, m3):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _sf(B1[tr]); b1t, b1e = _sa(B1[tr],mu1,sd1,k1), _sa(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _sf(B2[tr]); b2t0, b2e0 = _sa(B2[tr],mu2,sd2,k2), _sa(B2[te],mu2,sd2,k2)
        mu3, sd3, k3 = _sf(B3[tr]); b3t0, b3e0 = _sa(B3[tr],mu3,sd3,k3), _sa(B3[te],mu3,sd3,k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e_ = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e_[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            b2t, gM = _rs(Tat, b2t0); b2e = b2e0 - Tae @ gM
            b3ta, gX = _rs(Tat, b3t0); b3ea = b3e0 - Tae @ gX
            Ya, _ = _rs(Tat, Y0)
            p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e_ = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf); Tbt, Tbe = T2t[:,:be], T2e_[:,:be]
                yh_ab = yh_a + Tbe @ Q2[:,:be].T
                b3tb, gX2 = _rs(Tbt, b3ta); b3eb = b3ea - Tbe @ gX2
                Yb, _ = _rs(Tbt, Ya)
                p3, T3t, Q3, cf = _fit(b3tb, Yb, m3); T3e_ = p3.transform(b3eb)
                for c in range(1, m3+1):
                    ce = min(c, cf)
                    yh = ((yh_ab + T3e_[:,:ce] @ Q3[:,:ce].T)*sdy + muY).ravel()
                    P[te, a-1, b-1, c-1] = yh
                    rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None,None])**2).sum(0)/sst, rf

def cv_q2_2(B1, B2, Y, m1, m2):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _sf(B1[tr]); b1t, b1e = _sa(B1[tr],mu1,sd1,k1), _sa(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _sf(B2[tr]); b2t0, b2e0 = _sa(B2[tr],mu2,sd2,k2), _sa(B2[te],mu2,sd2,k2)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e_ = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e_[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            b2t, g2 = _rs(Tat, b2t0); b2e = b2e0 - Tae @ g2
            Ya, _ = _rs(Tat, Y0)
            p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e_ = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf)
                P[te, a-1, b-1] = ((yh_a + T2e_[:,:be] @ Q2[:,:be].T)*sdy + muY).ravel()
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None])**2).sum(0)/sst

def ss_surface(B1, B2, B3, Y, m1, m2, m3):
    mu, sd, k = _sf(B1); b1 = _sa(B1, mu, sd, k)
    mu, sd, k = _sf(B2); b20 = _sa(B2, mu, sd, k)
    mu, sd, k = _sf(B3); b30 = _sa(B3, mu, sd, k)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    out = np.zeros((m1, m2, m3))
    p1, T1, Q1, af = _fit(b1, ys, m1)
    for a in range(1, m1+1):
        ae = min(a, af); Ta = T1[:, :ae]
        b2a, _ = _rs(Ta, b20); b3a, _ = _rs(Ta, b30); ya, _ = _rs(Ta, ys)
        p2, T2v, Q2, bf = _fit(b2a, ya, m2)
        for b in range(1, m2+1):
            be = min(b, bf); Tb = T2v[:, :be]
            b3b, _ = _rs(Tb, b3a); yb, _ = _rs(Tb, ya)
            p3, T3, Q3, cf = _fit(b3b, yb, m3)
            for c in range(1, m3+1):
                ce = min(c, cf)
                out[a-1, b-1, c-1] = float(np.sum((T3[:, :ce] @ Q3[:, :ce].T)**2))
    return out

def rmsep_surface(B1, B2, B3, Y, S1, S2, S3, Yt, m1, m2, m3):
    mu1, sd1, k1 = _sf(B1); b1t, b1e = _sa(B1,mu1,sd1,k1), _sa(S1,mu1,sd1,k1)
    mu2, sd2, k2 = _sf(B2); b2t0, b2e0 = _sa(B2,mu2,sd2,k2), _sa(S2,mu2,sd2,k2)
    mu3, sd3, k3 = _sf(B3); b3t0, b3e0 = _sa(B3,mu3,sd3,k3), _sa(S3,mu3,sd3,k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Y0 = (Y-muY)/sdy
    out = np.zeros((m1, m2, m3))
    p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e_ = p1.transform(b1e)
    for a in range(1, m1+1):
        ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e_[:,:ae]
        yh_a = Tae @ Q1[:,:ae].T
        b2t, gM = _rs(Tat, b2t0); b2e = b2e0 - Tae @ gM
        b3ta, gX = _rs(Tat, b3t0); b3ea = b3e0 - Tae @ gX
        Ya, _ = _rs(Tat, Y0)
        p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e_ = p2.transform(b2e)
        for b in range(1, m2+1):
            be = min(b, bf); Tbt, Tbe = T2t[:,:be], T2e_[:,:be]
            yh_ab = yh_a + Tbe @ Q2[:,:be].T
            b3tb, gX2 = _rs(Tbt, b3ta); b3eb = b3ea - Tbe @ gX2
            Yb, _ = _rs(Tbt, Ya)
            p3, T3t, Q3, cf = _fit(b3tb, Yb, m3); T3e_ = p3.transform(b3eb)
            for c in range(1, m3+1):
                ce = min(c, cf)
                yh = ((yh_ab + T3e_[:,:ce] @ Q3[:,:ce].T)*sdy + muY).ravel()
                out[a-1, b-1, c-1] = np.sqrt(np.mean((yh - Yt.ravel())**2))
    return out

def rmsep_2(B1, B2, Y, S1, S2, Yt, n1, n2):
    mu1, sd1, k1 = _sf(B1); b1t, b1e = _sa(B1,mu1,sd1,k1), _sa(S1,mu1,sd1,k1)
    mu2, sd2, k2 = _sf(B2); b2t0, b2e0 = _sa(B2,mu2,sd2,k2), _sa(S2,mu2,sd2,k2)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Y0 = (Y-muY)/sdy
    p1, T1t, Q1, af = _fit(b1t, Y0, n1); T1e_ = p1.transform(b1e)
    ae = min(n1, af); Tat, Tae = T1t[:,:ae], T1e_[:,:ae]
    yh = Tae @ Q1[:,:ae].T
    b2t, g2 = _rs(Tat, b2t0); b2e = b2e0 - Tae @ g2
    Ya, _ = _rs(Tat, Y0)
    p2, T2t, Q2, bf = _fit(b2t, Ya, n2); T2e_ = p2.transform(b2e)
    be = min(n2, bf)
    pred = ((yh + T2e_[:,:be] @ Q2[:,:be].T)*sdy + muY).ravel()
    return float(np.sqrt(np.mean((pred - Yt.ravel())**2)))

# ---- data-driven benchmark: no M2 block, identical for every model ----
qDD = cv_q2_2(X1c, X2c, Yc, MAX_X1, MAX_X2)
iDD = np.unravel_index(qDD.argmax(), qDD.shape)
DD_E = rmsep_2(X1c, X2c, Yc, X1e, X2e, Ye, iDD[0]+1, iDD[1]+1)
DD_W = rmsep_2(X1c, X2c, Yc, X1w, X2w, Yw, iDD[0]+1, iDD[1]+1)
print(f"data-driven X1+X2 benchmark LV=({iDD[0]+1},{iDD[1]+1})  "
      f"RMSEP within={DD_W:.4f}  extrap={DD_E:.4f}  degradation={DD_E-DD_W:+.4f}\n", flush=True)

rows, longrows, store, fitted = [], [], {}, {}
t0 = time.time()
for name, rhs in MODELS.items():
    p, cost = fit_params(rhs)
    fitted[name] = [float(v) for v in p]
    M2c_ = build_M2(rhs, ICc, T2c, METAc, p)
    M2w_ = build_M2(rhs, ICw, T2w, METAw, p)
    M2e_ = build_M2(rhs, ICe, T2e, METAe, p)

    qF, rf = cv_3(X1c, M2c_, X2c, Yc, MAX_X1, MAX_M2, MAX_X2)
    qKD = cv_q2_2(X1c, M2c_, Yc, MAX_X1, MAX_M2)
    uM = qF - qDD[:, None, :]; uX = qF - qKD[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_q = np.where(uX > 1e-9, uM/uX, np.nan)
    ssX = ss_surface(X1c, M2c_, X2c, Yc, MAX_X1, MAX_M2, MAX_X2)
    ssM = ss_surface(X1c, X2c, M2c_, Yc, MAX_X1, MAX_X2, MAX_M2)
    ratio_ss = np.transpose(ssM, (0, 2, 1))/ssX

    rW = rmsep_surface(X1c, M2c_, X2c, Yc, X1w, M2w_, X2w, Yw, MAX_X1, MAX_M2, MAX_X2)
    rE = rmsep_surface(X1c, M2c_, X2c, Yc, X1e, M2e_, X2e, Ye, MAX_X1, MAX_M2, MAX_X2)

    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin]); se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    mask = rmse <= rmin + se
    idx = np.argwhere(mask); tot = idx.sum(1)+3
    cand = idx[tot == tot.min()]; pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])
    lvs = idx + 1
    vE, vW = rE[mask], rW[mask]
    rq, rs_ = ratio_q[mask], ratio_ss[mask]
    kdi = np.unravel_index(qKD.argmax(), qKD.shape)
    KD_E = rmsep_2(X1c, M2c_, Yc, X1e, M2e_, Ye, kdi[0]+1, kdi[1]+1)
    KD_W = rmsep_2(X1c, M2c_, Yc, X1w, M2w_, Yw, kdi[0]+1, kdi[1]+1)

    store[name] = dict(vE=vE, vW=vW, rq=rq, lvs=lvs, mask=mask,
                       opt=tuple(np.array(imin)+1), pick=tuple(np.array(pick)+1),
                       optE=float(rE[imin]), pickE=float(rE[pick]))
    rows.append(dict(model=name, Ea3=p[0], Ea4=p[1], A3=p[2], A4=p[3], fit_cost=cost,
        argmin_LV=str(tuple(int(v) for v in np.array(imin)+1)), RMSECV=rmin, SE=se, Q2_cal=float(qF[imin]),
        oneSE_LV=str(tuple(int(v) for v in np.array(pick)+1)), n_1SE=int(mask.sum()),
        Q2_cal_1SE=float(qF[pick]),
        ratioQ_argmin=float(ratio_q[imin]), ratioQ_pick=float(ratio_q[pick]),
        ratioQ_med=float(np.nanmedian(rq)), ratioQ_p10=float(np.nanpercentile(rq,10)),
        ratioQ_p90=float(np.nanpercentile(rq,90)), ratioQ_fracgt1=float(np.nanmean(rq>1)),
        ratioSS_med=float(np.median(rs_)), ratioSS_fracgt1=float((rs_>1).mean()),
        RMSEP_W_pick=float(rW[pick]), RMSEP_E_pick=float(rE[pick]),
        degrade_pick=float(rE[pick]-rW[pick]),
        RMSEP_W_med=float(np.median(vW)), RMSEP_E_med=float(np.median(vE)),
        degrade_med=float(np.median(vE-vW)),
        RMSEP_E_min=float(vE.min()), RMSEP_E_p25=float(np.percentile(vE,25)),
        RMSEP_E_p75=float(np.percentile(vE,75)), RMSEP_E_max=float(vE.max()),
        RMSEP_E_sd=float(vE.std(ddof=1)), frac_beat_DD=float((vE<DD_E).mean()),
        KD_LV=str((int(kdi[0]+1), int(kdi[1]+1))), KD_E=KD_E, KD_W=KD_W,
        rho_LVM2=spearmanr(lvs[:,1], vE).correlation,
        rho_LVX2=spearmanr(lvs[:,2], vE).correlation,
        rho_LVX1=spearmanr(lvs[:,0], vE).correlation,
        rho_ratioQ_vs_E=spearmanr(rq[np.isfinite(rq)], vE[np.isfinite(rq)]).correlation))
    for (a, b, c), e_, w_, q_, s_ in zip(lvs, vE, vW, rq, rs_):
        longrows.append((name, int(a), int(b), int(c), float(rmse[a-1,b-1,c-1]),
                         float(qF[a-1,b-1,c-1]), float(q_), float(s_), float(e_), float(w_)))
    r = rows[-1]
    print(f"{name:14s} p=[{p[0]:.4g},{p[1]:.4g},{p[2]:.4g},{p[3]:.4g}]  "
          f"argmin={r['argmin_LV']} 1SE={r['oneSE_LV']} n={r['n_1SE']:4d}  "
          f"ratioQ med={r['ratioQ_med']:6.3f} P(>1)={r['ratioQ_fracgt1']:.2f}  "
          f"RMSEP W={r['RMSEP_W_med']:.3f} E={r['RMSEP_E_med']:.3f} degr={r['degrade_med']:+.3f} "
          f"beatDD={r['frac_beat_DD']:.2f}", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 340, 'display.max_columns', 90)
print("\n=== fitted kinetic parameters (calibration only) ===")
print(df[['model','Ea3','Ea4','A3','A4','fit_cost']].round(3).to_string(index=False))
print("\n=== calibration-only metric (1-SE region) ===")
print(df[['model','argmin_LV','oneSE_LV','n_1SE','RMSECV','Q2_cal','ratioQ_med','ratioQ_p10',
          'ratioQ_p90','ratioQ_fracgt1','ratioSS_med','ratioSS_fracgt1']].round(4).to_string(index=False))
print("\n=== extrapolation outcome (1-SE region) ===")
print(df[['model','RMSEP_W_med','RMSEP_E_med','degrade_med','RMSEP_E_min','RMSEP_E_p25',
          'RMSEP_E_p75','RMSEP_E_max','RMSEP_E_sd','frac_beat_DD','KD_E']].round(4).to_string(index=False))
print(f"\n(data-driven benchmark: within {DD_W:.4f}, extrap {DD_E:.4f}, degradation {DD_E-DD_W:+.4f})")

print("\n=== ACROSS-MODEL: does the calibration metric predict extrapolation? ===")
for mc in ['ratioQ_med','ratioQ_pick','ratioSS_med','ratioQ_fracgt1','Q2_cal','RMSECV']:
    for oc in ['RMSEP_E_med','degrade_med','frac_beat_DD']:
        rho, pv = spearmanr(df[mc], df[oc])
        print(f"  Spearman({mc:15s}, {oc:15s}) = {rho:+.3f}  (p={pv:.3f})")
print("\n=== WITHIN-MODEL (across the 1-SE region) ===")
print(df[['model','rho_ratioQ_vs_E','rho_LVX1','rho_LVM2','rho_LVX2']].round(3).to_string(index=False))

lg = pd.DataFrame(longrows, columns=['model','LV_X1','LV_M2','LV_X2','RMSECV','Q2_cal',
                                     'ratio_Q2','ratio_SS','RMSEP_extrap','RMSEP_within'])
with pd.ExcelWriter('regen_fit_and_test.xlsx') as w:
    df.to_excel(w, sheet_name='summary', index=False)
    lg.to_excel(w, sheet_name='all_1SE_allocations', index=False)
json.dump(fitted, open('regen_fitted_params.json', 'w'), indent=2)

names = list(MODELS)
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax[0].scatter(df.ratioQ_med, df.RMSEP_E_med, s=80, c='#C44E52')
for _, r in df.iterrows():
    ax[0].annotate(r.model.split('_')[0], (r.ratioQ_med, r.RMSEP_E_med),
                   textcoords='offset points', xytext=(6, 4), fontsize=9)
ax[0].axvline(1, color='k', lw=1.3); ax[0].axhline(DD_E, color='0.5', lw=1.2, ls='--')
ax[0].set_xlabel('calibration metric: median ratio$_{Q^2}$ (1-SE region)')
ax[0].set_ylabel('median RMSEP extrapolation'); ax[0].set_title('metric vs extrapolation')
ax[1].boxplot([store[m]['vE'] for m in names], labels=[m.split('_')[0] for m in names],
              showfliers=False, medianprops=dict(color='#C44E52', lw=2))
ax[1].axhline(DD_E, color='k', lw=1.5)
ax[1].text(0.02, DD_E, ' data-driven', transform=ax[1].get_yaxis_transform(), fontsize=8, va='bottom')
ax[1].set_ylabel('RMSEP extrapolation'); ax[1].set_title('1-SE allocations, extrapolation')
ax[2].scatter(df.RMSEP_W_med, df.RMSEP_E_med, s=80, c='#4C72B0')
lim = [min(df.RMSEP_W_med.min(), df.RMSEP_E_med.min())*0.9, max(df.RMSEP_W_med.max(), df.RMSEP_E_med.max())*1.1]
ax[2].plot(lim, lim, 'k--', lw=1)
for _, r in df.iterrows():
    ax[2].annotate(r.model.split('_')[0], (r.RMSEP_W_med, r.RMSEP_E_med),
                   textcoords='offset points', xytext=(6, 4), fontsize=9)
ax[2].set_xlabel('median RMSEP within-domain'); ax[2].set_ylabel('median RMSEP extrapolation')
ax[2].set_title('degradation (proper in-domain control)')
fig.tight_layout(); fig.savefig('regen_fit_and_test.png', dpi=160)
print(f"\nSaved regen_fit_and_test.xlsx / .png / regen_fitted_params.json   ({time.time()-t0:.0f}s)")
