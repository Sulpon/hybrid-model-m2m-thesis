"""
Iterative (rolling) prediction and domain extrapolation.

Faithful implementation of Overgaard et al. Section 5.3, extended from 2 to all
7 candidate mechanisms and run on the regenerated Regime C data.

Procedure (paper, Sec. 5.3):
  1. Hybrid block construction. At decision point k, form X2^(k) by combining
     the OBSERVED trajectory for t <= k (from X2) with the KD PREDICTIONS for
     t > k (from M2).
  2. Model prediction. Predict end-of-batch Y from (X1, M2, X2^(k)).
  3. Performance evaluation. RMSEP against the true test values.

Critical detail from the paper: "Because future stage 2 trajectories are
unavailable at runtime, we refit the multiblock models at the first decision
point using (X1, M2, X2^(1)) to ensure calibration is based on KD forecasts
rather than the true (and still unseen) future trajectories." So CALIBRATION
uses X2^(1), not the full measured trajectory. Scaling, LV selection and the
SO-PLS orthogonalisation are all derived from that calibration.

Block layout matches the paper: the KD inputs present in X2 (t2, T2, CD0) are
moved into M2, so X2 holds the 6 x 7 species trajectory only and M2 holds the
6 x 7 simulated trajectory plus those three columns.

Two methods are run, as in the paper:
  SO-PLS   block-wise LVs, X2^(k) orthogonalised against M2
  MB-PLS   single global LV count on the block-scaled concatenation

Reported per decision point k and separately for the within-domain
(T2 in [330,370]) and extrapolation (T2 in [370,382]) test sets.

Diagnostics computed on calibration only, as required by the thesis method
section: point-estimate M2M at the argmin-RMSECV allocation, and the
flat-region (1-SE) median, for both the paper's SS ratio and the unique ratio.
Spearman correlations against extrapolation performance follow.
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
import warnings, json, time
warnings.filterwarnings('ignore')

R = 8.314
SPECIES = ['A', 'B', 'C', 'D', 'E', 'F']
MAX_X1, MAX_M2, MAX_X2 = 6, 15, 15
MAX_MB = 20
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
# kinetic parameters are refitted below with the paper-style initial conditions

def read(fn):
    x1 = pd.read_excel(fn, sheet_name='X1')[['CA0','T1','t1','CA1_final','CB1_final','CC1_final']].values
    x2 = pd.read_excel(fn, sheet_name='X2')
    y = pd.read_excel(fn, sheet_name='Y')['E_purity'].values.reshape(-1, 1)
    return x1, x2, y
X1c, D2c, Yc = read('RG_calibration.xlsx')
X1w, D2w, Yw = read('RG_within.xlsx')
X1e, D2e, Ye = read('RG_extrap.xlsx')
NPTS = (D2c.shape[1]-3)//6
TGRID = np.arange(NPTS)*15.0

DILUTION = 0.1/0.2      # V1/V2, as in the generator and the paper's Table 4

def parts(df, X1):
    """Stage-2 initial conditions follow the paper's Table 4: they come from the
    STAGE-1 outputs (diluted) and CD0 -- NOT from the first stage-2 measurement.

    This differs from the M2 construction used elsewhere in this project, which
    initialised from the measured first stage-2 sample. That choice is harmless
    offline but fatal here: it makes M2[:, t=1] exactly equal to the measured
    value, so X2^(1) becomes a column-for-column copy of M2 and the decision-
    point-1 calibration degenerates into fitting M2 twice.
    """
    n = len(df)
    CD0 = df['CD0'].values
    IC = np.column_stack([X1[:, 3]*DILUTION, X1[:, 4]*DILUTION, X1[:, 5]*DILUTION,
                          CD0, np.zeros(n), np.zeros(n)])
    flat = np.array([[df[f'{s}_t{k+1}'].iloc[i] for s in SPECIES for k in range(NPTS)] for i in range(n)])
    obs = np.array([[[df[f'{s}_t{k+1}'].iloc[i] for s in SPECIES] for k in range(NPTS)] for i in range(n)])
    meta = df[['t2_final', 'T2', 'CD0']].values
    return IC, flat, obs, meta, df['T2'].values
ICc, X2c, OBSc, MTc, T2c = parts(D2c, X1c)
ICw, X2w, OBSw, MTw, T2w = parts(D2w, X1w)
ICe, X2e, OBSe, MTe, T2e = parts(D2e, X1e)
print(f"decision points K = {NPTS}   n_cal={len(Yc)} n_within={len(Yw)} n_extrap={len(Ye)}", flush=True)

def build_M2(rhs, IC, T2v, meta, p):
    n = len(IC)
    M = np.zeros((n, 6, NPTS))
    for i in range(n):
        M[i] = odeint(rhs, IC[i], TGRID, args=(T2v[i], *p)).T
    return np.hstack([M.reshape(n, -1), meta]), M.reshape(n, -1)

def fit_params(rhs):
    """Refit [Ea3,Ea4,A3,A4] against the measured calibration trajectories using
    the paper-style initial conditions. The parameters in regen_fitted_params.json
    were fitted with the measured-first-point IC and do not apply here."""
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

def hybrid_X2(X2meas, M2traj, k):
    """X2^(k): measured for t <= k, KD prediction for t > k. Species-major layout."""
    out = X2meas.copy()
    for s in range(6):
        for t in range(k, NPTS):          # t is 0-based; k measured points are 0..k-1
            out[:, s*NPTS + t] = M2traj[:, s*NPTS + t]
    return out

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

# ---------------------------------------------------------------- SO-PLS
def sopls_cv(B1, B2, B3, Y, m1, m2, m3, B3scale=None):
    """B3scale supplies the statistics that define the X2 block's units -- the
    MEASURED calibration trajectories. Filling unobserved entries with KD values
    must not redefine those units: some KD columns are near-constant (M5 has one
    with sd = 1.4e-26), and scaling by such an sd amplifies the real measurements
    substituted at later decision points by up to 1e13."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _sf(B1[tr]); b1t, b1e = _sa(B1[tr],mu1,sd1,k1), _sa(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _sf(B2[tr]); b2t0, b2e0 = _sa(B2[tr],mu2,sd2,k2), _sa(B2[te],mu2,sd2,k2)
        mu3, sd3, k3 = _sf((B3scale if B3scale is not None else B3)[tr])
        b3t0, b3e0 = _sa(B3[tr],mu3,sd3,k3), _sa(B3[te],mu3,sd3,k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            b2t, gM = _rs(Tat, b2t0); b2e = b2e0 - Tae @ gM
            b3ta, gX = _rs(Tat, b3t0); b3ea = b3e0 - Tae @ gX
            Ya, _ = _rs(Tat, Y0)
            p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf); Tbt, Tbe = T2t[:,:be], T2e[:,:be]
                yh_ab = yh_a + Tbe @ Q2[:,:be].T
                b3tb, gX2 = _rs(Tbt, b3ta); b3eb = b3ea - Tbe @ gX2
                Yb, _ = _rs(Tbt, Ya)
                p3, T3t, Q3, cf = _fit(b3tb, Yb, m3); T3e = p3.transform(b3eb)
                for c in range(1, m3+1):
                    ce = min(c, cf)
                    yh = ((yh_ab + T3e[:,:ce] @ Q3[:,:ce].T)*sdy + muY).ravel()
                    P[te, a-1, b-1, c-1] = yh
                    rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None,None])**2).sum(0)/sst, rf

def sopls_fit(B1, B2, B3, Y, n1, n2, n3, B3scale=None):
    """Fit on the full calibration set; return everything needed to predict new blocks."""
    mu1, sd1, k1 = _sf(B1); b1 = _sa(B1, mu1, sd1, k1)
    mu2, sd2, k2 = _sf(B2); b20 = _sa(B2, mu2, sd2, k2)
    mu3, sd3, k3 = _sf(B3scale if B3scale is not None else B3); b30 = _sa(B3, mu3, sd3, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Y0 = (Y-muY)/sdy
    p1, T1, Q1, af = _fit(b1, Y0, n1); ae = min(n1, af); T1, Q1 = T1[:,:ae], Q1[:,:ae]
    b2r, gM = _rs(T1, b20); b3a, gX = _rs(T1, b30); Ya, _ = _rs(T1, Y0)
    p2, T2v, Q2, bf = _fit(b2r, Ya, n2); be = min(n2, bf); T2v, Q2 = T2v[:,:be], Q2[:,:be]
    b3b, gX2 = _rs(T2v, b3a); Yb, _ = _rs(T2v, Ya)
    p3, T3, Q3, cf = _fit(b3b, Yb, n3); ce = min(n3, cf); T3, Q3 = T3[:,:ce], Q3[:,:ce]
    ss = dict(SS_X1=float(np.sum((T1 @ Q1.T)**2)), SS_M2=float(np.sum((T2v @ Q2.T)**2)),
              SS_X2=float(np.sum((T3 @ Q3.T)**2)), SS_Y=float(np.sum(Y0**2)))
    ss['SS_F'] = ss['SS_Y'] - ss['SS_X1'] - ss['SS_M2'] - ss['SS_X2']
    return dict(sc=(mu1, sd1, k1, mu2, sd2, k2, mu3, sd3, k3, muY, sdy),
                p1=p1, Q1=Q1, ae=ae, gM=gM, gX=gX, p2=p2, Q2=Q2, be=be, gX2=gX2,
                p3=p3, Q3=Q3, ce=ce, ss=ss)

def sopls_predict(mdl, S1, S2, S3):
    mu1, sd1, k1, mu2, sd2, k2, mu3, sd3, k3, muY, sdy = mdl['sc']
    b1 = _sa(S1, mu1, sd1, k1); b2 = _sa(S2, mu2, sd2, k2); b3 = _sa(S3, mu3, sd3, k3)
    T1 = mdl['p1'].transform(b1)[:, :mdl['ae']]
    yh = T1 @ mdl['Q1'].T
    b2 = b2 - T1 @ mdl['gM']; b3 = b3 - T1 @ mdl['gX']
    T2v = mdl['p2'].transform(b2)[:, :mdl['be']]
    yh = yh + T2v @ mdl['Q2'].T
    b3 = b3 - T2v @ mdl['gX2']
    T3 = mdl['p3'].transform(b3)[:, :mdl['ce']]
    yh = yh + T3 @ mdl['Q3'].T
    return (yh*sdy + muY).ravel()

# ---------------------------------------------------------------- MB-PLS
def mb_concat(B1, B2, B3, sc=None, B3scale=None):
    if sc is None:
        sc = [_sf(B1), _sf(B2), _sf(B3scale if B3scale is not None else B3)]
    parts_ = [_sa(B, *s) for B, s in zip([B1, B2, B3], sc)]
    return np.hstack(parts_), sc

def mbpls_cv(B1, B2, B3, Y, mmax, B3scale=None):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    rf = np.zeros((N_SPLITS, mmax))
    for f, (tr, te) in enumerate(cv.split(Y)):
        Xt, sc = mb_concat(B1[tr], B2[tr], B3[tr],
                           B3scale=None if B3scale is None else B3scale[tr])
        Xe, _ = mb_concat(B1[te], B2[te], B3[te], sc)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p, T, Q, nf = _fit(Xt, Y0, mmax); Te = p.transform(Xe)
        for m in range(1, mmax+1):
            me = min(m, nf)
            yh = ((Te[:,:me] @ Q[:,:me].T)*sdy + muY).ravel()
            rf[f, m-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    return rf.mean(0)

def mbpls_fit(B1, B2, B3, Y, m, B3scale=None):
    Xt, sc = mb_concat(B1, B2, B3, B3scale=B3scale)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Y0 = (Y-muY)/sdy
    p, T, Q, nf = _fit(Xt, Y0, m)
    return dict(p=p, Q=Q, me=min(m, nf), sc=sc, muY=muY, sdy=sdy)

def mbpls_predict(mdl, S1, S2, S3):
    Xe, _ = mb_concat(S1, S2, S3, mdl['sc'])
    Te = mdl['p'].transform(Xe)[:, :mdl['me']]
    return ((Te @ mdl['Q'].T)*mdl['sdy'] + mdl['muY']).ravel()

# ================================================================= run
SSTw = float(((Yw - Yw.mean())**2).sum()); SSTe = float(((Ye - Ye.mean())**2).sum())
rows, curves = [], []
t0 = time.time()
for name, rhs in MODELS.items():
    p, fitcost = fit_params(rhs)
    M2c, Tc = build_M2(rhs, ICc, T2c, MTc, p)
    M2w, Tw_ = build_M2(rhs, ICw, T2w, MTw, p)
    M2e, Te_ = build_M2(rhs, ICe, T2e, MTe, p)

    # ---- calibration blocks use X2^(1), per the paper ----
    X2c_1 = hybrid_X2(X2c, Tc, 1)

    q, rf = sopls_cv(X1c, M2c, X2c_1, Yc, MAX_X1, MAX_M2, MAX_X2, B3scale=X2c)
    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin]); se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    a_arg, b_arg, c_arg = imin[0]+1, imin[1]+1, imin[2]+1
    mask = rmse <= rmin + se
    # The paper selects by parsimony inside the flat region, not by raw argmin:
    # "performance is flat near this minimum ... we select the configuration
    # {4,6,2}, keeping the number of LVs as low as possible to improve model
    # stability" (Sec. 5.2.1). This matters here: at k=1 the X2 block is nearly
    # collinear with M2, so the raw argmin lands on degenerate all-M2 or all-X2
    # splits that behave badly at later decision points.
    idxf = np.argwhere(mask); totf = idxf.sum(1) + 3
    cand = idxf[totf == totf.min()]
    pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])
    a, b, c = pick[0]+1, pick[1]+1, pick[2]+1

    mdl = sopls_fit(X1c, M2c, X2c_1, Yc, a, b, c, B3scale=X2c)
    ss = mdl['ss']
    m2m_point = ss['SS_M2']/ss['SS_X2']

    # flat-region M2M values (SS ratio and unique ratio), calibration only
    ssm_r, ssx_r, uniq_r = [], [], []
    for idx in np.argwhere(mask):
        aa, bb, cc = idx[0]+1, idx[1]+1, idx[2]+1
        m_ = sopls_fit(X1c, M2c, X2c_1, Yc, aa, bb, cc, B3scale=X2c)
        ssm_r.append(m_['ss']['SS_M2']); ssx_r.append(m_['ss']['SS_X2'])
    ssm_r, ssx_r = np.array(ssm_r), np.array(ssx_r)
    m2m_flat = float(np.median(ssm_r/ssx_r))

    # ---- variant B: diagnostics from the OFFLINE model (full measured X2) ----
    # The paper computes M2M offline (Sec. 5.2) and warns that the k=1 version is
    # "not correctly balanced" because X2^(1) blends measurements with KD
    # forecasts (Table 10). The offline model is then rolled forward on test data.
    qB, rfB = sopls_cv(X1c, M2c, X2c, Yc, MAX_X1, MAX_M2, MAX_X2)
    rmB = rfB.mean(0); iB = np.unravel_index(rmB.argmin(), rmB.shape)
    rminB = float(rmB[iB]); seB = float(rfB[:, iB[0], iB[1], iB[2]].std(ddof=1)/np.sqrt(N_SPLITS))
    maskB = rmB <= rminB + seB
    idxB = np.argwhere(maskB); totB = idxB.sum(1) + 3
    candB = idxB[totB == totB.min()]
    pickB = tuple(candB[np.array([rmB[tuple(u)] for u in candB]).argmin()])
    aB, bB, cB = pickB[0]+1, pickB[1]+1, pickB[2]+1
    mdlB = sopls_fit(X1c, M2c, X2c, Yc, aB, bB, cB)
    ssB = mdlB['ss']
    m2mB_point = ssB['SS_M2']/ssB['SS_X2']
    rr = []
    for idx in np.argwhere(maskB):
        mm = sopls_fit(X1c, M2c, X2c, Yc, idx[0]+1, idx[1]+1, idx[2]+1)
        rr.append(mm['ss']['SS_M2']/mm['ss']['SS_X2'])
    m2mB_flat = float(np.median(rr))

    mbg = mbpls_cv(X1c, M2c, X2c_1, Yc, MAX_MB, B3scale=X2c)
    mb_se = float(np.nan)
    mb_lv = int(mbg.argmin()+1)
    mbm = mbpls_fit(X1c, M2c, X2c_1, Yc, mb_lv, B3scale=X2c)

    # ---- rolling prediction over decision points ----
    rec = {}
    for meth, fitted_m, pred in [('SO-PLS', mdl, sopls_predict),
                                 ('SO-PLS-offlineCal', mdlB, sopls_predict),
                                 ('MB-PLS', mbm, mbpls_predict)]:
        for tag, X1s, M2s, X2s, Ts, Ys, SST in [
                ('within', X1w, M2w, X2w, Tw_, Yw, SSTw),
                ('extrap', X1e, M2e, X2e, Te_, Ye, SSTe)]:
            vals = []
            for k in range(1, NPTS+1):
                Xk = hybrid_X2(X2s, Ts, k)
                yh = pred(fitted_m, X1s, M2s, Xk)
                rmsep = float(np.sqrt(np.mean((yh - Ys.ravel())**2)))
                r2 = 1 - len(Ys)*rmsep**2/SST
                vals.append(rmsep)
                curves.append(dict(model=name, method=meth, regime=tag, k=k,
                                   RMSEP=rmsep, R2p=r2))
            rec[(meth, tag)] = np.array(vals)

    so_w, so_e = rec[('SO-PLS', 'within')], rec[('SO-PLS', 'extrap')]
    mb_w, mb_e = rec[('MB-PLS', 'within')], rec[('MB-PLS', 'extrap')]
    rows.append(dict(model=name,
        LV=f"({a},{b},{c})", LV_argmin=f"({a_arg},{b_arg},{c_arg})", RMSECV=rmin, SE=se, n_1SE=int(mask.sum()), Q2_cal=float(q[imin]),
        SS_X1=ss['SS_X1'], SS_M2=ss['SS_M2'], SS_X2=ss['SS_X2'], SS_F=ss['SS_F'],
        M2M_point=m2m_point, M2M_flatmed=m2m_flat,
        LV_offline=f"({aB},{bB},{cB})", M2M_off_point=m2mB_point, M2M_off_flatmed=m2mB_flat,
        Q2_cal_offline=float(qB[pickB]), RMSECV_offline=float(rmB[pickB]),
        MB_LV=mb_lv,
        SO_within_mean=so_w.mean(), SO_extrap_mean=so_e.mean(),
        SO_pct_increase=100*(so_e.mean()-so_w.mean())/so_w.mean(),
        SO_within_k1=so_w[0], SO_extrap_k1=so_e[0],
        SO_within_kK=so_w[-1], SO_extrap_kK=so_e[-1],
        MB_within_mean=mb_w.mean(), MB_extrap_mean=mb_e.mean(),
        MB_pct_increase=100*(mb_e.mean()-mb_w.mean())/mb_w.mean()))
    r = rows[-1]
    print(f"{name:14s} LV={r['LV']:10s} RMSECV={rmin:.4f} n1SE={int(mask.sum()):4d}  "
          f"M2M pt={m2m_point:6.2f} flat={m2m_flat:6.2f} | SO within={so_w.mean():.4f} "
          f"extrap={so_e.mean():.4f} (+{r['SO_pct_increase']:5.1f}%) | MB +{r['MB_pct_increase']:5.1f}%",
          flush=True)

df = pd.DataFrame(rows); cv_df = pd.DataFrame(curves)
pd.set_option('display.width', 340, 'display.max_columns', 80)
print("\n=== calibration at decision point 1 (X1, M2, X2^(1)) ===")
print(df[['model','LV','LV_argmin','RMSECV','Q2_cal','n_1SE','SS_X1','SS_M2','SS_X2','SS_F',
          'M2M_point','M2M_flatmed','MB_LV']].round(4).to_string(index=False))
print("\n=== rolling prediction, averaged over decision points ===")
print(df[['model','SO_within_mean','SO_extrap_mean','SO_pct_increase',
          'MB_within_mean','MB_extrap_mean','MB_pct_increase']].round(4).to_string(index=False))
print("\n=== SO-PLS RMSEP at the first and last decision point ===")
print(df[['model','SO_within_k1','SO_extrap_k1','SO_within_kK','SO_extrap_kK']].round(4).to_string(index=False))

print("\n=== Spearman: calibration diagnostic vs extrapolation performance (n=7) ===")
for dcol in ['M2M_point', 'M2M_flatmed']:
    for ocol in ['SO_extrap_mean', 'SO_pct_increase', 'SO_extrap_k1']:
        rho, pv = spearmanr(df[dcol], df[ocol])
        print(f"  rho({dcol:12s}, {ocol:16s}) = {rho:+.3f}  (p={pv:.3f})")

piv = cv_df[cv_df.method == 'SO-PLS'].pivot_table(index='k', columns=['model', 'regime'], values='RMSEP')
with pd.ExcelWriter('iterative_prediction.xlsx') as w:
    df.to_excel(w, sheet_name='summary', index=False)
    cv_df.to_excel(w, sheet_name='rmsep_by_k', index=False)

names = list(MODELS)
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
for i, nm in enumerate(names):
    ax = axes[i//4, i % 4]
    for meth, col in [('SO-PLS', '#DD8452'), ('MB-PLS', '#4C72B0')]:
        s = cv_df[(cv_df.model == nm) & (cv_df.method == meth)]
        w_ = s[s.regime == 'within'].sort_values('k')
        e_ = s[s.regime == 'extrap'].sort_values('k')
        ax.plot(w_.k, w_.RMSEP, '-o', ms=3.5, color=col, label=f'{meth} within')
        ax.plot(e_.k, e_.RMSEP, ':o', ms=3.5, color=col, label=f'{meth} extrap')
    ax.set_title(nm, fontsize=9.5); ax.tick_params(labelsize=8)
    if i//4 == 1: ax.set_xlabel('Decision point k', fontsize=9)
    if i % 4 == 0: ax.set_ylabel('RMSEP', fontsize=9)
    if i == 0: ax.legend(fontsize=6.5)
axes[1, 3].axis('off')
d = df.sort_values('M2M_flatmed')
axes[1, 3].axis('on')
axes[1, 3].scatter(d.M2M_flatmed, d.SO_extrap_mean, s=70, c='#C44E52')
for _, r in d.iterrows():
    axes[1, 3].annotate(r.model.split('_')[0], (r.M2M_flatmed, r.SO_extrap_mean),
                        textcoords='offset points', xytext=(6, 4), fontsize=8)
axes[1, 3].set_xlabel('flat-region median M2M', fontsize=9)
axes[1, 3].set_ylabel('mean RMSEP extrapolation', fontsize=9)
axes[1, 3].set_title('diagnostic vs extrapolation', fontsize=9.5)
fig.suptitle('Iterative (rolling) prediction: RMSEP at each decision point, '
             'solid = within-domain, dotted = extrapolation', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('iterative_prediction.png', dpi=160)
print(f"\nSaved iterative_prediction.xlsx / .png   ({time.time()-t0:.0f}s)")
