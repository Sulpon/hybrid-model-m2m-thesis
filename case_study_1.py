"""
Case study 1 -- two-stage batch reactor. Complete pipeline in one file.

    python case_study_1.py                # commonality + flatregion + q2cost
    python case_study_1.py commonality    # uM / uX / shared and the modified ratio
    python case_study_1.py flatregion     # ratio distribution over the 1/2/3-SE regions
    python case_study_1.py q2cost         # predictive cost of each stopping rule
    python case_study_1.py rolling        # rolling prediction + extrapolation (slow)

TWO DATASETS, DELIBERATELY
The case study grew across two generations of data and both are still in use,
so both are carried here rather than silently unified:

  A  "original"  X1/X2/Y/M2.xlsx (+ _mis for M6), kinetic parameters hard-coded
                 in FITTED_A. Used by `flatregion` and `q2cost`.
  C  "regenerated"  RG_calibration/within/extrap.xlsx with the corrected sigma
                 values and paper-style stage-2 initial conditions, parameters
                 from regen_fitted_params_paperIC.json or refitted on the fly.
                 Used by `commonality` and `rolling`.

They do not agree on the ranking of the seven mechanisms -- that disagreement
is itself a reported finding -- so the stages are NOT interchangeable and the
dataset each one uses is stated in its banner.

CONVENTIONS (unchanged throughout)
KFold(10, shuffle, seed 42); block scaling 1/sqrt(K); RMSECV = mean of the
per-fold RMSE; Q2 = 1 - pooled PRESS / SST; allocations chosen by parsimony
within the 1-SE flat region. Blocks are X1 -> M2 -> X2, with the stage-2
setpoints (CD0, t2, T2) held in M2 rather than X2 per the source paper's kappa
rule. Do not alter the SO-PLS routines.
"""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

R = 8.314
SPECIES = ['A', 'B', 'C', 'D', 'E', 'F']
MAX_X1, MAX_M2, MAX_X2 = 6, 15, 15
MAX_MB = 20
N_SPLITS, SEED = 10, 42
KS = [1, 2, 3]
DILUTION = 0.1/0.2                     # V1/V2, as in the generator and Table 4

# ===========================================================================
#  1. THE SEVEN CANDIDATE MECHANISMS
# ===========================================================================
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
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea3/(R*T2))    # Ea3 reused
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


MODELS = {'M0_correct': ode_correct, 'M1_no_side': ode_no_side,
          'M2_order1_D': ode_order1_D, 'M3_wrong_Ea4': ode_wrong_Ea4,
          'M4_lumped_EF': ode_lumped, 'M5_no_D': ode_no_D,
          'M6_author_mis': ode_author_mis}

# kinetic parameters [Ea3, Ea4, A3, A4] fitted on dataset A
FITTED_A = {
    'M0_correct':    [4.9380789e+04, 5.6481086e+04, 8.023, 33.615],
    'M1_no_side':    [5.0088694e+04, 5.5000000e+04, 11.377, 20.0],
    'M2_order1_D':   [3.1123942e+04, 4.3435688e+04, 6.397, 145.759],
    'M3_wrong_Ea4':  [4.99379e+04, 5.50000e+04, 9.75, 3.506],
    'M4_lumped_EF':  [5.0039241e+04, 5.5000000e+04, 9.883, 20.0],
    'M5_no_D':       [2577.581, 658929.499, 9735.492, 9822.344],
    'M6_author_mis': [3.5882909e+04, 5.5000000e+04, 38.713, 20.0],
}

# ===========================================================================
#  2. PLS / SO-PLS MACHINERY  (unchanged -- do not alter)
# ===========================================================================
def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n


def _sf(B, block_scale=False):
    """Column means and standard deviations, with a guard on near-constant
    columns, plus the block-scaling divisor.

    Block scaling (dividing by sqrt(K), K the number of columns) is needed
    only where several blocks are combined into one super-score, i.e. MB-PLS.
    SO-PLS fits each block separately, and a PLS model is invariant to a
    constant rescaling of its predictor block -- the weights are unchanged and
    the factor cancels between the scores and the response loading -- so it is
    not applied there. Hence the default is off, and mb_concat requests it."""
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, (np.sqrt(B.shape[1]) if block_scale else 1.0)


def _sa(B, mu, sd, k):
    return (B - mu)/sd/k


def _rs(T, tgt):
    g = np.linalg.pinv(T.T @ T) @ T.T @ tgt
    return tgt - T @ g, g


def cv_1(B, Y, m):
    """Q2 curve for a single block, plus the per-fold RMSE grid."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m)); rf = np.zeros((N_SPLITS, m))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu, sd, k = _sf(B[tr]); bt, be = _sa(B[tr], mu, sd, k), _sa(B[te], mu, sd, k)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p, T, Q, nf = _fit(bt, Y0, m); Te = p.transform(be)
        for a in range(1, m+1):
            ae = min(a, nf)
            yh = ((Te[:, :ae] @ Q[:, :ae].T)*sdy + muY).ravel()
            P[te, a-1] = yh
            rf[f, a-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    q = 1 - ((P - Y.reshape(-1, 1))**2).sum(0)/np.sum((Y - Y.mean(0))**2)
    return q, rf


def cv_2(B1, B2, Y, m1, m2):
    """Q2 surface for SO-PLS B1 -> B2, plus the per-fold RMSE cube."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2)); rf = np.zeros((N_SPLITS, m1, m2))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _sf(B1[tr]); b1t, b1e = _sa(B1[tr], mu1, sd1, k1), _sa(B1[te], mu1, sd1, k1)
        mu2, sd2, k2 = _sf(B2[tr]); b2t0, b2e0 = _sa(B2[tr], mu2, sd2, k2), _sa(B2[te], mu2, sd2, k2)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            b2t, g2 = _rs(Tat, b2t0); b2e = b2e0 - Tae @ g2
            Ya, _ = _rs(Tat, Y0)
            p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf)
                yh = ((yh_a + T2e[:, :be] @ Q2[:, :be].T)*sdy + muY).ravel()
                P[te, a-1, b-1] = yh
                rf[f, a-1, b-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    q = 1 - ((P - Y[:, :, None])**2).sum(0)/np.sum((Y - Y.mean(0))**2)
    return q, rf


def cv_3(B1, B2, B3, Y, m1, m2, m3, B3scale=None):
    """SO-PLS B1 -> B2 -> B3. Returns the Q2 surface and per-fold RMSE cube.

    B3scale supplies the statistics that define the third block's units. It
    matters only for the rolling stage, where B3 is a hybrid of measurements
    and KD forecasts: some KD columns are near-constant (M5 has one with
    sd = 1.4e-26), and scaling by such an sd amplifies the real measurements
    substituted at later decision points by up to 1e13.
    """
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2, m3)); rf = np.zeros((N_SPLITS, m1, m2, m3))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _sf(B1[tr]); b1t, b1e = _sa(B1[tr], mu1, sd1, k1), _sa(B1[te], mu1, sd1, k1)
        mu2, sd2, k2 = _sf(B2[tr]); b2t0, b2e0 = _sa(B2[tr], mu2, sd2, k2), _sa(B2[te], mu2, sd2, k2)
        mu3, sd3, k3 = _sf((B3scale if B3scale is not None else B3)[tr])
        b3t0, b3e0 = _sa(B3[tr], mu3, sd3, k3), _sa(B3[te], mu3, sd3, k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            b2t, gM = _rs(Tat, b2t0); b2e = b2e0 - Tae @ gM
            b3ta, gX = _rs(Tat, b3t0); b3ea = b3e0 - Tae @ gX
            Ya, _ = _rs(Tat, Y0)
            p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf); Tbt, Tbe = T2t[:, :be], T2e[:, :be]
                yh_ab = yh_a + Tbe @ Q2[:, :be].T
                b3tb, gX2 = _rs(Tbt, b3ta); b3eb = b3ea - Tbe @ gX2
                Yb, _ = _rs(Tbt, Ya)
                p3, T3t, Q3, cf = _fit(b3tb, Yb, m3); T3e = p3.transform(b3eb)
                for c in range(1, m3+1):
                    ce = min(c, cf)
                    yh = ((yh_ab + T3e[:, :ce] @ Q3[:, :ce].T)*sdy + muY).ravel()
                    P[te, a-1, b-1, c-1] = yh
                    rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:, :, None, None])**2).sum(0)/sst, rf


def ss_surface_3(B1, B2, B3, Y, m1, m2, m3):
    """Type-I sum of squares taken by the THIRD block, over the whole grid."""
    mu, sd, k = _sf(B1); b1 = _sa(B1, mu, sd, k)
    mu, sd, k = _sf(B2); b2_0 = _sa(B2, mu, sd, k)
    mu, sd, k = _sf(B3); b3_0 = _sa(B3, mu, sd, k)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    out = np.zeros((m1, m2, m3))
    p1, T1, Q1, af = _fit(b1, ys, m1)
    for a in range(1, m1+1):
        ae = min(a, af); Ta = T1[:, :ae]
        b2_a, _ = _rs(Ta, b2_0); b3_a, _ = _rs(Ta, b3_0); y_a, _ = _rs(Ta, ys)
        p2, T2v, Q2, bf = _fit(b2_a, y_a, m2)
        for b in range(1, m2+1):
            be = min(b, bf); Tb = T2v[:, :be]
            b3_b, _ = _rs(Tb, b3_a); y_b, _ = _rs(Tb, y_a)
            p3, T3, Q3, cf = _fit(b3_b, y_b, m3)
            for c in range(1, m3+1):
                ce = min(c, cf)
                out[a-1, b-1, c-1] = float(np.sum((T3[:, :ce] @ Q3[:, :ce].T)**2))
    return out


def sopls_fit(B1, B2, B3, Y, n1, n2, n3, B3scale=None):
    """Fit on the full calibration set; return everything needed to predict."""
    mu1, sd1, k1 = _sf(B1); b1 = _sa(B1, mu1, sd1, k1)
    mu2, sd2, k2 = _sf(B2); b20 = _sa(B2, mu2, sd2, k2)
    mu3, sd3, k3 = _sf(B3scale if B3scale is not None else B3); b30 = _sa(B3, mu3, sd3, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); Y0 = (Y-muY)/sdy
    p1, T1, Q1, af = _fit(b1, Y0, n1); ae = min(n1, af); T1, Q1 = T1[:, :ae], Q1[:, :ae]
    b2r, gM = _rs(T1, b20); b3a, gX = _rs(T1, b30); Ya, _ = _rs(T1, Y0)
    p2, T2v, Q2, bf = _fit(b2r, Ya, n2); be = min(n2, bf); T2v, Q2 = T2v[:, :be], Q2[:, :be]
    b3b, gX2 = _rs(T2v, b3a); Yb, _ = _rs(T2v, Ya)
    p3, T3, Q3, cf = _fit(b3b, Yb, n3); ce = min(n3, cf); T3, Q3 = T3[:, :ce], Q3[:, :ce]
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
    return ((yh + T3 @ mdl['Q3'].T)*sdy + muY).ravel()


def mb_concat(B1, B2, B3, sc=None, B3scale=None):
    if sc is None:
        # MB-PLS concatenates the blocks, so their relative widths matter here
        sc = [_sf(B1, True), _sf(B2, True),
              _sf(B3scale if B3scale is not None else B3, True)]
    return np.hstack([_sa(B, *s) for B, s in zip([B1, B2, B3], sc)]), sc


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
            yh = ((Te[:, :me] @ Q[:, :me].T)*sdy + muY).ravel()
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


def pick_1se(rmse, rf, k=1):
    """Parsimony within the k-SE flat region; ties broken by lower RMSECV."""
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin])
    se = float(rf[(slice(None),) + imin].std(ddof=1)/np.sqrt(N_SPLITS))
    mask = rmse <= rmin + k*se
    idx = np.argwhere(mask); tot = idx.sum(1) + rmse.ndim
    cand = idx[tot == tot.min()]
    pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])
    return pick, mask, rmin, se, imin


# ===========================================================================
#  3. DATASETS
# ===========================================================================
def load_A():
    """Original data: X1/X2/Y/M2.xlsx, with _mis variants for M6."""
    def grp(tag):
        sfx = '_mis' if tag == 'mis' else ''
        X1 = pd.read_excel(f'X1{sfx}.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
        X2df = pd.read_excel(f'X2{sfx}.xlsx')
        Y = pd.read_excel(f'Y{sfx}.xlsx')['E_pur'].values.reshape(-1, 1)
        meta = pd.read_excel(f'M2{sfx}.xlsx')[['t2', 'T2', 'D0']]
        n_pts = X2df.shape[1]//6
        IC = np.array([[X2df[f'{s}_1'].iloc[i] for s in SPECIES] for i in range(len(X2df))])
        X2 = np.array([[X2df[f'{s}_{k+1}'].iloc[i] for s in SPECIES for k in range(n_pts)]
                       for i in range(len(X2df))])
        return dict(X1=X1, X2=X2, Y=Y, meta=meta, n_pts=n_pts, IC=IC)
    return {'std': grp('std'), 'mis': grp('mis')}


def build_M2_A(rhs, d, params):
    n = len(d['X1']); tgrid = np.arange(d['n_pts'])*15.0
    M = np.zeros((n, 6, d['n_pts']))
    for i in range(n):
        M[i] = odeint(rhs, d['IC'][i], tgrid, args=(d['meta']['T2'].values[i], *params)).T
    return np.hstack([M.reshape(n, -1), d['meta'].values])


def load_C(fn):
    """Regenerated data. Stage-2 initial conditions follow the paper's Table 4:
    they come from the diluted STAGE-1 outputs plus CD0, not from the first
    stage-2 measurement. That distinction is harmless offline but fatal in the
    rolling stage -- it would make X2^(1) a column-for-column copy of M2."""
    X1 = pd.read_excel(fn, sheet_name='X1')[
        ['CA0', 'T1', 't1', 'CA1_final', 'CB1_final', 'CC1_final']].values
    df = pd.read_excel(fn, sheet_name='X2')
    Y = pd.read_excel(fn, sheet_name='Y')['E_purity'].values.reshape(-1, 1)
    n = len(Y); npts = (df.shape[1]-3)//6
    CD0 = df['CD0'].values
    IC = np.column_stack([X1[:, 3]*DILUTION, X1[:, 4]*DILUTION, X1[:, 5]*DILUTION,
                          CD0, np.zeros(n), np.zeros(n)])
    X2 = np.array([[df[f'{s}_t{k+1}'].iloc[i] for s in SPECIES for k in range(npts)]
                   for i in range(n)])
    obs = np.array([[[df[f'{s}_t{k+1}'].iloc[i] for s in SPECIES] for k in range(npts)]
                    for i in range(n)])
    return dict(X1=X1, X2=X2, Y=Y, IC=IC, obs=obs, npts=npts,
                meta=df[['t2_final', 'T2', 'CD0']].values, T2=df['T2'].values)


def build_M2_C(rhs, d, p):
    n = len(d['Y']); tgrid = np.arange(d['npts'])*15.0
    M = np.zeros((n, 6, d['npts']))
    for i in range(n):
        M[i] = odeint(rhs, d['IC'][i], tgrid, args=(d['T2'][i], *p)).T
    flat = M.reshape(n, -1)
    return np.hstack([flat, d['meta']]), flat


def hybrid_X2(X2meas, M2traj, k, npts):
    """X2^(k): measured for t <= k, KD prediction for t > k."""
    out = X2meas.copy()
    for s in range(6):
        out[:, s*npts + k:(s+1)*npts] = M2traj[:, s*npts + k:(s+1)*npts]
    return out


# ===========================================================================
#  STAGES
# ===========================================================================
def stage_commonality():
    print('=' * 74)
    print('COMMONALITY  --  dataset C (regenerated, corrected sigma, paper ICs)')
    print('=' * 74)
    d = load_C('RG_calibration.xlsx')
    cache = json.load(open('regen_fitted_params_paperIC.json'))
    Y = d['Y']
    qX1, rfX1 = cv_1(d['X1'], Y, MAX_X1)
    pX1, *_ = pick_1se(rfX1.mean(0), rfX1)
    rows = []
    for name, rhs in MODELS.items():
        M2, _ = build_M2_C(rhs, d, cache[name])
        qF, rf = cv_3(d['X1'], M2, d['X2'], Y, MAX_X1, MAX_M2, MAX_X2)
        qKD, rfKD = cv_2(d['X1'], M2, Y, MAX_X1, MAX_M2)          # X1 + M2
        qDD, rfDD = cv_2(d['X1'], d['X2'], Y, MAX_X1, MAX_X2)     # X1 + X2
        pick, mask, rmin, se, _ = pick_1se(rf.mean(0), rf)
        # Each reduced model is RE-OPTIMISED on its own grid rather than
        # inheriting the full model's per-block counts. Inheriting makes the
        # difference a measure of "this block at someone else's allocation"
        # instead of what the block actually adds, and the sub-model
        # allocation moves the ratio several times more than the full one.
        pkd, *_ = pick_1se(rfKD.mean(0), rfKD)
        pdd, *_ = pick_1se(rfDD.mean(0), rfDD)
        a, b, c = pick
        uM = float(qF[pick] - qDD[pdd])
        uX = float(qF[pick] - qKD[pkd])
        joint = float(qF[pick] - qX1[pX1])
        rows.append(dict(model=name, LV=f'({a+1},{b+1},{c+1})', n_1SE=int(mask.sum()),
                         Q2_full=float(qF[pick]), joint=joint, uM=uM, uX=uX,
                         shared=joint-uM-uX, ratio=uM/uX if uX > 1e-9 else np.nan,
                         shared_frac=(joint-uM-uX)/joint,
                         coverage=(uM+uX)/joint))
        print(f'{name:14s} LV={rows[-1]["LV"]:9s} Q2 {qF[pick]:.4f}  uM {uM:+.4f}  '
              f'uX {uX:+.4f}  shared {joint-uM-uX:+.4f}  M2M* {rows[-1]["ratio"]:7.3f}',
              flush=True)

    df = pd.DataFrame(rows).set_index('model')
    for src, col, new in [('regen_fit_and_test_paperIC.xlsx', 'fit_cost', 'fit_cost')]:
        if os.path.exists(src):
            df[new] = pd.read_excel(src).set_index('model')[col]
    if os.path.exists('case_study_1_rolling.xlsx'):
        c = pd.read_excel('case_study_1_rolling.xlsx', sheet_name='rmsep_by_k')
        c = c[(c.method == 'SO-PLS-offlineCal') & (c.regime == 'extrap')]
        df['roll_R2e'] = c.groupby('model').R2p.mean()
    df.to_excel('case_study_1_commonality.xlsx')

    if 'fit_cost' in df and 'roll_R2e' in df:
        print('\nSpearman against independent quality measures (n=7)')
        for col in ['uM', 'uX', 'shared', 'coverage', 'ratio', 'Q2_full']:
            s = df[[col, 'fit_cost']].dropna(); a_ = spearmanr(s[col], s.fit_cost)
            s2 = df[[col, 'roll_R2e']].dropna(); b_ = spearmanr(s2[col], s2.roll_R2e)
            print(f'  {col:10s} vs fit_cost {a_.correlation:+.3f} (p={a_.pvalue:.3f})'
                  f'   vs rolling R2 {b_.correlation:+.3f} (p={b_.pvalue:.3f})')
    print('\nSaved case_study_1_commonality.xlsx')


def _dataset_A_blocks():
    A = load_A()
    for name, rhs in MODELS.items():
        d = A['mis'] if name == 'M6_author_mis' else A['std']
        yield name, rhs, d, build_M2_A(rhs, d, FITTED_A[name])


def stage_flatregion():
    print('=' * 74)
    print('FLAT REGION  --  dataset A (original).  Ratio over the 1/2/3-SE regions')
    print('=' * 74)
    rows, store = [], {}
    for name, rhs, d, M2 in _dataset_A_blocks():
        X1, X2, Y = d['X1'], d['X2'], d['Y']
        ssX = ss_surface_3(X1, M2, X2, Y, MAX_X1, MAX_M2, MAX_X2)
        ssM = ss_surface_3(X1, X2, M2, Y, MAX_X1, MAX_X2, MAX_M2)
        ratio_ss = np.transpose(ssM, (0, 2, 1))/ssX

        qF, rf = cv_3(X1, M2, X2, Y, MAX_X1, MAX_M2, MAX_X2)
        qKD, rfKD = cv_2(X1, M2, Y, MAX_X1, MAX_M2)
        qDD, rfDD = cv_2(X1, X2, Y, MAX_X1, MAX_X2)
        # reduced models re-optimised once on their own grids (see stage_commonality)
        pkd, *_ = pick_1se(rfKD.mean(0), rfKD)
        pdd, *_ = pick_1se(rfDD.mean(0), rfDD)
        q_kd, q_dd = float(qKD[pkd]), float(qDD[pdd])
        with np.errstate(divide='ignore', invalid='ignore'):
            uM_q, uX_q = qF - q_dd, qF - q_kd
            ratio_q = np.where(uX_q > 1e-9, uM_q/uX_q, np.nan)

        rmse = rf.mean(0)
        imin = np.unravel_index(rmse.argmin(), rmse.shape); rmin = rmse[imin]
        se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
        a, b, c = imin[0]+1, imin[1]+1, imin[2]+1
        masks = {'full grid': np.ones_like(rmse, dtype=bool)}
        for k in KS:
            masks[f'{k}-SE'] = rmse <= rmin + k*se
        store[name] = dict(ratio_ss=ratio_ss, ratio_q=ratio_q, masks=masks, opt=(a, b, c))

        def desc(v):
            ok = v[np.isfinite(v)]
            return (float(np.median(ok)), float(np.percentile(ok, 10)),
                    float(np.percentile(ok, 90)), float(ok.min()), float(ok.max()),
                    float((ok > 1).mean())) if ok.size else (np.nan,)*6
        for lab, m in masks.items():
            s_ = desc(ratio_ss[m]); q_ = desc(ratio_q[m])
            nund = int(np.isnan(ratio_q[m]).sum())
            rows.append(dict(model=name, region=lab, n_cells=int(m.sum()),
                             RMSECV_min=rmin, SE=se, opt_LV=f'({a},{b},{c})',
                             ss_median=s_[0], ss_p10=s_[1], ss_p90=s_[2],
                             ss_min=s_[3], ss_max=s_[4], ss_frac_gt1=s_[5],
                             ss_at_opt=ratio_ss[a-1, b-1, c-1],
                             q_median=q_[0], q_p10=q_[1], q_p90=q_[2],
                             q_min=q_[3], q_max=q_[4], q_frac_gt1=q_[5],
                             q_n_undefined=nund, q_frac_undefined=nund/m.sum(),
                             q_at_opt=ratio_q[a-1, b-1, c-1]))
        print(f'{name:14s} opt=({a},{b},{c}) RMSECV={rmin:.4f} SE={se:.4f}  '
              f'sizes={ {l: int(m.sum()) for l, m in masks.items()} }', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('case_study_1_flat_regions.xlsx', index=False)
    print('\n--- median modified M2M* by region ---')
    print(df.pivot(index='model', columns='region', values='q_median')
            [['1-SE', '2-SE', '3-SE', 'full grid']].round(3).to_string())
    print('\n--- P(ratio > 1) by region ---')
    print(df.pivot(index='model', columns='region', values='q_frac_gt1')
            [['1-SE', '2-SE', '3-SE', 'full grid']].round(3).to_string())
    print('\nSaved case_study_1_flat_regions.xlsx')


def stage_q2cost():
    print('=' * 74)
    print('Q2 COST  --  dataset A (original).  What each stopping rule costs')
    print('=' * 74)
    rows = []
    for name, rhs, d, M2 in _dataset_A_blocks():
        qF, rf = cv_3(d['X1'], M2, d['X2'], d['Y'], MAX_X1, MAX_M2, MAX_X2)
        rmse = rf.mean(0)
        imin = np.unravel_index(rmse.argmin(), rmse.shape)
        rmin = float(rmse[imin]); qopt = float(qF[imin])
        se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(N_SPLITS))
        a0, b0, c0 = imin[0]+1, imin[1]+1, imin[2]+1
        regions = [('argmin (full)', np.zeros_like(rmse, dtype=bool))]
        regions[0][1][imin] = True
        for k in KS:
            regions.append((f'{k}-SE', rmse <= rmin + k*se))
        for lab, mask in regions:
            idx = np.argwhere(mask); tot = idx.sum(1) + 3
            cand = idx[tot == tot.min()]
            pick = cand[np.array([rmse[tuple(u)] for u in cand]).argmin()]
            a, b, c = pick[0]+1, pick[1]+1, pick[2]+1
            qsel, rsel = float(qF[tuple(pick)]), float(rmse[tuple(pick)])
            rows.append(dict(model=name, rule=lab, n_region=int(mask.sum()),
                             LV=f'({a},{b},{c})', total_LV=int(a+b+c),
                             RMSECV=rsel, Q2=qsel, dQ2_pp=100*(qsel-qopt),
                             dRMSECV=rsel-rmin, opt_LV=f'({a0},{b0},{c0})',
                             opt_total_LV=a0+b0+c0, opt_RMSECV=rmin,
                             opt_Q2=qopt, SE=se))
        print(f'{name:14s} argmin=({a0},{b0},{c0}) RMSECV={rmin:.4f} Q2={qopt:.4f} '
              f'SE={se:.4f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('case_study_1_q2_cost.xlsx', index=False)
    order = ['argmin (full)', '1-SE', '2-SE', '3-SE']
    p = df.pivot_table(index='model', columns='rule', values=['Q2', 'total_LV'])
    print('\n--- Q2 of the selected model, by rule ---')
    print(p['Q2'][order].round(4).to_string())
    print('\n--- averaged over the seven candidates, relative to argmin ---')
    a_, al = p['Q2']['argmin (full)'], p['total_LV']['argmin (full)']
    for r in ['1-SE', '2-SE', '3-SE']:
        print(f'  {r:6s} dQ2 {(p["Q2"][r]-a_).mean():+.4f}   '
              f'latent variables saved {(al-p["total_LV"][r]).mean():.1f}')
    print('\nSaved case_study_1_q2_cost.xlsx')


def stage_rolling():
    print('=' * 74)
    print('ROLLING  --  dataset C.  Iterative prediction and domain extrapolation')
    print('=' * 74)
    t0 = time.time()
    C = {k: load_C(f'RG_{k}.xlsx') for k in ['calibration', 'within', 'extrap']}
    cal, wit, ext = C['calibration'], C['within'], C['extrap']
    NPTS = cal['npts']
    SSTw = float(((wit['Y'] - wit['Y'].mean())**2).sum())
    SSTe = float(((ext['Y'] - ext['Y'].mean())**2).sum())
    print(f'decision points K = {NPTS}   n_cal={len(cal["Y"])} '
          f'n_within={len(wit["Y"])} n_extrap={len(ext["Y"])}', flush=True)

    def fit_params(rhs):
        """Refit [Ea3,Ea4,A3,A4] against the measured calibration trajectories
        using the paper-style ICs. The cached parameters were fitted with the
        measured-first-point IC and do not apply here."""
        def resid(p):
            return np.concatenate([
                (odeint(rhs, cal['IC'][i], np.arange(NPTS)*15.0,
                        args=(cal['T2'][i], *p)) - cal['obs'][i]).ravel()
                for i in range(len(cal['Y']))])
        r = least_squares(resid, [50000., 55000., 10., 20.],
                          bounds=([1e3, 1e3, 1e-4, 1e-4], [1e6, 1e6, 1e4, 1e4]),
                          xtol=1e-8, ftol=1e-8, max_nfev=60)
        return r.x, float(r.cost)

    rows, curves = [], []
    for name, rhs in MODELS.items():
        p, _ = fit_params(rhs)
        M2c, Tc = build_M2_C(rhs, cal, p)
        M2w, Tw = build_M2_C(rhs, wit, p)
        M2e, Te = build_M2_C(rhs, ext, p)
        X2c_1 = hybrid_X2(cal['X2'], Tc, 1, NPTS)     # calibration uses X2^(1)

        q, rf = cv_3(cal['X1'], M2c, X2c_1, cal['Y'], MAX_X1, MAX_M2, MAX_X2,
                     B3scale=cal['X2'])
        pick, mask, rmin, se, imin = pick_1se(rf.mean(0), rf)
        a, b, c = pick[0]+1, pick[1]+1, pick[2]+1
        mdl = sopls_fit(cal['X1'], M2c, X2c_1, cal['Y'], a, b, c, B3scale=cal['X2'])
        ss = mdl['ss']

        # Offline variant: diagnostics from the full measured X2, then rolled
        # forward. The paper computes M2M offline and warns the k=1 version is
        # "not correctly balanced" because X2^(1) blends measurement with forecast.
        qB, rfB = cv_3(cal['X1'], M2c, cal['X2'], cal['Y'], MAX_X1, MAX_M2, MAX_X2)
        pickB, maskB, rminB, seB, _ = pick_1se(rfB.mean(0), rfB)
        aB, bB, cB = pickB[0]+1, pickB[1]+1, pickB[2]+1
        mdlB = sopls_fit(cal['X1'], M2c, cal['X2'], cal['Y'], aB, bB, cB)
        # The original also reported the MEDIAN SS-ratio over every cell of the
        # flat region, which refits SO-PLS up to 935 times per model. Nothing
        # downstream consumes it, so it is not carried over here.

        mb_lv = int(mbpls_cv(cal['X1'], M2c, X2c_1, cal['Y'], MAX_MB,
                             B3scale=cal['X2']).argmin() + 1)
        mbm = mbpls_fit(cal['X1'], M2c, X2c_1, cal['Y'], mb_lv, B3scale=cal['X2'])

        rec = {}
        for meth, fm, pred in [('SO-PLS', mdl, sopls_predict),
                               ('SO-PLS-offlineCal', mdlB, sopls_predict),
                               ('MB-PLS', mbm, mbpls_predict)]:
            for tag, dd, M2s, Ts, SST in [('within', wit, M2w, Tw, SSTw),
                                          ('extrap', ext, M2e, Te, SSTe)]:
                vals = []
                for k in range(1, NPTS+1):
                    yh = pred(fm, dd['X1'], M2s, hybrid_X2(dd['X2'], Ts, k, NPTS))
                    rmsep = float(np.sqrt(np.mean((yh - dd['Y'].ravel())**2)))
                    vals.append(rmsep)
                    curves.append(dict(model=name, method=meth, regime=tag, k=k,
                                       RMSEP=rmsep,
                                       R2p=1 - len(dd['Y'])*rmsep**2/SST))
                rec[(meth, tag)] = np.array(vals)

        so_w, so_e = rec[('SO-PLS', 'within')], rec[('SO-PLS', 'extrap')]
        mb_w, mb_e = rec[('MB-PLS', 'within')], rec[('MB-PLS', 'extrap')]
        rows.append(dict(model=name, LV=f'({a},{b},{c})',
                         LV_argmin=f'({imin[0]+1},{imin[1]+1},{imin[2]+1})',
                         RMSECV=rmin, SE=se,
                         n_1SE=int(mask.sum()), Q2_cal=float(q[imin]),
                         SS_X1=ss['SS_X1'], SS_M2=ss['SS_M2'], SS_X2=ss['SS_X2'],
                         SS_F=ss['SS_F'], M2M_point=ss['SS_M2']/ss['SS_X2'],
                         LV_offline=f'({aB},{bB},{cB})',
                         M2M_off_point=mdlB['ss']['SS_M2']/mdlB['ss']['SS_X2'],
                         Q2_cal_offline=float(qB[pickB]),
                         RMSECV_offline=float(rfB.mean(0)[pickB]), MB_LV=mb_lv,
                         SO_within_mean=so_w.mean(), SO_extrap_mean=so_e.mean(),
                         SO_pct_increase=100*(so_e.mean()-so_w.mean())/so_w.mean(),
                         SO_within_k1=so_w[0], SO_extrap_k1=so_e[0],
                         SO_within_kK=so_w[-1], SO_extrap_kK=so_e[-1],
                         MB_within_mean=mb_w.mean(), MB_extrap_mean=mb_e.mean(),
                         MB_pct_increase=100*(mb_e.mean()-mb_w.mean())/mb_w.mean()))
        r = rows[-1]
        print(f'{name:14s} LV={r["LV"]:10s} RMSECV={rmin:.4f} n1SE={int(mask.sum()):4d}  '
              f'M2M pt={r["M2M_point"]:6.2f} | SO within={so_w.mean():.4f} '
              f'extrap={so_e.mean():.4f} (+{r["SO_pct_increase"]:5.1f}%)', flush=True)

    df, cv_df = pd.DataFrame(rows), pd.DataFrame(curves)
    with pd.ExcelWriter('case_study_1_rolling.xlsx') as w:
        df.to_excel(w, sheet_name='summary', index=False)
        cv_df.to_excel(w, sheet_name='rmsep_by_k', index=False)
    print('\n=== Spearman: calibration diagnostic vs extrapolation (n=7) ===')
    for dc in ['M2M_point', 'M2M_off_point']:
        for oc in ['SO_extrap_mean', 'SO_pct_increase']:
            rho, pv = spearmanr(df[dc], df[oc])
            print(f'  rho({dc:14s}, {oc:16s}) = {rho:+.3f}  (p={pv:.3f})')
    print(f'\nSaved case_study_1_rolling.xlsx  ({time.time()-t0:.0f}s)')


STAGES = {'commonality': stage_commonality, 'flatregion': stage_flatregion,
          'q2cost': stage_q2cost, 'rolling': stage_rolling}

if __name__ == '__main__':
    want = sys.argv[1:] or ['commonality', 'flatregion', 'q2cost']  # rolling on request
    for s in want:
        if s not in STAGES:
            raise SystemExit(f'unknown stage {s!r}; choose from {list(STAGES)}')
    for s in want:
        STAGES[s](); print()
