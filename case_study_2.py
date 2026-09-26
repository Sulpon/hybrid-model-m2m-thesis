"""
Case study 2 -- urethane semi-batch reactor. Complete pipeline in one file.

    python case_study_2.py               # everything
    python case_study_2.py replicate     # reproduce the authors' mole profiles
    python case_study_2.py compare       # published vs reconstruction + noise test
    python case_study_2.py feeds         # re-fit the feed programme (slow, optional)
    python case_study_2.py m2m           # the M2M comparison, both block layouts

WHAT IS TAKEN FROM THE ORIGINAL AUTHORS
The process model, its eight kinetic parameters, the measurement-noise levels,
the material balances and the density volume closure are the authors' own, read
out of the notebooks they distributed. The only substitution is the integrator:
they call GEKKO in IMODE=4, which is replaced here by scipy. That is exact --
their DAE is index-1 with an explicit algebraic part, so the three balances
solve directly for nA, nB and nS and there is nothing for an implicit solver to
do that substitution does not already do.

WHAT HAD TO BE RECONSTRUCTED
Their input workbook is not in the archive, so the charge and the feed
programme were recovered from the figures stored inside the notebooks. Each
recovered number is annotated with its source below. The reconstruction is
validated, not assumed: see the `compare` stage, which reproduces their
*individual* noise realisation (z = 11.1 against alternative seeds).

WHAT IS OURS
The block layout and the forty sampled batches. Neither source paper uses
multiblock regression at all -- their method is symbolic regression on
numerical derivatives -- so there is no block design to inherit. The layout
here follows the M2M paper by analogy, including its kappa rule.
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
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# ===========================================================================
#  1. THE AUTHORS' MODEL  (verbatim from their notebooks)
# ===========================================================================
R = 0.008314            # kJ/(mol K)
T_REF = 363.16          # K

PARAM = {'kref1': 1.25e-3, 'kref2': 7.29e-6, 'kref4': 8.8e-7,     # L/(mol h)
         'Ea1': 29.440, 'Ea2': 71.014, 'Ea4': 23.020,             # kJ/mol
         'kc2': 0.217, 'dh': -18.300}
MEAS_ERR = {'C': 5e-3, 'D': 5e-5, 'E': 5e-6}                      # mol
MOL_MASS = {'A': 0.11911, 'B': 0.07412, 'C': 0.19323,
            'D': 0.31234, 'E': 0.35733, 'S': 0.07806}             # kg/mol
RHO = {'A': 1095.0, 'B': 809.0, 'C': 1415.0,
       'D': 1528.0, 'E': 1451.0, 'S': 1101.0}                     # kg/m3

# The authors hard-code k3. It is exactly kref2/kc2 with Ea2 - dh, so the
# reverse step is thermodynamically consistent rather than independent.
K3REF, EA3 = 3.35945e-5, 89.314
assert abs(K3REF - PARAM['kref2']/PARAM['kc2']) < 1e-10
assert abs(EA3 - (PARAM['Ea2'] - PARAM['dh'])) < 1e-9

TRUE8 = np.array([PARAM['kref1'], PARAM['kref2'], PARAM['kref4'],
                  PARAM['Ea1'], PARAM['Ea2'], PARAM['Ea4'],
                  PARAM['kc2'], -PARAM['dh']])
TGRID = np.arange(0.0, 80.0 + 1e-9, 0.5)          # their 161-sample grid


def closure(nC, nD, nE, fv1, fv2, ch):
    """The authors' three algebraic balances plus the density volume."""
    nA = ch['nA0'] + fv1*ch['nAv1'] - nC - 2.0*nD - 3.0*nE
    nB = ch['nB0'] + fv2*ch['nBv2'] - nC - nD
    nS = ch['nS0'] + fv1*ch['nSv1'] + fv2*ch['nSv2']
    n = dict(A=nA, B=nB, C=nC, D=nD, E=nE, S=nS)
    return n, sum(n[s]*MOL_MASS[s]/RHO[s] for s in 'ABCDES')


def make_rhs(reversible=True, th=None):
    """The authors' rate laws. th=None uses their published parameters."""
    p = TRUE8 if th is None else th

    def f(t, y, inp, ch, _unused=None):
        nC, nD, nE = y
        fv1, fv2, T = inp(t)
        n, V = closure(nC, nD, nE, fv1, fv2, ch)
        if V <= 0:
            return [0.0, 0.0, 0.0]
        E = lambda Ea: np.exp((-Ea/R)*(1.0/T - 1.0/T_REF))
        k1, k2, k4 = p[0]*E(p[3]), p[1]*E(p[4]), p[2]*E(p[5])
        if reversible:
            Kc = p[6]*np.exp((p[7]/R)*(1.0/T - 1.0/T_REF))
            k3 = k2/Kc if Kc > 1e-30 else 0.0
        else:
            k3 = 0.0
        nA, nB = max(n['A'], 0.0), max(n['B'], 0.0)
        r1 = k1*nA*nB/V**2
        r2 = k2*nA*max(nC, 0.0)/V**2
        r3 = k3*max(nD, 0.0)/V
        r4 = k4*(nA/V)**2
        return [V*(r1 - r2 + r3), V*(r2 - r3), V*r4]
    return f


def simulate(rhs, inp, ch, t_eval=None, full=False):
    t_eval = TGRID if t_eval is None else t_eval
    s = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [0.0, 0.0, 0.0], t_eval=t_eval,
                  args=(inp, ch), method='LSODA', rtol=1e-10, atol=1e-14)
    if not s.success or s.y.shape[1] != len(t_eval):
        return None if full else np.full((len(t_eval), 3), np.nan)
    y = s.y.T
    if not full:
        return y
    out = {k: np.zeros(len(t_eval)) for k in 'ABCDES'}
    out['V'] = np.zeros(len(t_eval)); out['T'] = np.zeros(len(t_eval))
    for j, tt in enumerate(t_eval):
        fv1, fv2, T = inp(tt)
        n, V = closure(*y[j], fv1, fv2, ch)
        for k in 'ABCDES':
            out[k][j] = n[k]
        out['V'][j], out['T'][j] = V, T
    return out


def add_noise(out, seed=42):
    """The authors seed 42 and perturb only C, D and E, in that order."""
    rng = np.random.RandomState(seed)
    m = {k: (out[k].copy() if k in out else None) for k in out}
    for s in 'CDE':
        m[s] = out[s] + rng.normal(0, MEAS_ERR[s], len(out[s]))
    return m


def make_inputs(spec):
    def f(t):
        return (float(np.interp(t, spec['fv1_t'], spec['fv1_v'])),
                float(np.interp(t, spec['fv2_t'], spec['fv2_v'])),
                float(np.interp(t, spec['T_t'], spec['T_K'])))
    return f


# ===========================================================================
#  2. BATCH_1, RECONSTRUCTED FROM THE AUTHORS' STORED FIGURES
# ===========================================================================
#   T(t)        their "Temperature (T)" panel: 296 K -> 500 K over 0-8 h,
#               hold to 48 h, -> 455 K over 48-56 h, hold to 80 h
#   nA0, nB0    t = 0 of the nA and nB panels
#   V0          t = 0 of the Volume panel
#   nAv1        A balance at t = 80 h (fv1 = 1):  nA + nC + 2nD + 3nE - nA0
#   nBv2        B balance at t = 80 h (fv2 = 1):  nB + nC + nD - nB0
#   nS0         solved from V0 = sum n_i M_i / rho_i at t = 0
#   nSv1        from the shortfall in the Volume panel once all else is fixed:
#               1.446e-5 m3 / (M_S/rho_S) = 0.204 mol, and it scales with fv1,
#               so it belongs to the FIRST feed
#   nSv2        0: no residual volume left for the second feed
#   fv1, fv2    never plotted anywhere; fitted by the `feeds` stage below
def _nS0_from_V0(V0, nA0, nB0):
    return (V0 - nA0*MOL_MASS['A']/RHO['A']
               - nB0*MOL_MASS['B']/RHO['B'])/(MOL_MASS['S']/RHO['S'])

BATCH1_CHARGE = dict(nA0=0.028, nB0=0.170, nAv1=0.5495, nBv2=0.207,
                     nSv1=0.204, nSv2=0.0, V0=2.5e-5)
BATCH1_CHARGE['nS0'] = _nS0_from_V0(2.5e-5, 0.028, 0.170)

BATCH1_INPUTS = dict(
    name='Batch_1',
    T_t=[0.0, 8.0, 48.0, 56.0, 80.0], T_K=[296.0, 500.0, 500.0, 455.0, 455.0],
    fv1_t=[0.0, 1.5, 3.0, 5.0, 7.0, 50.0, 57.0, 80.0],
    fv1_v=[0.0, 0.2339, 0.3741, 0.5659, 0.8762, 0.8762, 1.0, 1.0],
    fv2_t=[0.0, 5.0, 10.0, 22.0, 30.0, 45.0, 50.0, 57.0, 80.0],
    fv2_v=[0.0, 0.6222, 0.6661, 0.6960, 0.7308, 0.8014, 0.8014, 1.0, 1.0])

# anchors digitised from their "Baseline Simulation Results" figure
ANCHORS = {
    'A': [(0, 0.028), (7, 0.046), (10, 0.003), (20, 0.0030), (80, 0.0015)],
    'B': [(0, 0.170), (10, 0.002), (55, 0.010), (80, 0.001)],
    'C': [(5, 0.230), (10, 0.115), (20, 0.125), (40, 0.160), (80, 0.180)],
    'D': [(8, 0.207), (25, 0.204), (45, 0.182), (57, 0.205), (80, 0.197)],
    'E': [(10, 3.20e-4), (80, 3.40e-4)],
    'V': [(0, 2.50e-5), (8, 7.80e-5), (48, 7.90e-5), (80, 8.60e-5)],
}


# ===========================================================================
#  3. RECOVERING THE PUBLISHED CURVES FROM THEIR STORED PNG
# ===========================================================================
PNG = os.path.join('urethane_authors_tmp', 'out', 'urethane_C_c0_1.png')
C0_RGB, GRID_GREY = np.array([31, 119, 180]), 176
PANELS = [('A', 0, 0, [0.00, 0.01, 0.02, 0.03, 0.04]),
          ('B', 0, 1, [0.0, .025, .05, .075, .10, .125, .150, .175]),
          ('C', 0, 2, [0.00, 0.05, 0.10, 0.15, 0.20]),
          ('D', 1, 0, [0.00, 0.05, 0.10, 0.15, 0.20]),
          ('E', 1, 1, [0.0, 5e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4]),
          ('V', 1, 2, [3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5])]
XTICKS = [0.0, 20.0, 40.0, 60.0, 80.0]


def _runs(idx, gap=3):
    if len(idx) == 0:
        return []
    out, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(int(np.mean(cur))); cur = [v]
    return out + [int(np.mean(cur))]


def _affine(pix, vals):
    A = np.vstack([np.asarray(pix, float), np.ones(len(pix))]).T
    return np.linalg.lstsq(A, np.asarray(vals, float), rcond=None)[0]


def digitise(path=PNG, verbose=False):
    """Recover the six published traces. The figure is a plain matplotlib
    render, so the traces are exactly C0 and the grid exactly (176,176,176)."""
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found -- it lives in the authors\' archive, '
                         'which is excluded from version control.')
    a = np.asarray(Image.open(path).convert('RGB'))
    dark = a.sum(2) < 250
    cols = _runs(np.nonzero(dark.sum(0) > 200)[0])
    rows = _runs(np.nonzero(dark.sum(1) > 200)[0])
    xs = [(cols[i], cols[i+1]) for i in range(0, len(cols)-1, 2)]
    ys = [(rows[i], rows[i+1]) for i in range(0, len(rows)-1, 2)]
    out = {}
    for name, r, c, yticks in PANELS:
        x0, x1 = xs[c]; y0, y1 = ys[r]
        sub = a[y0:y1+1, x0:x1+1]
        g = np.abs(sub.astype(int) - GRID_GREY).max(2) < 12
        gx = _runs(np.nonzero(g.sum(0) > 0.55*g.shape[0])[0])
        gy = _runs(np.nonzero(g.sum(1) > 0.55*g.shape[1])[0])
        if len(gx) != len(XTICKS) or len(gy) != len(yticks):
            raise RuntimeError(f'panel {name}: gridline count mismatch')
        mx, cx = _affine([x0+v for v in gx], XTICKS)
        my, cy = _affine([y0+v for v in gy], list(reversed(yticks)))
        mask = np.abs(sub.astype(int) - C0_RGB).sum(2) < 90
        tt, yy = [], []
        for j in range(mask.shape[1]):
            rr = np.nonzero(mask[:, j])[0]
            if len(rr):
                tt.append(mx*(x0+j) + cx); yy.append(my*(y0+rr.mean()) + cy)
        out[name] = (np.asarray(tt), np.asarray(yy))
        if verbose:
            print(f'  panel {name}: {len(tt)} points')
    return out


# ===========================================================================
#  4. PLS / SO-PLS MACHINERY  (unchanged -- do not alter)
# ===========================================================================
MAX_M, MAX_X = 15, 15
N_SPLITS, CV_SEED = 10, 42


def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n


def _sf(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, np.sqrt(B.shape[1])


def _sa(B, mu, sd, k):
    return (B - mu)/sd/k


def _rs(T, tgt):
    g = np.linalg.pinv(T.T @ T) @ T.T @ tgt
    return tgt - T @ g, g


def cv2(B1, B2, Y, m1, m2):
    """SO-PLS B1 -> B2. Returns the Q2 surface and the per-fold RMSE cube."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=CV_SEED)
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
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:, :, None])**2).sum(0)/sst, rf


def cv1(B, Y, m):
    cv = KFold(N_SPLITS, shuffle=True, random_state=CV_SEED)
    P = np.zeros((len(Y), m))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu, sd, k = _sf(B[tr]); bt, be = _sa(B[tr], mu, sd, k), _sa(B[te], mu, sd, k)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p, T, Q, nf = _fit(bt, Y0, m); Te = p.transform(be)
        for a in range(1, m+1):
            ae = min(a, nf)
            P[te, a-1] = ((Te[:, :ae] @ Q[:, :ae].T)*sdy + muY).ravel()
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y.reshape(-1, 1))**2).sum(0)/sst


def seq_ss(B1, B2, Y, n1, n2):
    """Type-I (sequential) sums of squares, which the conventional M2M uses."""
    mu1, sd1, k1 = _sf(B1); b1 = _sa(B1, mu1, sd1, k1)
    mu2, sd2, k2 = _sf(B2); b20 = _sa(B2, mu2, sd2, k2)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    sst = float(np.sum(ys**2))
    p1, T1, Q1, af = _fit(b1, ys, n1); ae = min(n1, af); T1, Q1 = T1[:, :ae], Q1[:, :ae]
    s1 = float(np.sum((T1 @ Q1.T)**2))
    b2r, _ = _rs(T1, b20); ya, _ = _rs(T1, ys)
    p2, T2v, Q2, bf = _fit(b2r, ya, n2); be = min(n2, bf); T2v, Q2 = T2v[:, :be], Q2[:, :be]
    return s1, float(np.sum((T2v @ Q2.T)**2)), sst - s1, sst


def analyse(Mb, Xb, Y):
    """Parsimony-within-1-SE allocation, then both ratios at that allocation."""
    qF, rf = cv2(Mb, Xb, Y, MAX_M, MAX_X)
    qM, qX = cv1(Mb, Y, MAX_M), cv1(Xb, Y, MAX_X)
    rmse = rf.mean(0)
    i = np.unravel_index(rmse.argmin(), rmse.shape)
    se = float(rf[:, i[0], i[1]].std(ddof=1)/np.sqrt(N_SPLITS))
    mask = rmse <= rmse[i] + se
    idx = np.argwhere(mask); tot = idx.sum(1) + 2
    cand = idx[tot == tot.min()]
    pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])
    a, b = pick[0]+1, pick[1]+1
    ss_m, ss_x, _, sst = seq_ss(Mb, Xb, Y, a, b)
    joint = float(qF[pick])
    uM = joint - float(qX[pick[1]])
    uX = joint - float(qM[pick[0]])
    return dict(LV_M=a, LV_X=b, n1SE=int(mask.sum()), RMSECV=float(rmse[i]),
                Q2_full=joint, Q2_M=float(qM[pick[0]]), Q2_X=float(qX[pick[1]]),
                SS_M=ss_m, SS_X=ss_x, SST=sst,
                M2M_conv=ss_m/ss_x if ss_x > 1e-12 else np.inf,
                uM=uM, uX=uX, shared=joint - uM - uX,
                M2M_mod=uM/uX if uX > 1e-9 else np.inf,
                coverage=(uM + uX)/joint if joint > 1e-9 else np.nan)


# ===========================================================================
#  5. THE TWO CANDIDATE MODELS AND THE SAMPLED BATCHES
# ===========================================================================
N_BATCH, N_FIT, DATA_SEED = 40, 8, 101
STATIC_S = [30.0, 50.0, 70.0]          # V1: shared M/X grid end
ROLL_K = [15.0, 30.0, 60.0]            # V2: decision points
CACHE = 'case_study_2_params.json'
CANDIDATES = {'U0_correct': (True, 8), 'U1_no_reverse': (False, 6)}


def sample_batch(rng):
    """Perturb the validated Batch_1. Scaling every mole number leaves all
    concentrations unchanged (V scales with them), so only RATIOS and the
    feed/temperature programme are varied."""
    j = lambda lo, hi: rng.uniform(lo, hi)
    ch = dict(BATCH1_CHARGE)
    for k, lo, hi in [('nB0', .85, 1.15), ('nAv1', .90, 1.10), ('nBv2', .85, 1.15),
                      ('nS0', .90, 1.10), ('nSv1', .90, 1.10)]:
        ch[k] = BATCH1_CHARGE[k]*j(lo, hi)
    b = BATCH1_INPUTS
    # draw order below is load-bearing: it fixes the random stream, and so the
    # forty batches, against the reported results. Do not reorder.
    t_ramp = 8.0*j(.85, 1.15)
    p1 = rng.uniform(488.0, 505.0)
    t_fall = 48.0*j(.92, 1.08)
    p2 = rng.uniform(445.0, 468.0)
    spec = dict(name='sampled',
                T_t=[0.0, t_ramp, t_fall, t_fall + 8.0, 80.0],
                T_K=[296.0, p1, p1, p2, p2],
                fv1_t=[x*j(.9, 1.1) if 0 < x < 10 else x for x in b['fv1_t']],
                fv1_v=[min(1.0, v*j(.9, 1.1)) if 0 < v < 1 else v for v in b['fv1_v']],
                fv2_t=list(b['fv2_t']),
                fv2_v=[min(1.0, v*j(.85, 1.15)) if 0 < v < 1 else v for v in b['fv2_v']])
    spec['T_K'][2] = spec['T_K'][1]                      # flat hold
    for k in ['fv1_t', 'fv1_v', 'fv2_v']:
        spec[k] = list(np.maximum.accumulate(spec[k]))
    return ch, spec


def generate(n=N_BATCH, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(seed + 1)
    truth = make_rhs(True)
    charges, inps, clean, meas = [], [], [], []
    for _ in range(n):
        ch, spec = sample_batch(rng)
        inp = make_inputs(spec)
        y = simulate(truth, inp, ch)
        if not np.isfinite(y).all():
            continue
        charges.append(ch); inps.append(inp); clean.append(y)
        meas.append(y + np.column_stack([nrng.normal(0, MEAS_ERR[s], len(TGRID))
                                         for s in 'CDE']))
    return charges, inps, np.array(clean), np.array(meas)


def fit_kinetics(reversible, npar, inps, charges, meas):
    """Unweighted least squares on the first N_FIT batches, all three species."""
    scale = np.array([MEAS_ERR[s] for s in 'CDE'])

    def res(z):
        th = np.exp(z)
        if npar == 6:
            th = np.concatenate([th, [PARAM['kc2'], -PARAM['dh']]])
        rhs = make_rhs(reversible, th)
        r = []
        for i in range(N_FIT):
            y = simulate(rhs, inps[i], charges[i])
            if not np.isfinite(y).all():
                return np.full(N_FIT*len(TGRID)*3, 50.0)
            r.append(((y - meas[i])/scale).ravel())
        return np.concatenate(r)

    out = least_squares(res, np.log(TRUE8[:npar]), xtol=1e-10, ftol=1e-10,
                        x_scale='jac', max_nfev=120)
    th = np.exp(out.x)
    if npar == 6:
        th = np.concatenate([th, [PARAM['kc2'], -PARAM['dh']]])
    return th, float(out.cost)


# ===========================================================================
#  STAGES
# ===========================================================================
def stage_replicate():
    print('=' * 72); print('REPLICATE -- the authors\' mole profiles'); print('=' * 72)
    out = simulate(make_rhs(True), make_inputs(BATCH1_INPUTS), BATCH1_CHARGE, full=True)
    meas = add_noise(out, 42)
    print('Batch_1 charge (mol):')
    for k in ['nA0', 'nB0', 'nS0', 'nAv1', 'nBv2', 'nSv1', 'nSv2']:
        print(f'   {k:6s} {BATCH1_CHARGE[k]:10.5f}')
    print(f'   V0     {BATCH1_CHARGE["V0"]:10.3e} m3')

    print(f'\n{"sp":3s} {"t[h]":>5s} {"figure":>11s} {"simulated":>11s} {"rel err":>9s}')
    tot = n = 0
    for sp, pts in ANCHORS.items():
        for tt, target in pts:
            j = int(np.argmin(np.abs(TGRID - tt)))
            v = out[sp][j]; rel = (v - target)/abs(target)
            tot += rel**2; n += 1
            print(f'{sp:3s} {tt:5.0f} {target:11.4g} {v:11.4g} {100*rel:8.1f}%')
    print(f'\nRMS relative error {100*np.sqrt(tot/n):.1f}%')

    fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))
    fig.suptitle('Baseline Simulation Results')
    for a, (title, key, yl) in zip(ax.ravel(),
            [('nA', 'A', 'nA [mol]'), ('nB', 'B', 'nB [mol]'), ('nC', 'C', 'nC [mol]'),
             ('nD', 'D', 'nD [mol]'), ('nE', 'E', 'nE [mol]'), ('Volume', 'V', 'V [m$^3$]')]):
        a.plot(TGRID, meas[key] if key in 'CDE' else out[key], lw=2)
        a.set_title(title); a.set_xlabel('Time [h]'); a.set_ylabel(yl)
        a.grid(alpha=0.4); a.set_xlim(0, 80)
    fig.tight_layout(); fig.savefig('cs2_baseline.png', dpi=200); plt.close(fig)
    print('Saved cs2_baseline.png')


def stage_compare():
    print('=' * 72); print('COMPARE -- published vs reconstruction'); print('=' * 72)
    paper = digitise()
    clean = simulate(make_rhs(True), make_inputs(BATCH1_INPUTS), BATCH1_CHARGE, full=True)
    meas = add_noise(clean, 42)

    names = [('A', '$n_A$ isocyanate'), ('B', '$n_B$ butanol'), ('C', '$n_C$ urethane'),
             ('D', '$n_D$ allophanate'), ('E', '$n_E$ isocyanurate'), ('V', 'reactor volume')]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    fig.suptitle('Urethane Batch_1 - published (HyMech notebooks) vs reconstruction, '
                 'both with measurement noise', fontsize=13)
    print(f'{"species":9s} {"n":>5s} {"RMS abs":>11s} {"RMS %range":>12s} {"max dev":>12s}')
    stats = []
    for ax, (key, title) in zip(axes.ravel(), names):
        tp, yp = paper[key]
        m = (tp >= 0) & (tp <= 80); tp, yp = tp[m], yp[m]
        yo = np.interp(tp, TGRID, meas[key])
        rng_ = yp.max() - yp.min(); rms = float(np.sqrt(np.mean((yo - yp)**2)))
        stats.append(100*rms/rng_)
        print(f'{"n"+key:9s} {len(tp):5d} {rms:11.4g} {100*rms/rng_:11.1f}% '
              f'{np.abs(yo-yp).max():12.4g}')
        ax.plot(tp, yp, '-', color='#1f77b4', lw=2.2, alpha=.9, label='published (digitised)')
        if key in 'CDE':
            ax.plot(TGRID, meas[key], 'o', ms=3, mfc='none', mec='#c0392b', mew=.9,
                    ls='none', label='ours (noisy samples)')
            ax.plot(TGRID, clean[key], '--', color='#c0392b', lw=1.2, alpha=.75,
                    label='ours (noise-free)')
        else:
            ax.plot(TGRID, clean[key], '--', color='#c0392b', lw=1.8, label='ours')
        ax.set_title(title, fontsize=11); ax.set_xlabel('time [h]')
        ax.set_xlim(0, 80); ax.grid(alpha=.3)
    axes[0, 2].legend(fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig('cs2_compare_published.png', dpi=200); plt.close(fig)

    # Did we land on the AUTHORS' noise realisation, or merely the same sigma?
    print('\nnoise-realisation test: our draw vs the published wiggle')
    print(f'{"seed":>6s} {"corr nC":>9s} {"corr nE":>9s} {"mean":>8s}')
    hf = {}
    for key in 'CE':
        tp, yp = paper[key]
        m = (tp >= 0) & (tp <= 80); tp, yp = tp[m], yp[m]
        hf[key] = np.interp(TGRID, tp, yp - savgol_filter(yp, 21, 3))
    rows = []
    for seed in [42] + list(range(25)):
        mm = add_noise(clean, seed)
        cs = []
        for key in 'CE':
            nz = mm[key] - clean[key]
            cs.append(float(np.corrcoef(nz - savgol_filter(nz, 21, 3), hf[key])[0, 1]))
        rows.append((seed, cs[0], cs[1], float(np.mean(cs))))
    for s, a, b, mn in rows[:1] + rows[1:6]:
        print(f'{s:6d} {a:9.3f} {b:9.3f} {mn:8.3f}'
              + ('   <-- the authors\' seed' if s == 42 else ''))
    others = [r[3] for r in rows[1:]]
    z = (rows[0][3] - np.mean(others))/np.std(others)
    print(f'... seed 42 mean corr {rows[0][3]:.3f}; 25 other seeds '
          f'{np.mean(others):+.3f} +/- {np.std(others):.3f}  ->  z = {z:.1f}')
    print(f'\noverall RMS deviation {np.sqrt(np.mean(np.square(stats))):.1f}% of panel range')
    print('Saved cs2_compare_published.png')


def stage_feeds():
    """Re-derive the feed programme baked into BATCH1_INPUTS. Optional."""
    print('=' * 72); print('FEEDS -- recover fv1(t), fv2(t)'); print('=' * 72)
    FV1_T = [0.0, 1.5, 3.0, 5.0, 7.0, 50.0, 57.0, 80.0]
    FV2_T = [0.0, 5.0, 10.0, 22.0, 30.0, 45.0, 50.0, 57.0, 80.0]
    N1, N2, FLOOR, SMOOTH = 4, 6, 2e-3, 0.1
    # Anchors alone do not pin the shape BETWEEN them: unregularised, the fit
    # puts a plateau then a jump into fv1 and nA/nB spike twice. Their nB falls
    # monotonically and their nC has one clean peak, so the early decay is
    # anchored explicitly and the increments are penalised for curvature.
    EXTRA = {'A': [(2, 0.040), (3, 0.036)],
             'B': [(1, 0.115), (2, 0.075), (3, 0.045), (5, 0.012)],
             'C': [(3, 0.135), (7, 0.200), (9, 0.118)]}
    ANCH = {k: sorted(list(v) + EXTRA.get(k, [])) for k, v in ANCHORS.items()}
    NRES = sum(len(v) for v in ANCH.values())
    tg = np.arange(0.0, 80.0 + 1e-9, 0.25)

    def build(z):
        v1 = np.clip(np.cumsum(np.abs(z[:N1])), 0, 1)
        v2 = np.clip(np.cumsum(np.abs(z[N1:N1+N2])), 0, 1)
        return dict(name='fit', T_t=BATCH1_INPUTS['T_t'], T_K=BATCH1_INPUTS['T_K'],
                    fv1_t=FV1_T, fv1_v=[0.0]+list(v1)+[v1[-1], 1.0, 1.0],
                    fv2_t=FV2_T, fv2_v=[0.0]+list(v2)+[1.0, 1.0])

    def residuals(z):
        try:
            out = simulate(make_rhs(True), make_inputs(build(z)), BATCH1_CHARGE, tg, full=True)
        except Exception:
            out = None
        if out is None:
            return np.full(NRES + (N1-2) + (N2-2), 10.0)
        r = []
        for sp, pts in ANCH.items():
            for tt, target in pts:
                j = int(np.argmin(np.abs(tg - tt)))
                r.append((out[sp][j] - target)/max(abs(target), FLOOR))
        r.extend(SMOOTH*np.diff(np.abs(z[:N1]), 2))
        r.extend(SMOOTH*np.diff(np.abs(z[N1:N1+N2]), 2))
        return np.asarray(r)

    z0 = np.array([.30, .25, .20, .16, .45, .12, .10, .08, .05, .03])
    best, bc, rng = None, np.inf, np.random.default_rng(0)
    for trial in range(10):
        z = z0 if trial == 0 else np.abs(z0*rng.uniform(.3, 2.0, len(z0)))
        try:
            r = least_squares(residuals, z, bounds=(0.0, 1.0), xtol=1e-12,
                              ftol=1e-12, x_scale='jac', max_nfev=400)
        except Exception:
            continue
        if r.cost < bc:
            best, bc = r.x, r.cost
    spec = build(best)
    print(f'fitted feed programme (cost {bc:.5f})')
    print('  fv1_v =', [round(float(x), 4) for x in spec['fv1_v']])
    print('  fv2_v =', [round(float(x), 4) for x in spec['fv2_v']])
    print('\n(BATCH1_INPUTS already carries this result; rerun only to re-derive it.)')


def stage_m2m():
    print('=' * 72); print('M2M -- true model U0 vs broken model U1'); print('=' * 72)
    t0 = time.time()
    charges, inps, clean, meas = generate()
    nb = len(inps)
    inputs_full = np.array([[f(tt) for tt in TGRID] for f in inps]
                           ).transpose(0, 2, 1).reshape(nb, -1)
    print(f'{nb} batches x {len(TGRID)} timepoints  ({time.time()-t0:.0f}s)\n')
    print('response signal-to-noise at t = 80 h:')
    for k, nm in enumerate(['nC', 'nD', 'nE']):
        sd = clean[:, -1, k].std(ddof=1)
        print(f'  {nm}: clean sd {sd:10.3e}  sigma {MEAS_ERR["CDE"[k]]:8.1e}  '
              f'SNR {sd/MEAS_ERR["CDE"[k]]:8.1f}')

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    preds = {}
    for name, (rev, npar) in CANDIDATES.items():
        if name not in cache:
            th, cost = fit_kinetics(rev, npar, inps, charges, meas)
            cache[name] = dict(th=[float(v) for v in th], cost=cost)
            json.dump(cache, open(CACHE, 'w'), indent=2)
        th, cost = np.array(cache[name]['th']), cache[name]['cost']
        rhs = make_rhs(rev, th)
        preds[name] = np.array([simulate(rhs, f, c) for f, c in zip(inps, charges)])
        print(f'\n{name}: fit cost {cost:.4g}')
        print('   kref1 %.4g  kref2 %.4g  kref4 %.4g  Ea1 %.4g  Ea2 %.4g  Ea4 %.4g'
              % tuple(th[:6]))

    rows = []
    for rname, ri in [('nC', 0), ('nD', 1)]:
        Y = meas[:, -1, ri].reshape(-1, 1)
        print(f'\n--- response: final {rname} '
              f'({Y.mean():.5f} +/- {Y.std(ddof=1):.5f} mol) ---')
        for name in CANDIDATES:
            P = preds[name]
            # V1 static: M states and X share ONE grid, response outside it.
            for Sg in STATIC_S:
                keep = TGRID <= Sg
                Mb = np.hstack([P[:, keep, :].transpose(0, 2, 1).reshape(nb, -1), inputs_full])
                Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
                r = analyse(Mb, Xb, Y)
                r.update(variant='V1_static', model=name, param=Sg, response=rname)
                rows.append(r)
                print(f'  V1 {name:14s} S={Sg:4.0f}h  LV=({r["LV_M"]:2d},{r["LV_X"]:2d})  '
                      f'uM {r["uM"]:+.4f}  uX {r["uX"]:+.4f}  sh {r["shared"]:+.4f}  '
                      f'M2M* {r["M2M_mod"]:9.3f}  M2M {r["M2M_conv"]:9.3f}', flush=True)
            # V2 rolling: the paper's hybrid block, measured <= k then KD-filled.
            Mb = np.hstack([P.transpose(0, 2, 1).reshape(nb, -1), inputs_full])
            for k in ROLL_K:
                Xk = np.where((TGRID <= k)[None, :, None], meas, P)
                r = analyse(Mb, Xk.transpose(0, 2, 1).reshape(nb, -1), Y)
                r.update(variant='V2_rolling', model=name, param=k, response=rname)
                rows.append(r)
                print(f'  V2 {name:14s} k={k:4.0f}h  LV=({r["LV_M"]:2d},{r["LV_X"]:2d})  '
                      f'uM {r["uM"]:+.4f}  uX {r["uX"]:+.4f}  sh {r["shared"]:+.4f}  '
                      f'M2M* {r["M2M_mod"]:9.3f}  M2M {r["M2M_conv"]:9.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('case_study_2_results.xlsx', index=False)

    print('\n' + '=' * 72)
    print('HEADLINE -- V1 static, shared grid 0-70 h')
    print('=' * 72)
    h = df[(df.variant == 'V1_static') & (df.param == 70.0)].set_index(['response', 'model'])
    print(f'{"resp":5s} {"model":14s} {"Q2 M":>8s} {"Q2 X":>8s} {"uM":>8s} {"uX":>8s} '
          f'{"M2M*":>10s} {"M2M":>10s}')
    for resp in ['nC', 'nD']:
        for mod in CANDIDATES:
            r = h.loc[(resp, mod)]
            print(f'{resp:5s} {mod:14s} {r.Q2_M:8.4f} {r.Q2_X:8.4f} {r.uM:+8.4f} '
                  f'{r.uX:+8.4f} {r.M2M_mod:10.3f} {r.M2M_conv:10.2f}')
    print('\nThe modified metric puts U0 above 1 and U1 below it for both responses;')
    print('the conventional ratio places both above 1, so it ranks but does not judge.')
    print('The SIDE of 1 is robust; the magnitude is not, because uX is small.')
    print(f'\nSaved case_study_2_results.xlsx  ({time.time()-t0:.0f}s)')


STAGES = {'replicate': stage_replicate, 'compare': stage_compare,
          'feeds': stage_feeds, 'm2m': stage_m2m}

if __name__ == '__main__':
    want = sys.argv[1:] or ['replicate', 'compare', 'm2m']   # `feeds` only on request
    for s in want:
        if s not in STAGES:
            raise SystemExit(f'unknown stage {s!r}; choose from {list(STAGES)}')
    for s in want:
        STAGES[s](); print()
