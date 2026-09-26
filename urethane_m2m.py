"""
Unique-M2M analysis on case study 2 (urethane).

BLOCK STRUCTURE
  M   mechanistic block: FP-model-predicted nC, nD, nE over the FULL batch,
      plus the KD inputs fv1, fv2, T (moved into M by the causal-ordering rule).
  X   measured block: measured nC, nD, nE up to a DECISION POINT only.
  Y   final nC at the end of the batch.

Why the decision point. Y is the measured final nC, which is literally a column
of the full measured trajectory -- with an untruncated X the problem is trivial
and Q2 -> 1. Truncating X to t <= horizon while letting M span the whole batch
is the honest construction and is exactly the asymmetry the methodology is about:
a mechanistic model can forecast past the last measurement, a measured block
cannot. This is the single-decision-point version of the rolling procedure.

CANDIDATE MECHANISMS (the M0-M6 analogue)
  U0_reversible  correct structure (A+C <-> D), Kc2 and dH held at nominal
  U1_irrev       second reaction irreversible          <- the papers' own FP model
  U2_no_E        3A -> E removed
  U3_order1_A    r4 = k4 cA          instead of k4 cA^2
  U4_lumped      r4 driven by k2     (E and D share one rate constant)
  U5_const_V     volume frozen at its initial value
  U6_isothermal  rate constants frozen at Tref (no temperature dependence)

Every candidate gets the same six kinetic parameters (kref1, kref2, kref4,
Ea1, Ea2, Ea4) fitted to the calibration trajectories by UNWEIGHTED least
squares -- the weighting shown to reproduce Geremia et al. in
urethane_replicate.py.

METRIC, as established on case study 1
  ratio_Q2 = (Q2_full - Q2_Xonly) / (Q2_full - Q2_Monly)
           = unique(M) / unique(X)
  reported as the median over the 1-SE flat region, with the paper's
  M2M = SS_M/SS_X alongside for comparison.
Conventions: KFold(10, shuffle, seed 42), block scaling 1/sqrt(K), RMSECV =
mean of per-fold RMSE, Q2 = 1 - pooled PRESS/SST, SS on standardised Y.
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import warnings, time, json
warnings.filterwarnings('ignore')
import urethane_case_study as U

MAX_M, MAX_X = 15, 15
N_SPLITS, SEED = 10, 42
HORIZONS = [15.0, 30.0, 60.0]
DEFAULT_HORIZON = 30.0

# RESPONSE CHOICE. Final nC is unusable: Bauer's sigma_C = 5e-2 mol exceeds the
# batch-to-batch spread of nC itself (clean sd 0.019-0.050 mol), so its
# signal-to-noise ratio is 0.37-0.99 at every time point and Q2 comes out
# negative for every model -- there is nothing there to predict. nD has SNR
# 143-288 and is also the species the structural mismatch acts on, which makes
# it the discriminating response.
RESPONSE = 'nD'          # 'nC' | 'nD' | 'nE'
RESP_IDX = {'nC': 0, 'nD': 1, 'nE': 2}[RESPONSE]
PARAM_CACHE = 'urethane_m2m_params.json'   # fits do not depend on Y; cache them

# ---------------------------------------------------------------- candidates
def _common(t, y, inputs, ic, th, const_V=False, isothermal=False):
    kref1, kref2, kref4, Ea1, Ea2, Ea4 = th
    nC, nD, nE = y
    fv1, fv2, T = inputs(t)
    n, V = U.closure(nC, nD, nE, fv1, fv2, ic)
    if const_V:
        _, V = U.closure(0.0, 0.0, 0.0, 0.0, 0.0, ic)
    if V <= 0:
        return None
    Teff = U.TREF if isothermal else T
    e = lambda Ea: np.exp(-Ea/U.R_GAS*(1.0/Teff - 1.0/U.TREF))
    k1, k2, k4 = kref1*1e-3*e(Ea1), kref2*1e-3*e(Ea2), kref4*1e-3*e(Ea4)
    cA, cB, cC, cD = (max(n[0], 0.0)/V, max(n[1], 0.0)/V,
                      max(nC, 0.0)/V, max(nD, 0.0)/V)
    return V, T, k1, k2, k4, cA, cB, cC, cD

def u0_reversible(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    Kc = U.KC2*1e-3*np.exp(-U.DELTA_H/U.R_GAS*(1.0/T - 1.0/U.TREF))
    k3 = k2/Kc
    r1, r2, r3, r4 = k1*cA*cB, k2*cA*cC, k3*cD, k4*cA*cA
    return [V*(r1 - r2 + r3), V*(r2 - r3), V*r4]

def u1_irrev(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    return [V*(k1*cA*cB - k2*cA*cC), V*k2*cA*cC, V*k4*cA*cA]

def u2_no_E(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    return [V*(k1*cA*cB - k2*cA*cC), V*k2*cA*cC, 0.0]

def u3_order1_A(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    return [V*(k1*cA*cB - k2*cA*cC), V*k2*cA*cC, V*k4*cA]

def u4_lumped(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    return [V*(k1*cA*cB - k2*cA*cC), V*k2*cA*cC, V*k2*cA*cA]

def u5_const_V(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th, const_V=True)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    return [V*(k1*cA*cB - k2*cA*cC), V*k2*cA*cC, V*k4*cA*cA]

def u6_isothermal(t, y, inputs, ic, th):
    r = _common(t, y, inputs, ic, th, isothermal=True)
    if r is None: return [0.0]*3
    V, T, k1, k2, k4, cA, cB, cC, cD = r
    return [V*(k1*cA*cB - k2*cA*cC), V*k2*cA*cC, V*k4*cA*cA]

CANDIDATES = {'U0_reversible': u0_reversible, 'U1_irrev': u1_irrev,
              'U2_no_E': u2_no_E, 'U3_order1_A': u3_order1_A,
              'U4_lumped': u4_lumped, 'U5_const_V': u5_const_V,
              'U6_isothermal': u6_isothermal}

TH0 = np.array([np.log10(U.KREF1), np.log10(U.KREF2), np.log10(U.KREF4),
                U.EA1/1e4, U.EA2/1e4, U.EA4/1e4])
LO = np.array([-6.0, -9.0, -10.0, 0.1, 0.1, 0.1])
HI = np.array([0.0, -2.0, -2.0, 15.0, 15.0, 15.0])
unpack = lambda z: np.array([10**z[0], 10**z[1], 10**z[2], z[3]*1e4, z[4]*1e4, z[5]*1e4])

def sim(rhs, th, inputs, ic, t):
    s = solve_ivp(rhs, (0.0, t[-1]), [0.0, 0.0, 0.0], t_eval=t,
                  args=(inputs, ic, th), method='LSODA', rtol=1e-8, atol=1e-11)
    if not s.success or s.y.shape[1] != len(t):
        return np.full((len(t), 3), np.nan)
    return s.y.T

def fit(rhs, inputs, meas, ic, t):
    def resid(z):
        th = unpack(z); out = []
        for i, f in enumerate(inputs):
            p = sim(rhs, th, f, ic, t)
            if not np.isfinite(p).all():
                return np.full(meas.size, 1e6)
            out.append((p - meas[i]).ravel())
        return np.concatenate(out)
    r = least_squares(resid, TH0, bounds=(LO, HI), xtol=1e-10, ftol=1e-10,
                      x_scale='jac', max_nfev=300)
    return unpack(r.x), float(r.cost)

# ---------------------------------------------------------------- PLS helpers
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

def cv2(B1, B2, Y, m1, m2):
    """SO-PLS B1 -> B2. Returns Q2 surface and per-fold RMSE cube."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m1, m2)); rf = np.zeros((N_SPLITS, m1, m2))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = _sf(B1[tr]); b1t, b1e = _sa(B1[tr],mu1,sd1,k1), _sa(B1[te],mu1,sd1,k1)
        mu2, sd2, k2 = _sf(B2[tr]); b2t0, b2e0 = _sa(B2[tr],mu2,sd2,k2), _sa(B2[te],mu2,sd2,k2)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = _fit(b1t, Y0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
            yh_a = Tae @ Q1[:,:ae].T
            b2t, g2 = _rs(Tat, b2t0); b2e = b2e0 - Tae @ g2
            Ya, _ = _rs(Tat, Y0)
            p2, T2t, Q2, bf = _fit(b2t, Ya, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf)
                yh = ((yh_a + T2e[:,:be] @ Q2[:,:be].T)*sdy + muY).ravel()
                P[te, a-1, b-1] = yh
                rf[f, a-1, b-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y[:,:,None])**2).sum(0)/sst, rf

def cv1(B, Y, m):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    P = np.zeros((len(Y), m))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu, sd, k = _sf(B[tr]); bt, be = _sa(B[tr],mu,sd,k), _sa(B[te],mu,sd,k)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p, T, Q, nf = _fit(bt, Y0, m); Te = p.transform(be)
        for a in range(1, m+1):
            ae = min(a, nf)
            P[te, a-1] = ((Te[:,:ae] @ Q[:,:ae].T)*sdy + muY).ravel()
    sst = np.sum((Y - Y.mean(0))**2)
    return 1 - ((P - Y.reshape(-1, 1))**2).sum(0)/sst

def seq_ss(B1, B2, Y, n1, n2):
    mu1, sd1, k1 = _sf(B1); b1 = _sa(B1, mu1, sd1, k1)
    mu2, sd2, k2 = _sf(B2); b20 = _sa(B2, mu2, sd2, k2)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    sst = float(np.sum(ys**2))
    p1, T1, Q1, af = _fit(b1, ys, n1); ae = min(n1, af); T1, Q1 = T1[:,:ae], Q1[:,:ae]
    s1 = float(np.sum((T1 @ Q1.T)**2))
    b2r, _ = _rs(T1, b20); ya, _ = _rs(T1, ys)
    p2, T2v, Q2, bf = _fit(b2r, ya, n2); be = min(n2, bf); T2v, Q2 = T2v[:,:be], Q2[:,:be]
    s2 = float(np.sum((T2v @ Q2.T)**2))
    return s1, s2, sst - s1 - s2, sst

# ================================================================= run
if __name__ == '__main__':
    t0 = time.time()
    t, inp, meas, clean, _, ic = U.generate(U.N_CALIBRATION, 101)
    Y = meas[:, -1, RESP_IDX].reshape(-1, 1)       # measured final response
    nb, nt = meas.shape[0], len(t)
    inputs_unf = np.array([[f(tt) for tt in t] for f in inp]).transpose(0, 2, 1).reshape(nb, -1)
    csd = clean[:, -1, RESP_IDX].std(ddof=1)
    print(f"{nb} batches, {nt} timepoints, Y = final {RESPONSE} "
          f"{Y.mean():.5f} +/- {Y.std(ddof=1):.5f} mol  (clean sd {csd:.5f}, "
          f"noise {U.NOISE_LEVELS[RESP_IDX]:.1e}, SNR {csd/U.NOISE_LEVELS[RESP_IDX]:.0f})\n",
          flush=True)
    import os
    cache = json.load(open(PARAM_CACHE)) if os.path.exists(PARAM_CACHE) else {}

    rows = []
    for name, rhs in CANDIDATES.items():
        if name in cache:
            th, cost = np.array(cache[name]['th']), cache[name]['cost']
        else:
            th, cost = fit(rhs, inp, meas, ic, t)
            cache[name] = dict(th=[float(v) for v in th], cost=cost)
            json.dump(cache, open(PARAM_CACHE, 'w'), indent=2)
        pred = np.array([sim(rhs, th, f, ic, t) for f in inp])
        M = np.hstack([pred.transpose(0, 2, 1).reshape(nb, -1), inputs_unf])
        rec = dict(model=name, fit_cost=cost,
                   kref1=th[0], kref2=th[1], kref4=th[2], Ea1=th[3], Ea2=th[4], Ea4=th[5])
        for H in HORIZONS:
            keep = t <= H
            X = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
            qF, rf = cv2(M, X, Y, MAX_M, MAX_X)
            qM = cv1(M, Y, MAX_M)
            qX = cv1(X, Y, MAX_X)
            uM = qF - qX[None, :]
            uX = qF - qM[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(uX > 1e-9, uM/uX, np.nan)
            rmse = rf.mean(0)
            i = np.unravel_index(rmse.argmin(), rmse.shape)
            rmin = float(rmse[i]); se = float(rf[:, i[0], i[1]].std(ddof=1)/np.sqrt(N_SPLITS))
            mask = rmse <= rmin + se
            idx = np.argwhere(mask); tot = idx.sum(1) + 2
            pick = tuple(idx[tot == tot.min()][
                np.array([rmse[tuple(u)] for u in idx[tot == tot.min()]]).argmin()])
            ss_m, ss_x, ss_f, sst = seq_ss(M, X, Y, pick[0]+1, pick[1]+1)
            ssr = [seq_ss(M, X, Y, u[0]+1, u[1]+1) for u in idx]
            m2m_flat = float(np.median([a/b for a, b, _, _ in ssr if b > 1e-9]))
            tag = f'H{int(H)}'
            rec.update({
                f'{tag}_LV': f'({pick[0]+1},{pick[1]+1})', f'{tag}_n1SE': int(mask.sum()),
                f'{tag}_RMSECV': rmin, f'{tag}_Q2full': float(qF[pick]),
                f'{tag}_Q2M': float(qM[pick[0]]), f'{tag}_Q2X': float(qX[pick[1]]),
                f'{tag}_uM': float(uM[pick]), f'{tag}_uX': float(uX[pick]),
                f'{tag}_shared': float(qF[pick] - uM[pick] - uX[pick]),
                f'{tag}_ratio_pick': float(ratio[pick]),
                f'{tag}_ratio_flatmed': float(np.nanmedian(ratio[mask])),
                f'{tag}_M2M_pick': ss_m/ss_x if ss_x > 1e-9 else np.nan,
                f'{tag}_M2M_flatmed': m2m_flat})
        rows.append(rec)
        r = rec; d = f'H{int(DEFAULT_HORIZON)}'
        print(f"{name:14s} cost={cost:.4e}  LV={r[d+'_LV']:7s} n1SE={r[d+'_n1SE']:3d}  "
              f"Q2full={r[d+'_Q2full']:.4f} Q2M={r[d+'_Q2M']:.4f} Q2X={r[d+'_Q2X']:.4f}  "
              f"ratio={r[d+'_ratio_flatmed']:8.3f}  M2M={r[d+'_M2M_flatmed']:8.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 340, 'display.max_columns', 90)
    print("\n=== fitted parameters and mechanism quality ===")
    print(df[['model', 'fit_cost', 'kref1', 'kref2', 'kref4', 'Ea1', 'Ea2', 'Ea4']]
          .sort_values('fit_cost').to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    for H in HORIZONS:
        tag = f'H{int(H)}'
        print(f"\n=== decision point t <= {H:.0f} h ===")
        cols = ['model', f'{tag}_LV', f'{tag}_n1SE', f'{tag}_RMSECV', f'{tag}_Q2full',
                f'{tag}_Q2M', f'{tag}_Q2X', f'{tag}_uM', f'{tag}_uX', f'{tag}_shared',
                f'{tag}_ratio_pick', f'{tag}_ratio_flatmed', f'{tag}_M2M_flatmed']
        s = df[cols].copy()
        s.columns = [c.replace(tag+'_', '') for c in cols]
        print(s.sort_values('ratio_flatmed', ascending=False).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print("\n=== Spearman: diagnostic vs mechanism quality (fit cost; lower = better) ===")
    for H in HORIZONS:
        tag = f'H{int(H)}'
        for col in [f'{tag}_ratio_flatmed', f'{tag}_M2M_flatmed', f'{tag}_Q2full']:
            rho, p = spearmanr(df[col], df['fit_cost'])
            print(f"  rho({col:22s}, fit_cost) = {rho:+.3f} (p={p:.3f})")
    df.to_excel('urethane_m2m.xlsx', index=False)
    print(f"\nSaved urethane_m2m.xlsx   ({time.time()-t0:.0f}s)")
