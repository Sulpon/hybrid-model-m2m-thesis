"""
Modified M2M on the urethane case study, rebuilt on the authors' own model.

This supersedes urethane_m2m.py / urethane_two_models.py, which were built
before "Urethane case study.zip" was available and therefore differed from
the authors' code in two ways that matter:

  * sigma_C was 5e-2; the authors' notebooks use 5e-3. The old value is 10x
    too noisy and is why n_C had to be abandoned as a response (SNR < 1).
  * the charge and input programme were Bauer Experiment #1 plus a reading of
    HyMech Fig. 7(c). They are now the Batch_1 reconstruction validated in
    urethane_compare_figure.py (5.0% RMS against the published curves, and
    the noise realisation confirmed at z = 11.1).

Everything else -- the SO-PLS routines, the 1-SE parsimony rule, the block
layout and both ratio definitions -- is reused unchanged from urethane_m2m.

The two candidates are the two the source papers actually define:

    U0  correct structure    A + B -> C,  A + C <-> D,  3A -> E   (8 params)
    U1  available FP model   A + B -> C,  A + C  -> D,  3A -> E   (6 params)

U1 is exactly HyMech's misspecification: the reverse step r3 is the term their
symbolic regression is meant to rediscover.

The batches are NOT the authors' 26 -- their workbook is not in the archive.
They are drawn by perturbing the validated Batch_1 programme; the sampling is
in sample_batch() and is ours, not theirs. Note that scaling the whole charge
leaves every concentration unchanged (V scales with it), so only ratios and
the feed/temperature programme are varied.
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import warnings, json, os, time
warnings.filterwarnings('ignore')
import urethane_authors_replicate as A
import urethane_m2m as M          # SO-PLS / CV machinery, reused unchanged

N_BATCH = 40
N_FIT = 8                  # batches used for kinetic estimation
HORIZONS = [15.0, 30.0, 60.0]
SEED = 101
TGRID = np.arange(0.0, 80.0 + 1e-9, 0.5)
CACHE = 'urethane_m2m_corrected_params.json'

TRUE8 = np.array([A.param['kref1'], A.param['kref2'], A.param['kref4'],
                  A.param['Ea1'], A.param['Ea2'], A.param['Ea4'],
                  A.param['kc2'], -A.param['dh']])


# ------------------------------------------------------------------ sampling
def sample_batch(rng, nominal=False):
    """Perturb the validated Batch_1 charge and programme."""
    j = (lambda lo, hi: 1.0 if nominal else rng.uniform(lo, hi))
    ch = dict(A.BATCH1_CHARGE)
    # ratios only: a pure scale-up of every mole number leaves concentrations fixed
    ch['nB0'] = A.BATCH1_CHARGE['nB0']*j(0.85, 1.15)
    ch['nAv1'] = A.BATCH1_CHARGE['nAv1']*j(0.90, 1.10)
    ch['nBv2'] = A.BATCH1_CHARGE['nBv2']*j(0.85, 1.15)
    ch['nS0'] = A.BATCH1_CHARGE['nS0']*j(0.90, 1.10)
    ch['nSv1'] = A.BATCH1_CHARGE['nSv1']*j(0.90, 1.10)

    b = A.BATCH1_INPUTS
    t_ramp = 8.0*j(0.85, 1.15)
    p1 = 500.0 if nominal else rng.uniform(488.0, 505.0)
    t_fall = 48.0*j(0.92, 1.08)
    p2 = 455.0 if nominal else rng.uniform(445.0, 468.0)
    spec = dict(
        name='sampled',
        T_t=[0.0, t_ramp, t_fall, t_fall + 8.0, 80.0],
        T_K=[296.0, p1, p1, p2, p2],
        fv1_t=[x*j(0.9, 1.1) if 0 < x < 10 else x for x in b['fv1_t']],
        fv1_v=[min(1.0, v*j(0.9, 1.1)) if 0 < v < 1 else v for v in b['fv1_v']],
        fv2_t=list(b['fv2_t']),
        fv2_v=[min(1.0, v*j(0.85, 1.15)) if 0 < v < 1 else v for v in b['fv2_v']])
    spec['fv1_t'] = list(np.maximum.accumulate(spec['fv1_t']))
    spec['fv1_v'] = list(np.maximum.accumulate(spec['fv1_v']))
    spec['fv2_v'] = list(np.maximum.accumulate(spec['fv2_v']))
    return ch, spec


# ------------------------------------------------------------- the two models
def rhs_factory(reversible):
    def f(t, y, inp, ch, th):
        nC, nD, nE = y
        fv1, fv2, T = inp(t)
        n, V = A.closure(nC, nD, nE, fv1, fv2, ch)
        if V <= 0:
            return [0.0, 0.0, 0.0]
        E = lambda Ea: np.exp((-Ea/A.R)*(1.0/T - 1.0/A.t_ref))
        k1 = th[0]*E(th[3]); k2 = th[1]*E(th[4]); k4 = th[2]*E(th[5])
        if reversible:
            Kc = th[6]*np.exp((th[7]/A.R)*(1.0/T - 1.0/A.t_ref))
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

CANDIDATES = {'U0_correct': (rhs_factory(True), 8),
              'U1_no_reverse': (rhs_factory(False), 6)}


def sim(rhs, th, inp, ch):
    s = solve_ivp(rhs, (0.0, 80.0), [0.0, 0.0, 0.0], t_eval=TGRID,
                  args=(inp, ch, th), method='LSODA', rtol=1e-9, atol=1e-13)
    if not s.success or s.y.shape[1] != len(TGRID):
        return np.full((len(TGRID), 3), np.nan)
    return s.y.T


def generate(n, seed):
    rng = np.random.default_rng(seed)
    charges, specs, inps, clean, meas = [], [], [], [], []
    truth = rhs_factory(True)
    nrng = np.random.default_rng(seed + 1)
    for i in range(n):
        ch, spec = sample_batch(rng)
        inp = A.make_inputs(spec)
        y = sim(truth, TRUE8, inp, ch)
        if not np.isfinite(y).all():
            continue
        charges.append(ch); specs.append(spec); inps.append(inp)
        clean.append(y)
        noise = np.column_stack([nrng.normal(0, A.meas_err_std[s], len(TGRID))
                                 for s in 'CDE'])
        meas.append(y + noise)
    return charges, specs, inps, np.array(clean), np.array(meas)


def fit(rhs, npar, inps, charges, meas):
    """Unweighted least squares on the first N_FIT batches, all three species."""
    scale = np.array([A.meas_err_std[s] for s in 'CDE'])
    z0 = np.log(TRUE8[:npar])

    def res(z):
        th = np.exp(z)
        if npar == 6:
            th = np.concatenate([th, [A.param['kc2'], -A.param['dh']]])
        r = []
        for i in range(N_FIT):
            y = sim(rhs, th, inps[i], charges[i])
            if not np.isfinite(y).all():
                return np.full(N_FIT*len(TGRID)*3, 50.0)
            r.append(((y - meas[i])/scale).ravel())
        return np.concatenate(r)

    out = least_squares(res, z0, xtol=1e-10, ftol=1e-10, x_scale='jac', max_nfev=120)
    th = np.exp(out.x)
    if npar == 6:
        th = np.concatenate([th, [A.param['kc2'], -A.param['dh']]])
    return th, float(out.cost)


# ------------------------------------------------------------------- analysis
def analyse(Mb, Xb, Y):
    qF, rf = M.cv2(Mb, Xb, Y, M.MAX_M, M.MAX_X)
    qM = M.cv1(Mb, Y, M.MAX_M)
    qX = M.cv1(Xb, Y, M.MAX_X)
    rmse = rf.mean(0)
    i = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[i]); se = float(rf[:, i[0], i[1]].std(ddof=1)/np.sqrt(M.N_SPLITS))
    mask = rmse <= rmin + se
    idx = np.argwhere(mask); tot = idx.sum(1) + 2
    cand = idx[tot == tot.min()]
    pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])
    a, b = pick[0] + 1, pick[1] + 1
    ss_m, ss_x, _, sst = M.seq_ss(Mb, Xb, Y, a, b)
    uM = float(qF[pick] - qX[pick[1]])
    uX = float(qF[pick] - qM[pick[0]])
    joint = float(qF[pick])
    return dict(LV_M=a, LV_X=b, n1SE=int(mask.sum()), RMSECV=rmin,
                Q2_full=joint, Q2_M=float(qM[pick[0]]), Q2_X=float(qX[pick[1]]),
                SS_M=ss_m, SS_X=ss_x, SST=sst,
                M2M_conv=ss_m/ss_x if ss_x > 1e-12 else np.inf,
                uM=uM, uX=uX, shared=joint - uM - uX,
                M2M_mod=uM/uX if uX > 1e-9 else np.inf,
                coverage=(uM + uX)/joint if joint > 1e-9 else np.nan)


if __name__ == '__main__':
    t0 = time.time()
    charges, specs, inps, clean, meas = generate(N_BATCH, SEED)
    nb = len(inps)
    print(f'{nb} batches x {len(TGRID)} timepoints  ({time.time()-t0:.0f}s)')

    print('\nresponse signal-to-noise at t = 80 h (final value):')
    for k, name in enumerate(['nC', 'nD', 'nE']):
        sd = clean[:, -1, k].std(ddof=1)
        sig = A.meas_err_std['CDE'[k]]
        print(f'  {name}: clean sd {sd:10.3e}  sigma {sig:8.1e}  SNR {sd/sig:8.1f}')

    resp = int(os.environ.get('RESP_IDX', 0))       # 0 = nC (now usable)
    rname = ['nC', 'nD', 'nE'][resp]
    Y = meas[:, -1, resp].reshape(-1, 1)
    print(f'\nresponse: final {rname}  ({Y.mean():.5f} +/- {Y.std(ddof=1):.5f} mol)')

    inputs_unf = np.array([[f(tt) for tt in TGRID] for f in inps]
                          ).transpose(0, 2, 1).reshape(nb, -1)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    rows = []
    for name, (rhs, npar) in CANDIDATES.items():
        if name in cache:
            th, cost = np.array(cache[name]['th']), cache[name]['cost']
        else:
            th, cost = fit(rhs, npar, inps, charges, meas)
            cache[name] = dict(th=[float(v) for v in th], cost=cost)
            json.dump(cache, open(CACHE, 'w'), indent=2)
        print(f'\n{name}: fit cost {cost:.4g}  ({time.time()-t0:.0f}s)')
        print('   kref1 %.4g  kref2 %.4g  kref4 %.4g  Ea1 %.4g  Ea2 %.4g  Ea4 %.4g'
              % tuple(th[:6]))

        pred = np.array([sim(rhs, th, f, c) for f, c in zip(inps, charges)])
        Mb = np.hstack([pred.transpose(0, 2, 1).reshape(nb, -1), inputs_unf])
        for H in HORIZONS:
            keep = TGRID <= H
            Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
            r = analyse(Mb, Xb, Y)
            r.update(model=name, H=H, fit_cost=cost, response=rname)
            rows.append(r)
            print(f'   H={H:4.0f}h  LV=({r["LV_M"]},{r["LV_X"]})  Q2 {r["Q2_full"]:.4f}  '
                  f'uM {r["uM"]:+.4f}  uX {r["uX"]:+.4f}  shared {r["shared"]:+.4f}  '
                  f'M2M* {r["M2M_mod"]:8.3f}   M2M {r["M2M_conv"]:8.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('urethane_m2m_corrected.xlsx', index=False)
    print('\n' + df[['model', 'H', 'LV_M', 'LV_X', 'Q2_full', 'Q2_M', 'Q2_X',
                     'uM', 'uX', 'shared', 'coverage', 'M2M_mod', 'M2M_conv']]
          .round(4).to_string(index=False))
    print(f'\nSaved urethane_m2m_corrected.xlsx  ({time.time()-t0:.0f}s)')
