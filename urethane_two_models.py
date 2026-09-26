"""
Urethane case study restricted to the TWO models the source papers actually use.

Neither source paper defines a family of candidate mechanisms. Each has exactly
one available first-principles model -- the second reaction treated as
irreversible -- alongside the ground truth. That is the same design as the M2M
paper's case (i) / case (ii), so the faithful comparison here is:

    U0  correct structure      A + B -> C,  A + C <-> D,  3A -> E
    U1  available FP model     A + B -> C,  A + C  -> D,  3A -> E   (r3 = 0)

Both carry the same six fitted kinetic parameters (kref1, kref2, kref4, Ea1,
Ea2, Ea4), estimated on the calibration batches by unweighted least squares, so
the only difference between them is the missing reverse step.

Reported for each model, following the M2M paper's own presentation:
  - latent-variable allocation (parsimonious member of the 1-SE flat region)
  - R2 and Q2, with the block-wise split
  - CONVENTIONAL M2M  = SS_M / SS_X
  - MODIFIED   M2M  = unique_M / unique_X, with the commonality terms
  - bootstrap log-ratio comparison and the ratio-of-ratios (RoR), as the paper does
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import ttest_ind
import warnings, json, time
warnings.filterwarnings('ignore')
import urethane_case_study as U
import urethane_m2m as M

HORIZONS = [30.0, 60.0]
N_BOOT = 200          # the paper uses 50; more costs nothing here
RESP_IDX = M.RESP_IDX

def blocks(rhs, th, inp, ic, t, meas, H, inputs_unf):
    nb = len(inp)
    pred = np.array([M.sim(rhs, th, f, ic, t) for f in inp])
    Mb = np.hstack([pred.transpose(0, 2, 1).reshape(nb, -1), inputs_unf])
    keep = t <= H
    Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
    return Mb, Xb

def analyse(Mb, Xb, Y):
    """LV selection, contributions, and both ratios at the selected allocation."""
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
    a, b = pick[0]+1, pick[1]+1
    ss_m, ss_x, ss_f, sst = M.seq_ss(Mb, Xb, Y, a, b)
    uM = float(qF[pick] - qX[pick[1]])
    uX = float(qF[pick] - qM[pick[0]])
    return dict(LV=(a, b), n1SE=int(mask.sum()), RMSECV=rmin,
                R2_M=100*ss_m/sst, R2_X=100*ss_x/sst, R2_F=100*ss_f/sst,
                R2_tot=100*(ss_m+ss_x)/sst,
                Q2_full=float(qF[pick]), Q2_M=float(qM[pick[0]]), Q2_X=float(qX[pick[1]]),
                uM=uM, uX=uX, shared=float(qF[pick]) - uM - uX,
                M2M_conv=ss_m/ss_x if ss_x > 1e-12 else np.nan,
                M2M_mod=uM/uX if uX > 1e-12 else np.nan)

def boot_ratios(Mb, Xb, Y, lv, rng, n_boot):
    """Bootstrap batches; recompute both ratios at the FIXED allocation."""
    n = len(Y); a, b = lv
    conv, mod = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        Ms, Xs, Ys = Mb[s], Xb[s], Y[s]
        try:
            ss_m, ss_x, _, _ = M.seq_ss(Ms, Xs, Ys, a, b)
            conv.append(ss_m/ss_x if ss_x > 1e-12 else np.nan)
            qF, _ = M.cv2(Ms, Xs, Ys, a, b)
            qM = M.cv1(Ms, Ys, a); qX = M.cv1(Xs, Ys, b)
            uM = qF[a-1, b-1] - qX[b-1]; uX = qF[a-1, b-1] - qM[a-1]
            mod.append(uM/uX if uX > 1e-12 else np.nan)
        except Exception:
            conv.append(np.nan); mod.append(np.nan)
    return np.array(conv), np.array(mod)

def log_ratio_test(r0, r1, label):
    """Welch t-test on log ratios, back-transformed to a ratio-of-ratios."""
    a = r0[np.isfinite(r0) & (r0 > 0)]
    b = r1[np.isfinite(r1) & (r1 > 0)]
    if len(a) < 5 or len(b) < 5:
        return dict(metric=label, n_U0=len(a), n_U1=len(b), RoR=np.nan,
                    lo=np.nan, hi=np.nan, p=np.nan)
    la, lb = np.log(a), np.log(b)
    t, p = ttest_ind(la, lb, equal_var=False)
    d = la.mean() - lb.mean()
    se = np.sqrt(la.var(ddof=1)/len(la) + lb.var(ddof=1)/len(lb))
    return dict(metric=label, n_U0=len(a), n_U1=len(b), RoR=float(np.exp(d)),
                lo=float(np.exp(d - 1.96*se)), hi=float(np.exp(d + 1.96*se)), p=float(p))

if __name__ == '__main__':
    t0 = time.time()
    import os
    cache = json.load(open(M.PARAM_CACHE)) if os.path.exists(M.PARAM_CACHE) else {}
    t, inp, meas, clean, _, ic = U.generate(U.N_CALIBRATION, 101)
    Y = meas[:, -1, RESP_IDX].reshape(-1, 1)
    nb = len(inp)
    inputs_unf = np.array([[f(tt) for tt in t] for f in inp]).transpose(0, 2, 1).reshape(nb, -1)
    print(f"{nb} batches; Y = final {M.RESPONSE} = {Y.mean():.5f} +/- {Y.std(ddof=1):.5f} mol\n", flush=True)

    MODELS = [('U0_correct', M.u0_reversible), ('U1_FPmodel', M.u1_irrev)]
    rng = np.random.default_rng(11)
    rows, tests = [], []
    for H in HORIZONS:
        res, bts = {}, {}
        for tag, rhs in MODELS:
            key = 'U0_reversible' if tag.startswith('U0') else 'U1_irrev'
            if key not in cache:
                print(f"    fitting {key} ...", flush=True)
                th_, cost_ = M.fit(rhs, inp, meas, ic, t)
                cache[key] = dict(th=[float(v) for v in th_], cost=cost_)
                json.dump(cache, open(M.PARAM_CACHE, 'w'), indent=2)
            th = np.array(cache[key]['th'])
            Mb, Xb = blocks(rhs, th, inp, ic, t, meas, H, inputs_unf)
            r = analyse(Mb, Xb, Y)
            r.update(model=tag, horizon=H, fit_cost=cache[key]['cost'])
            res[tag] = r
            bts[tag] = boot_ratios(Mb, Xb, Y, r['LV'], rng, N_BOOT)
            rows.append(r)
            print(f"  H={H:.0f}h {tag:11s} LV={r['LV']} n1SE={r['n1SE']:3d} "
                  f"Q2full={r['Q2_full']:.4f} uM={r['uM']:+.4f} uX={r['uX']:+.4f} "
                  f"conv={r['M2M_conv']:.3f} mod={r['M2M_mod']}", flush=True)
        for j, lbl in [(0, 'conventional M2M'), (1, 'modified M2M')]:
            d = log_ratio_test(bts['U0_correct'][j], bts['U1_FPmodel'][j], lbl)
            d['horizon'] = H
            tests.append(d)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    for H in HORIZONS:
        s = df[df.horizon == H]
        print(f"\n=== decision point t <= {H:.0f} h ===")
        print("\n-- explained variance (in-sample R2 split and cross-validated Q2) --")
        print(s[['model', 'LV', 'n1SE', 'R2_M', 'R2_X', 'R2_F', 'R2_tot',
                 'Q2_full', 'Q2_M', 'Q2_X']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print("\n-- commonality split of the cross-validated Q2 --")
        print(s[['model', 'uM', 'uX', 'shared', 'Q2_full']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
        print("\n-- the two ratios --")
        print(s[['model', 'M2M_conv', 'M2M_mod']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    tt = pd.DataFrame(tests)
    print("\n=== bootstrap log-ratio comparison, U0 vs U1 "
          f"({N_BOOT} replicates; paper uses 50) ===")
    print(tt[['horizon', 'metric', 'n_U0', 'n_U1', 'RoR', 'lo', 'hi', 'p']]
          .to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print("\nRoR > 1 means the correct model relies proportionally more on the "
          "mechanistic block than the misspecified one, which is the expected direction.")

    with pd.ExcelWriter('urethane_two_models.xlsx') as w:
        df.to_excel(w, sheet_name='summary', index=False)
        tt.to_excel(w, sheet_name='bootstrap_test', index=False)
    print(f"\nSaved urethane_two_models.xlsx   ({time.time()-t0:.0f}s)")
