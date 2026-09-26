"""
Urethane M2M under two block layouts that follow the M2M paper, not my own.

urethane_m2m_corrected.py truncated the measured block at a horizon H while
the mechanistic block kept all 161 timepoints. The M2M paper never does that.
It truncates X2 only to tau = min{t2_i}, purely to equalise trajectory lengths
across batches, and for rolling prediction it FILLS the future rather than
cutting it:

    "For decision point k, form X2^(k) by combining observed trajectories for
     t <= k (from X2) with KD predictions for t > k (from M2)."

That truncation is what manufactured uX -> 0, so both paper-faithful layouts
are rebuilt here:

  V1  static.  M states and X share ONE grid 0..S, exactly as the paper's M2
      and X2 share their 7 sample times, and the response sits outside it.
      S in {30, 50, 70} h, Y measured at 80 h.

  V2  rolling. The paper's hybrid block. X^(k) is always full width: measured
      for t <= k, KD-predicted by the candidate for t > k. k in {15, 30, 60} h.

The input programme stays in M in both, which is the paper's kappa rule --
"an augmented mechanistic block is formed by appending the selected columns of
X to M and removing them from X" -- since fv1, fv2 and T are inputs to M.

cv2/cv1/seq_ss are reused unchanged from urethane_m2m; its _sf already guards
near-constant columns (sd < 1e-12 -> 1), which is what the KD-filled columns
of X^(k) can approach.
"""
import numpy as np
import pandas as pd
import warnings, json, os, time
warnings.filterwarnings('ignore')
import urethane_authors_replicate as A
import urethane_m2m_corrected as C
import urethane_m2m as M

STATIC_S = [30.0, 50.0, 70.0]     # V1: shared M/X grid end
ROLL_K = [15.0, 30.0, 60.0]       # V2: decision points
RESP = {'nC': 0, 'nD': 1}
TG = C.TGRID


def main():
    t0 = time.time()
    charges, specs, inps, clean, meas = C.generate(C.N_BATCH, C.SEED)
    nb = len(inps)
    inputs_full = np.array([[f(tt) for tt in TG] for f in inps]
                           ).transpose(0, 2, 1).reshape(nb, -1)
    cache = json.load(open(C.CACHE)) if os.path.exists(C.CACHE) else {}
    print(f'{nb} batches, {len(TG)} timepoints  ({time.time()-t0:.0f}s)')

    preds = {}
    for name, (rhs, npar) in C.CANDIDATES.items():
        if name not in cache:
            raise SystemExit(f'no cached fit for {name}; run urethane_m2m_corrected first')
        th = np.array(cache[name]['th'])
        preds[name] = np.array([C.sim(rhs, th, f, c) for f, c in zip(inps, charges)])
        print(f'  {name}: fit cost {cache[name]["cost"]:.4g}')

    rows = []
    for rname, ri in RESP.items():
        Y = meas[:, -1, ri].reshape(-1, 1)
        print(f'\n===== response: final {rname} '
              f'({Y.mean():.5f} +/- {Y.std(ddof=1):.5f} mol) =====')

        for name in C.CANDIDATES:
            P = preds[name]

            # ---------------------------------------------- V1 static layout
            for S in STATIC_S:
                keep = TG <= S
                Mb = np.hstack([P[:, keep, :].transpose(0, 2, 1).reshape(nb, -1),
                                inputs_full])
                Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
                r = C.analyse(Mb, Xb, Y)
                r.update(variant='V1_static', model=name, param=S,
                         response=rname, nM=Mb.shape[1], nX=Xb.shape[1])
                rows.append(r)
                print(f'  V1 {name:14s} S={S:4.0f}h  M{Mb.shape[1]:4d}/X{Xb.shape[1]:4d}  '
                      f'LV=({r["LV_M"]:2d},{r["LV_X"]:2d})  Q2 {r["Q2_full"]:.4f}  '
                      f'uM {r["uM"]:+.4f}  uX {r["uX"]:+.4f}  sh {r["shared"]:+.4f}  '
                      f'M2M* {r["M2M_mod"]:9.3f}  M2M {r["M2M_conv"]:9.3f}', flush=True)

            # -------------------------------------- V2 rolling hybrid block
            Mb = np.hstack([P.transpose(0, 2, 1).reshape(nb, -1), inputs_full])
            for k in ROLL_K:
                obs = TG <= k
                Xk = np.where(obs[None, :, None], meas, P)      # measured <=k, KD after
                Xb = Xk.transpose(0, 2, 1).reshape(nb, -1)
                r = C.analyse(Mb, Xb, Y)
                r.update(variant='V2_rolling', model=name, param=k,
                         response=rname, nM=Mb.shape[1], nX=Xb.shape[1])
                rows.append(r)
                print(f'  V2 {name:14s} k={k:4.0f}h  M{Mb.shape[1]:4d}/X{Xb.shape[1]:4d}  '
                      f'LV=({r["LV_M"]:2d},{r["LV_X"]:2d})  Q2 {r["Q2_full"]:.4f}  '
                      f'uM {r["uM"]:+.4f}  uX {r["uX"]:+.4f}  sh {r["shared"]:+.4f}  '
                      f'M2M* {r["M2M_mod"]:9.3f}  M2M {r["M2M_conv"]:9.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('urethane_m2m_blockvariants.xlsx', index=False)

    print('\n\n================ ranking check: does the metric pick U0? ================')
    print(f'{"variant":11s} {"resp":5s} {"param":>6s} {"M2M* U0":>10s} {"M2M* U1":>10s} '
          f'{"mod ok":>7s} {"M2M U0":>10s} {"M2M U1":>10s} {"conv ok":>8s} '
          f'{"uX U0":>8s} {"uX U1":>8s}')
    okm = okc = tot = 0
    for (v, rn, p), g in df.groupby(['variant', 'response', 'param'], sort=True):
        try:
            a = g[g.model == 'U0_correct'].iloc[0]
            b = g[g.model == 'U1_no_reverse'].iloc[0]
        except IndexError:
            continue
        m_ok = a.M2M_mod > b.M2M_mod
        c_ok = a.M2M_conv > b.M2M_conv
        okm += m_ok; okc += c_ok; tot += 1
        print(f'{v:11s} {rn:5s} {p:6.0f} {a.M2M_mod:10.3f} {b.M2M_mod:10.3f} '
              f'{("YES" if m_ok else "no"):>7s} {a.M2M_conv:10.3f} {b.M2M_conv:10.3f} '
              f'{("YES" if c_ok else "no"):>8s} {a.uX:8.4f} {b.uX:8.4f}')
    print(f'\nmodified M2M correct {okm}/{tot}    conventional M2M correct {okc}/{tot}')
    print(f'\nSaved urethane_m2m_blockvariants.xlsx  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
