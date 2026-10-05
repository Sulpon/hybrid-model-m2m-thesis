"""
Is the case study 2 instability caused by having only 40 batches?

N_BATCH = 40 was a pragmatic choice, not a derived one. With 906 columns in
the mechanistic block that is a 1:23 sample-to-variable ratio, and 10-fold CV
leaves 36 samples per training fold. Case study 1 uses 100.

The kinetic parameters are fitted on the first N_FIT = 8 batches only, and
generate() draws from a fixed RNG stream, so batches 1-8 are identical for any
N. The cached fits therefore stay valid and only the block size changes.

Reports M2M* mean, sd and P(>1) over the same 20 CV seeds at each N.
"""
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import case_study_2 as U
from check_lv_rule import q2_at, SEEDS

S = 70.0
NS = [40, 100, 200]


def run(N):
    charges, inps, clean, meas = U.generate(n=N)
    nb = len(inps)
    TG = U.TGRID
    keep = TG <= S
    inp_full = np.array([[f(tt) for tt in TG] for f in inps]
                        ).transpose(0, 2, 1).reshape(nb, -1)
    cache = json.load(open(U.CACHE))
    out = []
    for rname, ri in [('nC', 0), ('nD', 1)]:
        Y = meas[:, -1, ri].reshape(-1, 1)
        for name, (rev, npar) in U.CANDIDATES.items():
            th = np.array(cache[name]['th'])
            P = np.array([U.simulate(U.make_rhs(rev, th), f, c)
                          for f, c in zip(inps, charges)])
            Mb = np.hstack([P[:, keep, :].transpose(0, 2, 1).reshape(nb, -1), inp_full])
            Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
            sel = U.analyse(Mb, Xb, Y)
            a, b = sel['LV_M'], sel['LV_X']
            acc = []
            for sd_ in SEEDS:
                qf = q2_at([Mb, Xb], Y, (a, b), sd_)
                qm = q2_at([Mb], Y, (a,), sd_)
                qx = q2_at([Xb], Y, (b,), sd_)
                acc.append((qf, qf-qx, qf-qm))
            acc = np.array(acc)*100
            rat = acc[:, 1]/acc[:, 2]
            out.append(dict(N=nb, response=rname, model=name, LV=f'({a},{b})',
                            Q2_full=acc[:, 0].mean(), uM=acc[:, 1].mean(),
                            uX=acc[:, 2].mean(), ratio=np.nanmean(rat),
                            median=float(np.nanmedian(rat)),
                            sd=np.nanstd(rat, ddof=1),
                            frac_gt1=float(np.nanmean(rat > 1))))
            r = out[-1]
            print(f'  N={nb:3d}  {rname}  {name:14s} LV {r["LV"]:7s} '
                  f'uM {r["uM"]:6.2f} uX {r["uX"]:6.2f}  '
                  f'M2M* med {r["median"]:8.2f}  mean {r["ratio"]:9.2f} '
                  f'+/- {r["sd"]:8.2f}   P(>1) {r["frac_gt1"]:.2f}', flush=True)
    return out


if __name__ == '__main__':
    rows = []
    for N in NS:
        print(f'--- {N} batches ---', flush=True)
        rows += run(N)
    df = pd.DataFrame(rows)
    df.to_excel('check_n_batches.xlsx', index=False)

    print('\n' + '='*74)
    print('P(M2M* > 1) -- how reliably each model is placed on the right side of 1')
    print('='*74)
    print(df.pivot_table(index=['response', 'model'], columns='N',
                         values='frac_gt1').round(2).to_string())
    print('\nuX (%) -- the denominator that drives the instability')
    print(df.pivot_table(index=['response', 'model'], columns='N',
                         values='uX').round(3).to_string())
    print('\nSaved check_n_batches.xlsx')
