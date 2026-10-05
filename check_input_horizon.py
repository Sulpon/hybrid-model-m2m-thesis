"""
Does giving the mechanistic block the FULL input programme bias the result?

In the V1 static layout the states of M and X share a grid (0-S h), but the
unfolded input programme appended to M spans the whole batch (0-80 h). So M
carries 3 x 20 = 60 columns describing t > 70 h -- the window in which the
response is measured -- and X carries none.

The justification is that the input programme is the RECIPE: it is known
before the batch starts, and case study 1 does the same thing (M2 holds
t2, T2, CD0, which describe all of stage 2 including past the truncation).
But it is still an asymmetry, so this measures it: the same analysis run with
the inputs truncated to the states' grid, making M and X cover exactly the
same time span.
"""
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import case_study_2 as U

S = 70.0


def main():
    charges, inps, clean, meas = U.generate()
    nb = len(inps)
    TG = U.TGRID
    keep = TG <= S
    raw = np.array([[f(tt) for tt in TG] for f in inps])          # (nb, 161, 3)
    inp_full = raw.transpose(0, 2, 1).reshape(nb, -1)             # 3 x 161
    inp_trunc = raw[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)  # 3 x 141

    cache = json.load(open(U.CACHE))
    preds = {}
    for name, (rev, npar) in U.CANDIDATES.items():
        th = np.array(cache[name]['th'])
        rhs = U.make_rhs(rev, th)
        preds[name] = np.array([U.simulate(rhs, f, c) for f, c in zip(inps, charges)])

    rows = []
    for rname, ri in [('nC', 0), ('nD', 1)]:
        Y = meas[:, -1, ri].reshape(-1, 1)
        for name in U.CANDIDATES:
            st = preds[name][:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
            Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
            for tag, inp in [('inputs 0-80 h (current)', inp_full),
                             ('inputs 0-70 h (symmetric)', inp_trunc)]:
                Mb = np.hstack([st, inp])
                r = U.analyse(Mb, Xb, Y)
                r.update(response=rname, model=name, variant=tag, nM=Mb.shape[1])
                rows.append(r)
                print(f'{rname}  {name:14s} {tag:26s} M {Mb.shape[1]:4d}  '
                      f'uM {r["uM"]:+.4f}  uX {r["uX"]:+.4f}  '
                      f'M2M* {r["M2M_mod"]:9.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('check_input_horizon.xlsx', index=False)
    print('\n' + '='*72)
    print('M2M* under each input span')
    print('='*72)
    p = df.pivot_table(index=['response', 'model'], columns='variant', values='M2M_mod')
    print(p.round(3).to_string())
    print('\nverdict per response: does the correct model still sit above 1 '
          'and the broken one below?')
    for rname in ['nC', 'nD']:
        for tag in ['inputs 0-80 h (current)', 'inputs 0-70 h (symmetric)']:
            u0 = p.loc[(rname, 'U0_correct'), tag]
            u1 = p.loc[(rname, 'U1_no_reverse'), tag]
            ok = 'YES' if (u0 > 1 and u1 < 1) else 'NO'
            print(f'  {rname}  {tag:26s} U0 {u0:9.3f}  U1 {u1:7.3f}   {ok}')
    print('\nSaved check_input_horizon.xlsx')


if __name__ == '__main__':
    main()
