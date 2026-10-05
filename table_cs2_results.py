"""
Case study 2 results table, matching the case study 1 layout.

The deck's case study 2 numbers were single-seed point estimates. Here the
same V1 static analysis (shared M/X grid 0-70 h, response at 80 h) is run over
the SAME 20 cross-validation seeds used for case study 1, so the two tables
carry the same columns and the same meaning of "+/- sd" and "P(>1)".

Allocations are fixed once by parsimony inside the 1-SE flat region at the
reference seed, then held while the seed varies -- exactly the protocol used
for the case study 1 table.
"""
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import case_study_2 as U
from check_lv_rule import q2_at, SEEDS

S = 70.0
NAVY = '#1F2A63'
ROW_A, ROW_B = '#FFFFFF', '#F2F4F9'
RULE = '#D8DCE8'
LABEL = {('U0_correct', 'nC'): 'U0  correct            (n$_C$)',
         ('U1_no_reverse', 'nC'): 'U1  no reverse step    (n$_C$)',
         ('U0_correct', 'nD'): 'U0  correct            (n$_D$)',
         ('U1_no_reverse', 'nD'): 'U1  no reverse step    (n$_D$)'}


def compute():
    charges, inps, clean, meas = U.generate()
    nb = len(inps)
    TG = U.TGRID
    keep = TG <= S
    inp_full = np.array([[f(tt) for tt in TG] for f in inps]
                        ).transpose(0, 2, 1).reshape(nb, -1)
    cache = json.load(open(U.CACHE))

    rows = []
    for rname, ri in [('nC', 0), ('nD', 1)]:
        Y = meas[:, -1, ri].reshape(-1, 1)
        for name, (rev, npar) in U.CANDIDATES.items():
            th = np.array(cache[name]['th'])
            P = np.array([U.simulate(U.make_rhs(rev, th), f, c)
                          for f, c in zip(inps, charges)])
            Mb = np.hstack([P[:, keep, :].transpose(0, 2, 1).reshape(nb, -1), inp_full])
            Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)

            sel = U.analyse(Mb, Xb, Y)          # allocation by 1-SE parsimony
            a, b = sel['LV_M'], sel['LV_X']
            acc = []
            for sd_ in SEEDS:
                qf = q2_at([Mb, Xb], Y, (a, b), sd_)
                qm = q2_at([Mb], Y, (a,), sd_)      # M alone  -> Q2 "no-X"
                qx = q2_at([Xb], Y, (b,), sd_)      # X alone  -> Q2 "no-M"
                acc.append((qf, qx, qm, qf-qx, qf-qm))
            acc = np.array(acc)*100
            rat = acc[:, 3]/acc[:, 4]
            rows.append(dict(model=name, response=rname, LV=f'({a},{b})',
                             Q2_full=acc[:, 0].mean(), Q2_noM=acc[:, 1].mean(),
                             Q2_noX=acc[:, 2].mean(), uM=acc[:, 3].mean(),
                             uX=acc[:, 4].mean(), ratio=np.nanmean(rat),
                             sd=np.nanstd(rat, ddof=1),
                             frac_gt1=float(np.nanmean(rat > 1))))
            r = rows[-1]
            print(f'{rname}  {name:14s} LV {r["LV"]:7s} Q2 {r["Q2_full"]:6.2f}  '
                  f'uM {r["uM"]:6.2f}  uX {r["uX"]:6.2f}  '
                  f'M2M* {r["ratio"]:9.3f} +/- {r["sd"]:.3f}  '
                  f'P(>1) {r["frac_gt1"]:.2f}', flush=True)
    return pd.DataFrame(rows)


def render(df):
    order = [('U0_correct', 'nC'), ('U1_no_reverse', 'nC'),
             ('U0_correct', 'nD'), ('U1_no_reverse', 'nD')]
    hdr = ['model', 'Q$^2$ full', 'Q$^2$ no-M', 'Q$^2$ no-X',
           'u$_m$', 'u$_x$', 'M2M*', '± sd', 'P(>1)']
    W = 15.4
    xs = [0.0, 4.05, 5.60, 7.15, 8.70, 10.05, 11.40, 12.95, 14.10, W]
    hrow, rrow = 0.66, 0.70
    total = hrow + rrow*len(order)

    fig, ax = plt.subplots(figsize=(W, total + 0.25))
    ax.set_xlim(0, W); ax.set_ylim(0, total); ax.axis('off')

    y = total
    ax.add_patch(Rectangle((0, y-hrow), W, hrow, facecolor=NAVY, edgecolor='none'))
    for j, t in enumerate(hdr):
        cx = xs[j] + 0.22 if j == 0 else (xs[j]+xs[j+1])/2
        ax.text(cx, y-hrow/2, t, ha='left' if j == 0 else 'center', va='center',
                fontsize=15, color='white', fontweight='bold')
    y -= hrow

    g = df.set_index(['model', 'response'])
    for i, key in enumerate(order):
        r = g.loc[key]
        ax.add_patch(Rectangle((0, y-rrow), W, rrow,
                               facecolor=ROW_A if i % 2 == 0 else ROW_B,
                               edgecolor='none'))
        ax.plot([0, W], [y-rrow, y-rrow], color=RULE, lw=0.8)
        vals = [LABEL[key], f'{r.Q2_full:.2f}', f'{r.Q2_noM:.2f}', f'{r.Q2_noX:.2f}',
                f'{r.uM:.2f}', f'{r.uX:.2f}',
                (f'{r.ratio:.2f}' if r.ratio < 100 else f'{r.ratio:.0f}'),
                f'{r.sd:.2f}' if r.sd < 100 else f'{r.sd:.0f}', f'{r.frac_gt1:.2f}']
        for j, t in enumerate(vals):
            cx = xs[j] + 0.22 if j == 0 else (xs[j]+xs[j+1])/2
            ax.text(cx, y-rrow/2, t, ha='left' if j == 0 else 'center',
                    va='center', fontsize=14.5, color='#1A1A1A')
        y -= rrow

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    fig.savefig('deck_cs2_results_table.png', dpi=200, facecolor='white',
                bbox_inches='tight', pad_inches=0.06)


if __name__ == '__main__':
    df = compute()
    df.to_excel('table_cs2_results.xlsx', index=False)
    render(df)
    print('\nSaved table_cs2_results.xlsx and deck_cs2_results_table.png')
