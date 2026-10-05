"""
The seven-candidate table under the SUM-OF-SQUARES estimator.

Companion to the cross-validated table (check_lv_rule.py). Both metrics are
computed here from in-sample sequential (Type-I) sums of squares:

  conventional  M2M  = SS_M2 (forward, entering 2nd) / SS_X2 (forward, 3rd)
  modified      M2M* = SS_M2 (reverse, entering 3rd) / SS_X2 (forward, 3rd)

The modified numerator uses the reverse ordering X1 -> X2 -> M2, where M2
enters last and so is credited only with what X1 and X2 could not already
explain -- the in-sample analogue of unique_M. Both orderings are held at the
full model's per-block latent-variable counts, so the comparison is not
contaminated by the reverse run re-optimising M2 upward.

Selection is unchanged (RMSECV, argmin and the 1-SE parsimonious member), and
the flat-region spread is reported because, unlike the CV estimator, the SS
one moves a great deal across statistically equivalent allocations.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import case_study_1 as CS
from plot_m0_flat_region_ss import ss_surface_2

BAD = {'M1_no_side', 'M5_no_D', 'M6_author_mis'}
ORDER = ['M0_correct', 'M2_order1_D', 'M3_wrong_Ea4', 'M4_lumped_EF',
         'M6_author_mis', 'M1_no_side', 'M5_no_D']


def main():
    A = CS.load_A()
    rows = []
    for name in ORDER:
        d = A['mis'] if name == 'M6_author_mis' else A['std']
        X1, X2, Y = d['X1'], d['X2'], d['Y']
        M2 = CS.build_M2_A(CS.MODELS[name], d, CS.FITTED_A[name])

        _, rf = CS.cv_3(X1, M2, X2, Y, CS.MAX_X1, CS.MAX_M2, CS.MAX_X2)
        rmse = rf.mean(0)
        imin = np.unravel_index(rmse.argmin(), rmse.shape)
        rmin = float(rmse[imin])
        se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(CS.N_SPLITS))
        mask = rmse <= rmin + se
        pick, _, _, _, _ = CS.pick_1se(rmse, rf)

        ssX = CS.ss_surface_3(X1, M2, X2, Y, CS.MAX_X1, CS.MAX_M2, CS.MAX_X2)
        ssM = np.transpose(CS.ss_surface_3(X1, X2, M2, Y, CS.MAX_X1, CS.MAX_X2,
                                           CS.MAX_M2), (0, 2, 1))
        ssF = ss_surface_2(X1, M2, Y, CS.MAX_X1, CS.MAX_M2)
        with np.errstate(divide='ignore', invalid='ignore'):
            mod = np.where(ssX > 1e-12, ssM/ssX, np.nan)
            conv = np.where(ssX > 1e-12, ssF[:, :, None]/ssX, np.nan)

        fmt = lambda t: '(' + ','.join(str(int(v)+1) for v in t) + ')'
        fm = mod.ravel()[mask.ravel()]; fm = fm[np.isfinite(fm)]
        rows.append(dict(
            model=name, sound='sound' if name not in BAD else 'BROKEN',
            LV_argmin=fmt(imin), LV_1SE=fmt(pick), n_1SE=int(mask.sum()),
            mod_argmin=float(mod[imin]), mod_1SE=float(mod[pick]),
            conv_argmin=float(conv[imin]), conv_1SE=float(conv[pick]),
            mod_flat_med=float(np.median(fm)),
            mod_flat_min=float(fm.min()), mod_flat_max=float(fm.max()),
            frac_gt1=float((fm > 1).mean())))
        r = rows[-1]
        print(f'{name:14s} {r["sound"]:6s} argmin {r["LV_argmin"]:8s} '
              f'mod {r["mod_argmin"]:6.3f}  conv {r["conv_argmin"]:6.2f}   |   '
              f'1-SE {r["LV_1SE"]:8s} mod {r["mod_1SE"]:6.3f}  conv {r["conv_1SE"]:6.2f}',
              flush=True)

    df = pd.DataFrame(rows).set_index('model')
    df.to_excel('table_ss_estimator.xlsx')

    print('\n' + '='*78)
    print('SS ESTIMATOR -- does it separate sound from broken?')
    print('='*78)
    for col, lab in [('mod_argmin', 'modified, argmin'), ('mod_1SE', 'modified, 1-SE'),
                     ('conv_argmin', 'conventional, argmin'), ('conv_1SE', 'conventional, 1-SE')]:
        lo = df.loc[df.sound == 'sound', col].min()
        hi = df.loc[df.sound == 'BROKEN', col].max()
        print(f'  {lab:22s} worst sound {lo:7.3f}   best broken {hi:7.3f}   '
              f'{"SEPARATES" if lo > hi else "*** OVERLAPS ***"}')

    print('\nspread of the modified SS ratio across each flat region')
    print(f'  {"model":14s} {"median":>8s} {"min":>8s} {"max":>8s} {"P(>1)":>7s}')
    for m, r in df.iterrows():
        flag = '  <-- crosses 1' if r.mod_flat_min < 1.0 < r.mod_flat_max else ''
        print(f'  {m:14s} {r.mod_flat_med:8.3f} {r.mod_flat_min:8.3f} '
              f'{r.mod_flat_max:8.3f} {r.frac_gt1:7.2f}{flag}')
    print('\nSaved table_ss_estimator.xlsx')


if __name__ == '__main__':
    main()
