"""
M0 (correct model): the flat region under the SUM-OF-SQUARES estimator.

Companion to plot_m0_flat_region.py, which uses the cross-validated Q2
estimator. Here both metrics are computed from in-sample sequential (Type-I)
sums of squares instead:

  conventional  M2M  = SS_M2 (forward, entering 2nd) / SS_X2 (forward, 3rd)
  modified      M2M* = SS_M2 (reverse, entering 3rd) / SS_X2 (forward, 3rd)

The modified numerator comes from the REVERSE ordering X1 -> X2 -> M2, where
M2 enters last and is therefore credited only with what X1 and X2 could not
already explain -- the in-sample analogue of unique_M.

Selection (RMSECV, the 1-SE flat region) is unchanged; only the quantity the
points are coloured by differs.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import case_study_1 as CS

MODEL = 'M0_correct'


def ss_surface_2(B1, B2, Y, m1, m2):
    """Type-I SS taken by B2 when it enters SECOND, over the (a,b) grid."""
    mu, sd, k = CS._sf(B1); b1 = CS._sa(B1, mu, sd, k)
    mu, sd, k = CS._sf(B2); b2_0 = CS._sa(B2, mu, sd, k)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    out = np.zeros((m1, m2))
    p1, T1, Q1, af = CS._fit(b1, ys, m1)
    for a in range(1, m1+1):
        ae = min(a, af); Ta = T1[:, :ae]
        b2_a, _ = CS._rs(Ta, b2_0); y_a, _ = CS._rs(Ta, ys)
        p2, T2v, Q2, bf = CS._fit(b2_a, y_a, m2)
        for b in range(1, m2+1):
            be = min(b, bf)
            out[a-1, b-1] = float(np.sum((T2v[:, :be] @ Q2[:, :be].T)**2))
    return out


def main():
    A = CS.load_A()
    d = A['std']
    X1, X2, Y = d['X1'], d['X2'], d['Y']
    M2 = CS.build_M2_A(CS.MODELS[MODEL], d, CS.FITTED_A[MODEL])

    # selection is still by RMSECV / 1-SE -- only the coloured quantity changes
    _, rf = CS.cv_3(X1, M2, X2, Y, CS.MAX_X1, CS.MAX_M2, CS.MAX_X2)
    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin])
    se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(CS.N_SPLITS))
    thr = rmin + se
    mask = rmse <= thr
    pick, _, _, _, _ = CS.pick_1se(rmse, rf)

    ssX = CS.ss_surface_3(X1, M2, X2, Y, CS.MAX_X1, CS.MAX_M2, CS.MAX_X2)   # (a,b,c)
    ssM = CS.ss_surface_3(X1, X2, M2, Y, CS.MAX_X1, CS.MAX_X2, CS.MAX_M2)   # (a,c,b)
    ssM = np.transpose(ssM, (0, 2, 1))                                      # -> (a,b,c)
    ssM2_fwd = ss_surface_2(X1, M2, Y, CS.MAX_X1, CS.MAX_M2)                # (a,b)

    with np.errstate(divide='ignore', invalid='ignore'):
        mod = np.where(ssX > 1e-12, ssM/ssX, np.nan)
        conv = np.where(ssX > 1e-12, ssM2_fwd[:, :, None]/ssX, np.nan)

    fmt = lambda t: '(' + ','.join(str(int(v)) for v in t) + ')'
    best_lv, par_lv = fmt(v+1 for v in imin), fmt(v+1 for v in pick)
    best_tot = sum(int(v)+1 for v in imin); par_tot = sum(int(v)+1 for v in pick)

    a, b, c = np.indices(rmse.shape)
    tot = (a + b + c + 3).ravel()
    rv, mv = rmse.ravel(), mask.ravel()

    print(f'{MODEL}  (SS estimator, dataset A)')
    print(f'  argmin {best_lv} tot {best_tot}  RMSECV {rmin:.4f}   '
          f'modified {mod[imin]:.3f}   conventional {conv[imin]:.2f}')
    print(f'  1-SE   {par_lv} tot {par_tot}  RMSECV {rmse[pick]:.4f}   '
          f'modified {mod[pick]:.3f}   conventional {conv[pick]:.2f}')
    print(f'  flat region n = {mask.sum()}   threshold {thr:.4f}')
    for nm, arr in [('modified  SS_M(rev)/SS_X(fwd)', mod),
                    ('conventional  SS_M/SS_X', conv)]:
        v = arr.ravel()[mv]; v = v[np.isfinite(v)]
        print(f'  {nm:32s} median {np.median(v):7.3f}  p10 {np.percentile(v,10):7.3f}'
              f'  p90 {np.percentile(v,90):7.3f}  min {v.min():7.3f}  max {v.max():7.3f}')

    fig, ax = plt.subplots(1, 3, figsize=(20.5, 5.6))

    # ------------------------------------------------------ (a) the landscape
    ax[0].scatter(tot[~mv], rv[~mv], s=7, c='0.72', edgecolor='none',
                  label=f'all configurations ({rmse.size})')
    ax[0].scatter(tot[mv], rv[mv], s=22, c='#2e6f9e', edgecolor='none',
                  label=f'within 1-SE (n={mask.sum()})')
    ax[0].axhline(thr, color='k', ls='--', lw=1.4)
    ax[0].text(35.5, thr+0.012, f'1-SE = {thr:.3f}', ha='right',
               fontsize=11, fontweight='bold')
    ax[0].scatter([best_tot], [rmin], s=430, marker='*', c='gold',
                  edgecolor='k', linewidth=1.3, zorder=6,
                  label=f'best / argmin {best_lv}')
    ax[0].scatter([par_tot], [rmse[pick]], s=185, marker='D', c='none',
                  edgecolor='#0b2545', linewidth=2.4, zorder=7,
                  label=f'1-SE parsimonious {par_lv}')
    ax[0].set_xlabel('Total components (a+b+c)', fontsize=12)
    ax[0].set_ylabel('RMSECV', fontsize=12)
    ax[0].set_title('M0 correct model — RMSECV landscape', fontsize=13, fontweight='bold')
    ax[0].legend(fontsize=9.5, loc='upper right'); ax[0].grid(alpha=0.3)

    # --------------------------------------- (b),(c) flat region by each metric
    for k, (arr, nm, lab, ref) in enumerate(
            [(mod, 'modified   M2M* = SS$_M$(rev) / SS$_X$(fwd)',
              'modified M2M*  (SS)', 1.0),
             (conv, 'conventional   M2M = SS$_M$ / SS$_X$',
              'conventional M2M  (SS)', None)], start=1):
        v = arr.ravel()[mv]; fin = v[np.isfinite(v)]
        sc = ax[k].scatter(tot[mv], rv[mv], c=v, s=62, cmap='RdYlBu',
                           edgecolor='0.25', linewidth=0.5,
                           vmin=float(np.percentile(fin, 2)),
                           vmax=float(np.percentile(fin, 98)))
        cb = fig.colorbar(sc, ax=ax[k]); cb.set_label(lab, fontsize=11)
        ax[k].axhline(thr, color='k', ls='--', lw=1.4)
        ax[k].scatter([best_tot], [rmin], s=430, marker='*', c='gold',
                      edgecolor='k', linewidth=1.3, zorder=6)
        ax[k].scatter([par_tot], [rmse[pick]], s=185, marker='D', c='none',
                      edgecolor='#0b2545', linewidth=2.4, zorder=7)
        ax[k].annotate(f'argmin {best_lv}\n{arr[imin]:.2f}',
                       xy=(best_tot, rmin), xytext=(best_tot+1.5, rmin+0.005),
                       fontsize=10, fontweight='bold', va='center',
                       arrowprops=dict(arrowstyle='-', lw=1.0, color='0.35'))
        ax[k].annotate(f'1-SE {par_lv}\n{arr[pick]:.2f}',
                       xy=(par_tot, rmse[pick]), xytext=(par_tot+0.8, rmse[pick]-0.020),
                       fontsize=10, fontweight='bold', va='center',
                       arrowprops=dict(arrowstyle='-', lw=1.0, color='0.35'))
        ax[k].set_ylim(rmin-0.017, thr+0.012)
        ax[k].set_xlabel('Total components (a+b+c)', fontsize=12)
        ax[k].set_ylabel('RMSECV', fontsize=12)
        ax[k].set_title(f'{nm}\nmedian {np.median(fin):.2f}   '
                        f'range {fin.min():.2f}–{fin.max():.2f}', fontsize=12)
        ax[k].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('deck_m0_flat_region_ss.png', dpi=185)
    print('\nSaved deck_m0_flat_region_ss.png')


if __name__ == '__main__':
    main()
