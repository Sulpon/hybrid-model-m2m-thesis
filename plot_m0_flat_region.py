"""
M0 (correct model): the RMSECV landscape and the flat region.

Left  -- every one of the 6 x 15 x 15 allocations, RMSECV against total latent
         variables, with the 1-SE flat region picked out and the two candidate
         selections marked.
Right -- the flat region only, coloured by the modified metric M2M* = uM/uX,
         which shows how much the metric moves across allocations that are
         statistically indistinguishable in RMSECV.

Dataset A, machinery imported from case_study_1 unchanged.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import case_study_1 as CS

MODEL = 'M0_correct'


def main():
    A = CS.load_A()
    d = A['std']
    X1, X2, Y = d['X1'], d['X2'], d['Y']
    M2 = CS.build_M2_A(CS.MODELS[MODEL], d, CS.FITTED_A[MODEL])

    qF, rf = CS.cv_3(X1, M2, X2, Y, CS.MAX_X1, CS.MAX_M2, CS.MAX_X2)
    qKD, _ = CS.cv_2(X1, M2, Y, CS.MAX_X1, CS.MAX_M2)
    qDD, _ = CS.cv_2(X1, X2, Y, CS.MAX_X1, CS.MAX_X2)

    # The two sub-models are re-optimised on their OWN grids -- the protocol the
    # reported table uses. (Letting them inherit the full model's allocation
    # instead gives a different, larger ratio; that variant is the one in the
    # earlier flat-region analysis, median 3.84 rather than ~1.9 here.)
    from check_lv_rule import pick2
    kd = pick2(X1, M2, Y, CS.MAX_X1, CS.MAX_M2)
    dd = pick2(X1, X2, Y, CS.MAX_X1, CS.MAX_X2)
    q_kd, q_dd = float(qKD[kd]), float(qDD[dd])
    print(f'sub-models (1-SE, re-optimised):  KD {tuple(v+1 for v in kd)} '
          f'Q2 {q_kd:.4f}    DD {tuple(v+1 for v in dd)} Q2 {q_dd:.4f}')
    with np.errstate(divide='ignore', invalid='ignore'):
        uM, uX = qF - q_dd, qF - q_kd
        ratio = np.where(uX > 1e-9, uM/uX, np.nan)

    rmse = rf.mean(0)
    imin = np.unravel_index(rmse.argmin(), rmse.shape)
    rmin = float(rmse[imin])
    se = float(rf[:, imin[0], imin[1], imin[2]].std(ddof=1)/np.sqrt(CS.N_SPLITS))
    thr = rmin + se
    mask = rmse <= thr
    pick, _, _, _, _ = CS.pick_1se(rmse, rf)

    a, b, c = np.indices(rmse.shape)
    tot = (a + b + c + 3).ravel()
    rv, mv, rat = rmse.ravel(), mask.ravel(), ratio.ravel()

    fmt = lambda t: '(' + ','.join(str(int(v)) for v in t) + ')'
    best_lv = fmt(v+1 for v in imin)
    par_lv = fmt(v+1 for v in pick)
    best_tot = sum(int(v)+1 for v in imin)
    par_tot = sum(int(v)+1 for v in pick)
    print(f'\n{MODEL}:  argmin {best_lv} tot {best_tot}  RMSECV {rmin:.4f}  '
          f'M2M* {ratio[imin]:.3f}')
    print(f'{" "*len(MODEL)}   1-SE   {par_lv} tot {par_tot}  '
          f'RMSECV {rmse[pick]:.4f}  M2M* {ratio[pick]:.3f}')
    fr = rat[mv]; fr = fr[np.isfinite(fr)]
    print(f'flat region: n = {mask.sum()} of {rmse.size}   threshold {thr:.4f} '
          f'(= {rmin:.4f} + {se:.4f})')
    print(f'M2M* across the flat region: median {np.median(fr):.3f}  '
          f'p10 {np.percentile(fr,10):.3f}  p90 {np.percentile(fr,90):.3f}  '
          f'min {fr.min():.3f}  max {fr.max():.3f}')

    fig, ax = plt.subplots(1, 2, figsize=(15.2, 5.6))

    # ---------------------------------------------------- left: whole grid
    ax[0].scatter(tot[~mv], rv[~mv], s=7, c='0.72', edgecolor='none',
                  label=f'all configurations (6×15×15 = {rmse.size})')
    ax[0].scatter(tot[mv], rv[mv], s=22, c='#2e6f9e', edgecolor='none',
                  label=f'within 1-SE (n={mask.sum()})')
    ax[0].axhline(thr, color='k', ls='--', lw=1.4)
    ax[0].text(35.5, thr+0.012, f'1-SE = {thr:.3f}', ha='right', fontsize=11,
               fontweight='bold')
    ax[0].scatter([best_tot], [rmin], s=430, marker='*', c='gold',
                  edgecolor='k', linewidth=1.3, zorder=6,
                  label=f'best / argmin {best_lv}')
    ax[0].scatter([par_tot], [rmse[pick]], s=185, marker='D', c='none',
                  edgecolor='#0b2545', linewidth=2.4, zorder=7,
                  label=f'1-SE parsimonious {par_lv}')
    ax[0].set_xlabel('Total components (a+b+c)', fontsize=12)
    ax[0].set_ylabel('RMSECV', fontsize=12)
    ax[0].set_title('M0 correct model — the full RMSECV landscape',
                    fontsize=13, fontweight='bold')
    ax[0].legend(fontsize=10, loc='upper right')
    ax[0].grid(alpha=0.3)

    # ------------------------------------- right: flat region, coloured by M2M*
    sc = ax[1].scatter(tot[mv], rv[mv], c=rat[mv], s=62, cmap='RdYlBu',
                       edgecolor='0.25', linewidth=0.5,
                       vmin=float(fr.min()), vmax=float(fr.max()))
    cb = fig.colorbar(sc, ax=ax[1]); cb.set_label('modified M2M*  =  u$_M$/u$_X$', fontsize=11)
    ax[1].axhline(thr, color='k', ls='--', lw=1.4)
    ax[1].scatter([best_tot], [rmin], s=430, marker='*', c='gold',
                  edgecolor='k', linewidth=1.3, zorder=6)
    ax[1].scatter([par_tot], [rmse[pick]], s=185, marker='D', c='none',
                  edgecolor='#0b2545', linewidth=2.4, zorder=7)
    # annotate inline -- a legend here would sit on top of the argmin star
    ax[1].annotate(f'best / argmin {best_lv}\nM2M* = {ratio[imin]:.2f}',
                   xy=(best_tot, rmin), xytext=(best_tot+1.4, rmin+0.004),
                   fontsize=10, fontweight='bold', va='center',
                   arrowprops=dict(arrowstyle='-', lw=1.0, color='0.35'))
    ax[1].annotate(f'1-SE parsimonious {par_lv}\nM2M* = {ratio[pick]:.2f}',
                   xy=(par_tot, rmse[pick]), xytext=(par_tot+0.8, rmse[pick]-0.018),
                   fontsize=10, fontweight='bold', va='center',
                   arrowprops=dict(arrowstyle='-', lw=1.0, color='0.35'))
    ax[1].set_ylim(rmin-0.015, thr+0.012)
    ax[1].set_xlabel('Total components (a+b+c)', fontsize=12)
    ax[1].set_ylabel('RMSECV', fontsize=12)
    ax[1].set_title(f'flat region coloured by M2M*   '
                    f'median = {np.median(fr):.2f}   range {fr.min():.2f}–{fr.max():.2f}',
                    fontsize=13, fontweight='bold')
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('deck_m0_flat_region.png', dpi=190)
    print('\nSaved deck_m0_flat_region.png')


if __name__ == '__main__':
    main()
