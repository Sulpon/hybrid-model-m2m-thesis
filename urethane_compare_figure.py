"""
Overlay the authors' published mole profiles against our reconstruction.

The published curves come from urethane_digitise_figure.py, which recovers
them from the PNG stored inside the authors' own notebook. Ours come from
urethane_authors_replicate.py, which runs the authors' model and parameters
on the charge and feed programme reconstructed from those same figures.

Both sides carry measurement noise, and not merely noise of the same size:
the authors call np.random.seed(42) and then draw C, D and E in that order
from the legacy MT19937 stream. Reproducing the seed, the draw order and the
0.5 h sample grid therefore reproduces their *individual* realisation, so the
published wiggles and ours should coincide point by point. The check at the
bottom of this script tests exactly that, by correlating our known noise
vector against the high-frequency part of the digitised published trace.

That makes the comparison sharper than a smooth-curve overlay: it separates
"our model is right" from "our model is right and we are on their grid".
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import urethane_authors_replicate as U
import urethane_digitise_figure as D

PANELS = [('A', '$n_A$  isocyanate', 'mol'), ('B', '$n_B$  butanol', 'mol'),
          ('C', '$n_C$  urethane', 'mol'), ('D', '$n_D$  allophanate', 'mol'),
          ('E', '$n_E$  isocyanurate', 'mol'), ('V', 'reactor volume', 'm$^3$')]
NOISY = set('CDE')          # the authors perturb only these three

if __name__ == '__main__':
    paper = D.extract(verbose=False)
    t = np.arange(0.0, 80.0 + 1e-9, 0.5)        # the authors' sample grid, 161 pts
    clean = U.simulate(U.BATCH1_CHARGE, U.BATCH1_INPUTS, t)
    meas = U.add_noise(clean, seed=42)
    print(f'grid: {len(t)} samples at {t[1]-t[0]} h')

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    fig.suptitle('Urethane process, Batch_1 — published (HyMech notebooks) vs '
                 'reconstruction, both with measurement noise', fontsize=13)

    print(f'\n{"species":9s} {"n":>5s} {"RMS abs":>11s} {"RMS %range":>12s} '
          f'{"max abs dev":>13s} {"peak paper":>11s} {"peak ours":>10s}')
    stats = []
    for ax, (key, title, unit) in zip(axes.ravel(), PANELS):
        tp, yp = paper[key]
        m = (tp >= 0.0) & (tp <= 80.0)
        tp, yp = tp[m], yp[m]
        yo = np.interp(tp, t, meas[key])

        rng = yp.max() - yp.min()
        res = yo - yp
        rms = float(np.sqrt(np.mean(res**2)))
        stats.append(100*rms/rng)
        print(f'{"n"+key:9s} {len(tp):5d} {rms:11.4g} {100*rms/rng:11.1f}% '
              f'{np.abs(res).max():13.4g} {yp.max():11.4g} {meas[key].max():10.4g}')

        ax.plot(tp, yp, '-', color='#1f77b4', lw=2.2, alpha=0.9,
                label='published (digitised)')
        if key in NOISY:
            ax.plot(t, meas[key], 'o', ms=3.0, mfc='none', mec='#c0392b',
                    mew=0.9, ls='none', label='ours (noisy samples)')
            ax.plot(t, clean[key], '--', color='#c0392b', lw=1.2, alpha=0.75,
                    label='ours (noise-free)')
        else:
            ax.plot(t, clean[key], '--', color='#c0392b', lw=1.8,
                    label='ours')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('time [h]'); ax.set_ylabel(f'[{unit}]')
        ax.set_xlim(0, 80); ax.set_xticks([0, 20, 40, 60, 80])
        ax.grid(alpha=0.3); ax.tick_params(direction='in', top=True, right=True)
    axes[0, 2].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig('urethane_compare_published.png', dpi=200)

    # ------------------------------------------------ residual-only companion
    fig2, axes2 = plt.subplots(2, 3, figsize=(13.5, 6.4))
    fig2.suptitle("Reconstruction minus published, as % of each panel's range",
                  fontsize=13)
    for ax, (key, title, unit) in zip(axes2.ravel(), PANELS):
        tp, yp = paper[key]
        m = (tp >= 0.0) & (tp <= 80.0)
        tp, yp = tp[m], yp[m]
        yo = np.interp(tp, t, clean[key])
        rng = yp.max() - yp.min()
        ax.axhline(0, color='k', lw=0.8)
        ax.fill_between(tp, 0, 100*(yo - yp)/rng, color='#c0392b', alpha=0.55)
        ax.set_title(title + '   (noise-free vs published)', fontsize=10)
        ax.set_xlabel('time [h]'); ax.set_ylabel('deviation [% of range]')
        ax.set_xlim(0, 80); ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_ylim(-45, 45); ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('urethane_compare_residuals.png', dpi=200)

    # ------------------------------- did we land on the SAME noise realisation?
    print('\nnoise-realisation test  (our known draw vs the published wiggle)')
    print(f'{"sp":4s} {"sigma":>9s} {"our sd":>9s} {"paper hf sd":>12s} {"corr":>8s}')
    for key in 'CDE':
        ours_noise = meas[key] - clean[key]
        tp, yp = paper[key]
        m = (tp >= 0.0) & (tp <= 80.0)
        tp, yp = tp[m], yp[m]
        # high-frequency part of the published trace = its own noise
        sm = savgol_filter(yp, 21, 3)
        hf = yp - sm
        hf_on_grid = np.interp(t, tp, hf)
        ours_hf = ours_noise - savgol_filter(ours_noise, 21, 3)
        c = float(np.corrcoef(ours_hf, hf_on_grid)[0, 1])
        print(f'n{key:3s} {U.meas_err_std[key]:9.1e} {ours_noise.std(ddof=1):9.3g} '
              f'{hf.std(ddof=1):12.3g} {c:8.3f}')

    print(f'\noverall RMS deviation {np.sqrt(np.mean(np.square(stats))):.1f}% of panel range')
    print('Saved urethane_compare_published.png and urethane_compare_residuals.png')
