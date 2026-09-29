"""
Figures for the progress presentation.

Case study 1 numbers come from the 20-seed CV estimator in
_archive/metric_unique_m2m.xlsx -- the most robust estimator produced, and the
one behind the handwritten summary table. Case study 2 numbers come from
case_study_2_results.xlsx.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GOOD, BAD, GREY = '#2e7d32', '#c0392b', '#9e9e9e'
BLUE, ORANGE = '#1f4e9c', '#e08214'
SRC = '_archive/metric_unique_m2m.xlsx'
SHORT = {'M0_correct': 'M0\ncorrect', 'M1_no_side': 'M1\nno side rxn',
         'M2_order1_D': 'M2\norder-1 in D', 'M3_wrong_Ea4': 'M3\nwrong Ea4',
         'M4_lumped_EF': 'M4\nlumped E/F', 'M5_no_D': 'M5\nno D dep.',
         'M6_author_mis': 'M6\nauthors\' mis.'}
BADSET = {'M1_no_side', 'M5_no_D', 'M6_author_mis'}


def cs1():
    d = pd.read_excel(SRC, sheet_name='summary').set_index('model')
    ps = pd.read_excel(SRC, sheet_name='per_seed')
    order = list(SHORT)
    d = d.reindex(order)
    lab = [SHORT[m] for m in order]
    x = np.arange(len(order))
    col = [BAD if m in BADSET else GOOD for m in order]

    # ---- Figure A: the modified metric, with the reference at 1 ----------
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.bar(x, d.CV_ratio_mean, 0.62, color=col,
           yerr=d.CV_ratio_sd, capsize=5, ecolor='#444444', error_kw=dict(lw=1.3))
    ax.axhline(1.0, color='k', ls='--', lw=2.0)
    ax.text(len(x)-0.35, 1.06, 'reference = 1', ha='right', fontsize=12, style='italic')
    ax.axhspan(0, 1.0, color='#c0392b', alpha=0.06)
    for i, m in enumerate(order):
        ax.text(i, d.CV_ratio_mean[m] + d.CV_ratio_sd[m] + 0.09,
                f'{d.CV_ratio_mean[m]:.2f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=11)
    ax.set_ylabel('modified M2M*  =  unique$_M$ / unique$_X$', fontsize=13)
    ax.set_ylim(0, 2.85); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Sound mechanisms sit near 2; broken ones fall to 1 or below',
                 fontsize=14, pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GOOD, label='sound mechanism'),
                       Patch(color=BAD, label='broken mechanism')],
              fontsize=11, loc='upper right')
    fig.tight_layout(); fig.savefig('deck_cs1_ratio.png', dpi=200); plt.close(fig)

    # ---- Figure B: original vs modified, side by side ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax[0].bar(x, d.M2M_current, 0.62, color=col)
    ax[0].axhline(1.0, color='k', ls='--', lw=1.6)
    for i, m in enumerate(order):
        ax[0].text(i, d.M2M_current[m] + 0.35, f'{d.M2M_current[m]:.1f}',
                   ha='center', fontsize=10.5)
    ax[0].set_title('Original   M2M = SS$_M$ / SS$_X$', fontsize=13)
    ax[0].set_ylabel('ratio'); ax[0].set_ylim(0, 15.5)

    ax[1].bar(x, d.CV_ratio_mean, 0.62, color=col)
    ax[1].axhline(1.0, color='k', ls='--', lw=2.0)
    ax[1].axhspan(0, 1.0, color='#c0392b', alpha=0.07)
    for i, m in enumerate(order):
        ax[1].text(i, d.CV_ratio_mean[m] + 0.07, f'{d.CV_ratio_mean[m]:.2f}',
                   ha='center', fontsize=10.5)
    ax[1].set_title('Modified   M2M* = u$_M$ / u$_X$', fontsize=13)
    ax[1].set_ylabel('ratio'); ax[1].set_ylim(0, 2.6)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([s.replace('\n', ' ') for s in lab],
                                           fontsize=8.5, rotation=20, ha='right')
        a.grid(axis='y', alpha=0.3)
    fig.suptitle('Both order the models. Only the modified one has an absolute meaning.',
                 fontsize=13.5)
    fig.tight_layout(); fig.savefig('deck_cs1_vs.png', dpi=200); plt.close(fig)

    # ---- Figure C: where the ratio comes from ----------------------------
    q = (ps.groupby('model')[['unique_M', 'unique_X']].mean()*100).reindex(order)
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.bar(x - 0.2, q.unique_M, 0.4, color=BLUE, label='unique$_M$  (mechanism only)')
    ax.bar(x + 0.2, q.unique_X, 0.4, color=ORANGE, label='unique$_X$  (measurements only)')
    for i, m in enumerate(order):
        ax.text(i-0.2, q.unique_M[m]+0.5, f'{q.unique_M[m]:.1f}', ha='center', fontsize=9.5)
        ax.text(i+0.2, q.unique_X[m]+0.5, f'{q.unique_X[m]:.1f}', ha='center', fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([s.replace('\n', ' ') for s in lab],
                                         fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('unique contribution to $Q^2$  [%]', fontsize=12)
    ax.set_ylim(0, 31); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=11)
    ax.set_title('Break the mechanism and the measurements take over: '
                 'u$_X$ jumps from ~6 % to 16–27 %', fontsize=13, pad=10)
    fig.tight_layout(); fig.savefig('deck_cs1_uniques.png', dpi=200); plt.close(fig)
    return d


def cs2():
    d = pd.read_excel('case_study_2_results.xlsx')
    d = d[(d.variant == 'V1_static') & (d.param == 70.0)].set_index(['response', 'model'])
    labels, mod, conv, qm, qx = [], [], [], [], []
    for resp in ['nC', 'nD']:
        for m, s in [('U0_correct', 'U0 correct'), ('U1_no_reverse', 'U1 broken')]:
            labels.append(f'{s}\n({resp})')
            mod.append(d.loc[(resp, m), 'M2M_mod']); conv.append(d.loc[(resp, m), 'M2M_conv'])
            qm.append(d.loc[(resp, m), 'Q2_M']); qx.append(d.loc[(resp, m), 'Q2_X'])
    x = np.arange(4)
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax[0].bar(x-0.2, qm, 0.4, color=BLUE, label='$Q^2$  mechanistic block alone')
    ax[0].bar(x+0.2, qx, 0.4, color=ORANGE, label='$Q^2$  measured block alone')
    ax[0].set_ylim(0, 1.15); ax[0].set_ylabel('$Q^2$', fontsize=12)
    ax[0].legend(fontsize=9.5, loc='lower left')
    ax[0].set_title('Break the mechanism and the blocks swap roles', fontsize=12.5)

    c2 = [GOOD, BAD, GOOD, BAD]
    ax[1].bar(x-0.2, mod, 0.4, color=c2, label='modified M2M*')
    ax[1].bar(x+0.2, conv, 0.4, color=GREY, label='original M2M')
    ax[1].axhline(1.0, color='k', ls='--', lw=2.0)
    ax[1].text(3.45, 1.35, 'reference = 1', ha='right', fontsize=10, style='italic')
    ax[1].set_yscale('log'); ax[1].set_ylabel('ratio (log scale)', fontsize=12)
    ax[1].legend(fontsize=9.5, loc='upper center')
    ax[1].set_title('Correct above 1, broken below. Original: both above.', fontsize=12.5)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=9.5); a.grid(axis='y', alpha=0.3)
    fig.suptitle('Case study 2 (urethane) — true model U0 vs broken model U1', fontsize=13.5)
    fig.tight_layout(); fig.savefig('deck_cs2.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    d = cs1(); cs2()
    print(d[['M2M_current', 'CV_ratio_mean', 'CV_ratio_sd', 'CV_frac_gt1']].round(3).to_string())
    print('\nSaved deck_cs1_ratio.png, deck_cs1_vs.png, deck_cs1_uniques.png, deck_cs2.png')
