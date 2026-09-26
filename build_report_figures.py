"""
Figures for the two-case-study report.

Every panel is drawn from a saved result workbook, never from retyped numbers,
so the figures cannot drift from the analyses that produced them.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BLUE, RED, GREY = '#1f4e9c', '#c0392b', '#9e9e9e'
GREEN, ORANGE = '#2e8b57', '#e08214'
SHORT = lambda s: s.replace('_correct', '').replace('_no_side', '').replace('_order1_D', '') \
                   .replace('_wrong_Ea4', '').replace('_lumped_EF', '').replace('_no_D', '') \
                   .replace('_author_mis', '').replace('_no_reverse', '')


def fig1_commonality():
    d = pd.read_excel('shared_vs_ranking.xlsx').set_index('model').sort_values('fit_cost')
    lab = [SHORT(m) for m in d.index]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(x, d.uM, 0.62, label='unique$_M$  (mechanistic only)', color=BLUE)
    ax.bar(x, d.shared, 0.62, bottom=d.uM, label='shared', color=GREY)
    ax.bar(x, d.uX, 0.62, bottom=d.uM + d.shared, label='unique$_X$  (measured only)', color=RED)
    for i, (m, r) in enumerate(d.iterrows()):
        ax.text(i, r.uM + r.shared + r.uX + 0.012, f'{r.ratio:.2f}', ha='center', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel('share of jointly explained $Q^2$')
    ax.set_title('Case study 1 — commonality split (models ordered by kinetic fit cost)',
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper left', ncol=3, framealpha=0.95)
    ax.set_ylim(0, 1.06); ax.grid(axis='y', alpha=0.3)
    ax.text(0.995, 0.06, 'value above each bar = modified M2M*', transform=ax.transAxes,
            ha='right', fontsize=8, style='italic', color='#555555')
    fig.tight_layout(); fig.savefig('rep_cs1_commonality.png', dpi=200); plt.close(fig)


def fig2_metric_vs_quality():
    d = pd.read_excel('shared_vs_ranking.xlsx').set_index('model')
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for a, (xc, xl, logx) in zip(ax, [('fit_cost', 'kinetic fit cost  (lower = better mechanism)', True),
                                      ('roll_R2e', 'rolling extrapolation $R^2$', False)]):
        s = spearmanr(d[xc], d.ratio)
        a.scatter(d[xc], d.ratio, s=58, color=BLUE, zorder=3, edgecolor='k', linewidth=0.5)
        for m, r in d.iterrows():
            a.annotate(SHORT(m), (r[xc], r.ratio), fontsize=7.5,
                       xytext=(4, 4), textcoords='offset points')
        a.axhline(1.0, color=RED, ls='--', lw=1.2)
        a.text(0.02, 0.93, r'$\rho$ = %+.3f   (p = %.3f)' % (s.correlation, s.pvalue),
               transform=a.transAxes, fontsize=9)
        if logx:
            a.set_xscale('log')
        a.set_xlabel(xl, fontsize=9); a.set_ylabel('modified M2M*  ($u_M/u_X$)', fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle('Case study 1 — does the modified metric track mechanism quality?', fontsize=10.5)
    fig.tight_layout(); fig.savefig('rep_cs1_metric_vs_quality.png', dpi=200); plt.close(fig)


def fig3_flat_region():
    d = pd.read_excel('metric_ratio_flat_regions.xlsx')
    regions = ['1-SE', '2-SE', '3-SE', 'full grid']
    models = sorted(d.model.unique())
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    w = 0.2
    for j, reg in enumerate(regions):
        g = d[d.region == reg].set_index('model').reindex(models)
        x = np.arange(len(models)) + (j - 1.5)*w
        lo = (g.q_median - g.q_p10).clip(lower=0)
        hi = (g.q_p90 - g.q_median).clip(lower=0)
        ax.errorbar(x, g.q_median, yerr=[lo, hi], fmt='o', ms=4.5, capsize=2.5,
                    lw=1.2, label=reg)
    ax.axhline(1.0, color=RED, ls='--', lw=1.2)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([SHORT(m) for m in models], fontsize=9)
    ax.set_ylabel('modified M2M*  (median, p10-p90)')
    ax.set_title('Case study 1 — stability of the ratio inside the flat region', fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=4); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig('rep_cs1_flat_region.png', dpi=200); plt.close(fig)


def fig4_q2_cost():
    d = pd.read_excel('q2_flat_region_comparison.xlsx')
    order = ['argmin (full)', '1-SE', '2-SE', '3-SE']   # argmin = unpenalised baseline
    order = [r for r in order if r in set(d.rule)]
    models = sorted(d.model.unique())
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for m in models:
        g = d[d.model == m].set_index('rule').reindex(order)
        ax[0].plot(range(len(order)), g.Q2, 'o-', ms=4, lw=1.3, label=SHORT(m))
        ax[1].plot(range(len(order)), g.total_LV, 'o-', ms=4, lw=1.3)
    for a in ax:
        a.axvline(1, color='#888888', ls=':', lw=1.0)
    for a, yl, ti in zip(ax, ['$Q^2$ at the selected allocation', 'total latent variables'],
                         ['predictive cost of widening the region',
                          'parsimony gained']):
        a.set_xticks(range(len(order))); a.set_xticklabels(order, fontsize=9)
        a.set_ylabel(yl, fontsize=9); a.set_title(ti, fontsize=10); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle('Case study 1 — why the 1-SE rule is the stopping point', fontsize=10.5)
    fig.tight_layout(); fig.savefig('rep_cs1_q2_cost.png', dpi=200); plt.close(fig)


def fig5_rolling():
    d = pd.read_excel('iterative_prediction.xlsx', sheet_name='rmsep_by_k')
    d = d[d.method == 'SO-PLS-offlineCal']
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for a, reg, ti in zip(ax, ['within', 'extrap'],
                          ['within the calibration domain', 'extrapolation set']):
        for m, g in d[d.regime == reg].groupby('model'):
            g = g.sort_values('k')
            a.plot(g.k, g.RMSEP, 'o-', ms=3.5, lw=1.3, label=SHORT(m))
        a.set_xlabel('decision point $k$ (sample index)', fontsize=9)
        a.set_title(ti, fontsize=10); a.grid(alpha=0.3)
    ax[0].set_ylabel('RMSEP of end-of-batch purity', fontsize=9)
    ax[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle('Case study 1 — rolling prediction as measurements accumulate', fontsize=10.5)
    fig.tight_layout(); fig.savefig('rep_cs1_rolling.png', dpi=200); plt.close(fig)


def fig6_urethane_m2m():
    d = pd.read_excel('urethane_m2m_blockvariants.xlsx')
    d = d[(d.variant == 'V1_static') & (d.param == 70.0)]
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.8))

    # (a) block-alone predictive ability
    for a_i, (resp, mk) in enumerate([('nC', 0), ('nD', 1)]):
        pass
    g = d.set_index(['response', 'model'])
    labels, qm, qx = [], [], []
    for resp in ['nC', 'nD']:
        for mod, sh in [('U0_correct', 'U0 correct'), ('U1_no_reverse', 'U1 broken')]:
            labels.append(f'{sh}\n({resp})'); qm.append(g.loc[(resp, mod), 'Q2_M'])
            qx.append(g.loc[(resp, mod), 'Q2_X'])
    x = np.arange(len(labels))
    ax[0].bar(x - 0.2, qm, 0.4, label='$Q^2$ mechanistic block alone', color=BLUE)
    ax[0].bar(x + 0.2, qx, 0.4, label='$Q^2$ measured block alone', color=RED)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel('$Q^2$'); ax[0].set_ylim(0, 1.09); ax[0].grid(axis='y', alpha=0.3)
    ax[0].legend(fontsize=7.5, loc='lower left'); ax[0].set_title('(a) each block alone', fontsize=10)

    # (b) commonality split
    uM = [g.loc[(r, m), 'uM'] for r in ['nC', 'nD'] for m in ['U0_correct', 'U1_no_reverse']]
    uX = [g.loc[(r, m), 'uX'] for r in ['nC', 'nD'] for m in ['U0_correct', 'U1_no_reverse']]
    sh_ = [g.loc[(r, m), 'shared'] for r in ['nC', 'nD'] for m in ['U0_correct', 'U1_no_reverse']]
    ax[1].bar(x, uM, 0.6, label='unique$_M$', color=BLUE)
    ax[1].bar(x, sh_, 0.6, bottom=uM, label='shared', color=GREY)
    ax[1].bar(x, uX, 0.6, bottom=np.array(uM) + np.array(sh_), label='unique$_X$', color=RED)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel('share of jointly explained $Q^2$')
    ax[1].legend(fontsize=7.5, loc='upper center'); ax[1].grid(axis='y', alpha=0.3)
    ax[1].set_ylim(0, 1.18); ax[1].set_title('(b) commonality split', fontsize=10)

    # (c) the ratio, log scale, reference at 1
    mod = [g.loc[(r, m), 'M2M_mod'] for r in ['nC', 'nD'] for m in ['U0_correct', 'U1_no_reverse']]
    conv = [g.loc[(r, m), 'M2M_conv'] for r in ['nC', 'nD'] for m in ['U0_correct', 'U1_no_reverse']]
    ax[2].bar(x - 0.2, mod, 0.4, label='modified  M2M* = $u_M/u_X$', color=BLUE)
    ax[2].bar(x + 0.2, conv, 0.4, label='conventional  $SS_M/SS_X$', color=ORANGE)
    ax[2].axhline(1.0, color=RED, ls='--', lw=1.4)
    ax[2].text(len(x) - 0.45, 1.25, 'reference = 1', color=RED, fontsize=8, ha='right')
    ax[2].set_yscale('log'); ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylabel('ratio (log scale)'); ax[2].grid(axis='y', alpha=0.3, which='both')
    ax[2].legend(fontsize=7.5, loc='upper right'); ax[2].set_title('(c) the two metrics', fontsize=10)

    fig.suptitle('Case study 2 (urethane) — true model U0 vs broken model U1, '
                 'static layout, shared grid 0-70 h', fontsize=10.5)
    fig.tight_layout(); fig.savefig('rep_cs2_m2m.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    for fn in [fig1_commonality, fig2_metric_vs_quality, fig3_flat_region,
               fig4_q2_cost, fig5_rolling, fig6_urethane_m2m]:
        try:
            fn(); print('ok  ', fn.__name__)
        except Exception as e:
            print('FAIL', fn.__name__, type(e).__name__, e)
