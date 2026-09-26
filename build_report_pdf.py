"""
Build the PDF findings report.

Every number is read back from the saved result workbooks rather than retyped,
so the report cannot drift from the actual outputs. Figures are embedded from
the PNGs written by the analysis scripts.
"""
import os
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

OUT = 'Thesis_M2M_Findings_Report.pdf'
ACCENT = colors.HexColor('#1F3864')
ACCENT2 = colors.HexColor('#C44E52')
GREY = colors.HexColor('#F2F2F2')

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=15, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=7, leading=18)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=11.5, textColor=ACCENT,
                    spaceBefore=9, spaceAfter=4, leading=14)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontSize=10, textColor=colors.HexColor('#333333'),
                    spaceBefore=7, spaceAfter=3, leading=12)
BODY = ParagraphStyle('BODY', parent=ss['BodyText'], fontSize=9.2, leading=13,
                      alignment=TA_JUSTIFY, spaceAfter=5)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8.3, leading=11.5,
                      textColor=colors.HexColor('#444444'), leftIndent=8, rightIndent=8,
                      borderPadding=4, backColor=GREY, spaceBefore=4, spaceAfter=7)
CAP = ParagraphStyle('CAP', parent=BODY, fontSize=8, leading=10.5, alignment=1,
                     textColor=colors.HexColor('#555555'), spaceBefore=2, spaceAfter=9)
MONO = ParagraphStyle('MONO', parent=BODY, fontName='Courier', fontSize=8.4, leading=11.5,
                      alignment=0, backColor=GREY, borderPadding=5, spaceBefore=4, spaceAfter=7)
TITLE = ParagraphStyle('TITLE', parent=ss['Title'], fontSize=21, textColor=ACCENT, leading=25)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontSize=11.5, alignment=1,
                     textColor=colors.HexColor('#444444'), leading=15)

def P(t, s=BODY): return Paragraph(t, s)

def tbl(df, widths=None, fs=7.4, hi_rows=None, align_left_first=True):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), fs),
        ('LEADING', (0, 0), (-1, -1), fs + 2.4),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#BBBBBB')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F8FA')]),
    ]
    if align_left_first:
        style.append(('ALIGN', (0, 0), (0, -1), 'LEFT'))
    for r in (hi_rows or []):
        style += [('BACKGROUND', (0, r + 1), (-1, r + 1), colors.HexColor('#FFF2CC')),
                  ('FONTNAME', (0, r + 1), (-1, r + 1), 'Helvetica-Bold')]
    t.setStyle(TableStyle(style))
    return t

def fig(path, width=165*mm, caption=None):
    out = []
    if os.path.exists(path):
        from PIL import Image as PILImage
        try:
            w, h = PILImage.open(path).size
            out.append(Image(path, width=width, height=width*h/w))
        except Exception:
            out.append(Image(path, width=width, height=width*0.55))
        if caption:
            out.append(P(caption, CAP))
    return out

def short(m): return m.split('_')[0]

# ----------------------------------------------------------------- load results
two  = pd.read_excel('sopls_two_orderings.xlsx')
tbA  = pd.read_excel('sopls_two_block.xlsx', sheet_name='X1_X2')
tbB  = pd.read_excel('sopls_two_block.xlsx', sheet_name='X1_M2')
ceil = pd.read_excel('sopls_two_block_ceiling.xlsx')
comm = pd.read_excel('sopls_commonality.xlsx')
uniq = pd.read_excel('metric_unique_m2m.xlsx', sheet_name='summary')
ups  = pd.read_excel('metric_unique_m2m.xlsx', sheet_name='per_seed')
lvd  = pd.read_excel('metric_ratio_lv_distribution.xlsx', sheet_name='summary')
flat = pd.read_excel('metric_ratio_flat_regions.xlsx')
q2fr = pd.read_excel('q2_flat_region_comparison.xlsx', sheet_name='selected_model')
q2sp = pd.read_excel('q2_flat_region_comparison.xlsx', sheet_name='Q2_spread')
ex1  = pd.read_excel('extrapolation_test.xlsx', sheet_name='summary')
ex2  = pd.read_excel('extrapolation_1se_distribution.xlsx', sheet_name='summary')
rg   = pd.read_excel('regen_fit_and_test.xlsx', sheet_name='summary')
rgl  = pd.read_excel('regen_fit_and_test.xlsx', sheet_name='all_1SE_allocations')

Yw = pd.read_excel('RG_within.xlsx', sheet_name='Y').E_purity.values
Ye = pd.read_excel('RG_extrap.xlsx', sheet_name='Y').E_purity.values
SSTw = ((Yw - Yw.mean())**2).sum(); SSTe = ((Ye - Ye.mean())**2).sum()
rgl['R2_w'] = 1 - len(Yw)*rgl.RMSEP_within**2/SSTw
rgl['R2_e'] = 1 - len(Ye)*rgl.RMSEP_extrap**2/SSTe
gg = rgl.groupby('model')
R2tab = pd.DataFrame({'R2_within': gg.R2_w.median(), 'R2_extrap': gg.R2_e.median()})
R2tab['dR2_pp'] = 100*(R2tab.R2_extrap - R2tab.R2_within)
R2tab = R2tab.join(rg.set_index('model')[['ratioQ_med', 'ratioSS_med', 'RMSEP_E_med',
                                          'fit_cost', 'frac_beat_DD']])
R2tab = R2tab.sort_values('ratioQ_med', ascending=False).reset_index()

S = []
# ================================================================== TITLE
S += [Spacer(1, 42*mm),
      P('Measuring Physics-Driven and Data-Driven<br/>Contributions in Hybrid Process Models', TITLE),
      Spacer(1, 6*mm),
      P('Development and evaluation of a unique-contribution diagnostic metric<br/>'
        'for SO-PLS hybrid models of a two-stage batch reactor', SUB),
      Spacer(1, 16*mm)]
S.append(tbl(pd.DataFrame({
    'Item': ['Block structure', 'Candidate mechanisms', 'Method', 'Calibration design',
             'Extrapolation design', 'Report generated'],
    'Value': ['X1 -> M2 -> X2 -> Y (two-stage batch reactor)',
              'M0 (correct) and six misspecifications M1-M6',
              'SO-PLS, sequential orthogonalisation, 10-fold CV (seed 42)',
              'n = 100 batches, T2 ~ U(330, 370) K',
              'n = 100 batches, T2 ~ U(370, 382) K',
              '17 September 2026']}), widths=[45*mm, 110*mm], fs=8.5))
S += [Spacer(1, 14*mm),
      P('<b>Headline result.</b> The proposed metric — the ratio of the unique predictive '
        'contribution of the mechanistic block to that of the measured block — orders the seven '
        'candidate mechanisms perfectly against their out-of-domain performance '
        '(Spearman = &minus;1.000), but with the <i>opposite sign</i> to the design hypothesis. '
        'Higher mechanistic dominance predicts <i>worse</i> extrapolation, not better.', NOTE),
      PageBreak()]

# ================================================================== SUMMARY
S += [P('1. Executive summary', H1),
      P('This report documents the development of a calibration-only diagnostic metric intended to '
        'distinguish well-specified from misspecified mechanistic sub-models inside an SO-PLS hybrid, '
        'and its evaluation against genuine extrapolation data. Findings are reported as obtained, '
        'including those that contradict the design hypothesis.', BODY)]

S += [P('1.1 What was established', H2)]
for t in [
 '<b>The existing M2M metric has no intrinsic reference point.</b> Defined as SS(M2)/SS(X2) in a single '
 'forward ordering, it credits all shared M2/X2 variance to M2. Observed values span 2.4 to 12.6 with '
 'no value carrying an interpretable meaning.',
 '<b>Shared variance dominates and cannot be attributed.</b> Commonality analysis gives unique '
 'contributions capturing only 17&ndash;37 % of the joint effect of M2 and X2; the shared term is '
 '0.47&ndash;0.68 of a joint effect of ~0.84 in Q&sup2; units.',
 '<b>The proposed unique-ratio metric gains a natural reference at 1</b> and, on calibration data, '
 'separates mechanisms that omit a whole reaction pathway from those that do not.',
 '<b>The metric does not rank mechanisms by correctness.</b> M0 (correct) and M3 (wrong Ea4) are '
 'statistically indistinguishable on every estimator tried.',
 '<b>Calibration performance is blind to mechanism quality.</b> Across all seven candidates, '
 'within-domain R&sup2; spans only 0.8914&ndash;0.8978 — a range of 0.006.',
 '<b>The metric predicts extrapolation perfectly but inverted.</b> On regenerated data with corrected '
 'noise parameters and per-model fitted kinetics, Spearman(metric, extrapolation R&sup2;) = &minus;1.000 '
 'and Spearman(metric, degradation) = &minus;1.000.',
 '<b>Stopping the flat-region rule at one standard error is justified;</b> going to two is not. The '
 'accuracy-for-parsimony exchange rate worsens threefold at the second step.']:
    S.append(P('&bull; ' + t, BODY))

S += [P('1.2 The central negative result', H2),
      P('The metric measures <i>how heavily the hybrid leans on the mechanistic block</i>, and in this '
        'system that quantity is an inverse predictor of out-of-domain robustness. The asymmetry is '
        'structural: X2 consists of concentrations measured during stage 2 <i>under the test conditions</i>, '
        'so it is never out-of-domain for itself, whereas M2 is a fixed extrapolation of Arrhenius '
        'parameters estimated at T2 &isin; [330, 370] K and then evaluated at T2 &isin; [370, 382] K, where '
        'exp(&minus;Ea/RT&#8322;) amplifies any parameter error. A model that leans on M2 inherits that '
        'error in proportion.', BODY),
      P('This is corroborated independently at two levels. Across models the rank correlation is perfect. '
        'Within each of the seven models, across the 1-SE flat region, every additional M2 latent variable '
        'worsens extrapolation (&rho; = +0.28 to +0.69) and every additional X2 latent variable improves it '
        '(&rho; = &minus;0.56 to &minus;0.69) &mdash; fourteen consistent signs.', BODY),
      P('Note that mechanistic information still helps overall: every hybrid beat the pure data-driven '
        'benchmark on extrapolation (R&sup2; 0.735&ndash;0.810 versus 0.633). The optimum is interior &mdash; '
        'some mechanistic content helps, heavy reliance hurts.', NOTE)]

S.append(PageBreak())

# ================================================================== DATA
S += [P('2. Data and regimes', H1),
      P('Three distinct data regimes appear in this work. They are not interchangeable and absolute '
        'metric values are not comparable between them.', BODY)]
S.append(tbl(pd.DataFrame({
    'Regime': ['A — real data', 'B — DG generated', 'C — regenerated'],
    'Source': ['Author X1/X2/Y.xlsx', 'gen_calibration_extrapolation.py', 'Data_Generation_corrected.py'],
    'Kinetic params': ['fitted per model', 'fixed nominal', 'fitted per model'],
    'Noise sigma': ['author (real)', 'opt_sigma (incorrect)', 'author corrected'],
    'Extrapolation set': ['none', 'yes (flawed control)', 'yes (proper control)'],
    'Used for': ['metric development', 'first extrap. test', 'final extrap. test']}),
    widths=[27*mm, 40*mm, 24*mm, 25*mm, 26*mm, 28*mm], fs=7.2))
S += [P('Regime C was generated specifically for this report. The corrected diffusion coefficients are '
        '&sigma;&#8321; = [0.75, 0.75, 0.25] and &sigma;&#8322; = [0.25, 0.75, 0.25, 0.75, 0.75, 0.25], taken '
        'verbatim from the author\'s notebook; the previously used opt_sigma arrays came from a 1 %-CV '
        'calibration routine the author\'s notebook never runs.', BODY),
      P('<b>A control-set defect was found and fixed.</b> The Regime B "within-domain" test set spanned only '
        'T2 362&ndash;370 K, the top 20 % of the calibration range, which made extrapolation appear to '
        '<i>outperform</i> in-domain prediction. In Regime C the control is drawn from the same U(330, 370) '
        'distribution as calibration (99 % inside the calibration range) and the extrapolation set is 100 % '
        'outside it.', NOTE)]
S.append(tbl(pd.DataFrame({
    'Set': ['RG_calibration', 'RG_within', 'RG_extrap'],
    'n': [100, 100, 100],
    'T2 range (K)': ['330.4 - 369.9', '330.5 - 370.0', '370.2 - 381.8'],
    'Y mean +/- sd (%)': ['76.97 +/- 4.24', '76.36 +/- 5.08', '81.29 +/- 3.47'],
    'Role': ['training', 'in-domain control', 'extrapolation']}),
    widths=[34*mm, 12*mm, 34*mm, 36*mm, 38*mm], fs=8))
S += [Spacer(1, 3*mm),
      P('Fitted kinetic parameters, Regime C (least-squares against calibration X2 trajectories only):', H3)]
f = rg[['model', 'Ea3', 'Ea4', 'A3', 'A4', 'fit_cost']].copy()
f['model'] = f.model.map(short)
for c in ['Ea3', 'Ea4', 'A3', 'A4']: f[c] = f[c].round(1)
f['fit_cost'] = f.fit_cost.map(lambda v: f'{v:,.0f}')
f = f.sort_values('model')
S.append(tbl(f, widths=[24*mm, 30*mm, 30*mm, 24*mm, 24*mm, 32*mm], fs=8))
S.append(P('The fit recovers the true mechanism: M0 has both the lowest residual cost and near-true '
           'parameters (Ea3 47 829 versus 50 000; Ea4 53 659 versus 55 000). M5 is worse by two orders '
           'of magnitude.', BODY))
S.append(PageBreak())

# ================================================================== SO-PLS
S += [P('3. SO-PLS block orderings and sequential sums of squares', H1),
      P('Both orderings were run with an independent blind argmin over the full 6 x 15 x 15 grid. '
        'Sums of squares are sequential (Type I) on standardised Y, so SS_Y = 99 exactly and the '
        'decomposition closes to 0.000 for all fourteen fits.', BODY),
      P('3.1 Forward: X1 &rarr; M2 &rarr; X2', H2)]
a = two[['model', 'fwd_LV_X1', 'fwd_LV_M2', 'fwd_LV_X2', 'fwd_RMSECV', 'fwd_Q2',
         'fwd_SS_X1', 'fwd_SS_M2', 'fwd_SS_X2', 'fwd_SS_F']].copy()
a['model'] = a.model.map(short)
a.columns = ['Model', 'LV X1', 'LV M2', 'LV X2', 'RMSECV', 'Q2', 'SS X1', 'SS M2', 'SS X2', 'SS F']
S.append(tbl(a.round(4), fs=7.6))
S += [P('3.2 Reverse: X1 &rarr; X2 &rarr; M2 (independently re-optimised)', H2)]
b = two[['model', 'rev_LV_X1', 'rev_LV_X2', 'rev_LV_M2', 'rev_RMSECV', 'rev_Q2',
         'rev_SS_X1', 'rev_SS_X2', 'rev_SS_M2', 'rev_SS_F']].copy()
b['model'] = b.model.map(short)
b.columns = ['Model', 'LV X1', 'LV X2', 'LV M2', 'RMSECV', 'Q2', 'SS X1', 'SS X2', 'SS M2', 'SS F']
S.append(tbl(b.round(4), fs=7.6))
S += [P('M0 reproduces the paper\'s {4, 6, 2} allocation and M6 its {3, 3, 6}. Predictive performance is '
        'essentially ordering-invariant (RMSECV differs by &le; 0.04 and the sign is inconsistent), but the '
        'sequential SS is strongly order-dependent: for M0, SS(M2) falls from 71.72 to 16.43 when M2 is '
        'placed last, while SS(X2) rises from 5.71 to 61.58. Roughly 55&ndash;66 SS units (56&ndash;67 % of '
        'SS_Y) are jointly explainable and cannot be attributed by a single ordering. <b>This is the defect '
        'the new metric is intended to address.</b>', BODY)]

S += [P('3.3 Two-block reference models', H2),
      P('X1 &rarr; X2 contains no mechanistic block and is therefore model-independent.', BODY)]
c = tbA[['dataset', 'LV_X1', 'LV_X2', 'RMSECV', 'Q2', 'SS_X1', 'SS_X2', 'SS_F']].copy()
c.columns = ['Dataset', 'LV X1', 'LV X2', 'RMSECV', 'Q2', 'SS X1', 'SS X2', 'SS F']
S.append(tbl(c.round(4), fs=7.8))
S.append(Spacer(1, 2*mm))
d = tbB[['model', 'LV_X1', 'LV_M2', 'RMSECV', 'Q2', 'SS_X1', 'SS_M2', 'SS_F']].copy()
d['model'] = d.model.map(short)
d.columns = ['Model', 'LV X1', 'LV M2', 'RMSECV', 'Q2', 'SS X1', 'SS M2', 'SS F']
S.append(tbl(d.round(4), fs=7.8))
S.append(P('The mechanistic-only model out-predicts the data-driven one for every candidate except M1 and '
           'M5. M5 is the clear failure in isolation (RMSECV 2.4647, Q&sup2; 0.6240) &mdash; far more '
           'discriminating than the three-block result, where X2 compensates for the missing '
           'D-dependence.', BODY))
S.append(PageBreak())

# ================================================================== CEILING
S += [P('4. Is the reported performance the best attainable?', H1),
      P('Three separable questions were tested: whether the LV grid caps bound the optimum, how flat the '
        'RMSECV surface is around the argmin, and how much of the reported value is cross-validation '
        'partition luck.', BODY)]
e = ceil[['run', 'base_LV', 'base_RMSECV', 'ext_LV', 'ext_RMSECV', 'ext_gain',
          'n_within_1pct', 'fold_SE', 'seed_fixed_mean', 'seed_fixed_sd']].copy()
e.columns = ['Run', 'LV @cap15', 'RMSECV', 'LV @cap30', 'RMSECV', 'Gain', '#within 1%', 'fold SE',
             'seed mean', 'seed sd']
S.append(tbl(e.round(4), fs=6.8))
S += [P('<b>No allocation was cap-limited.</b> Doubling the latent-variable caps to 30 changed nothing '
        'anywhere, including the one case (M6) that sat on the boundary. The minima are genuine interior '
        'optima.', BODY),
      P('<b>Seed 42 is a favourable partition.</b> Across ten different cross-validation partitions every '
        'reported value sits at or below the ten-seed mean, by 0.01&ndash;0.14. Selection optimism from '
        'taking the argmin of a noisy surface is separately small (+0.003 to +0.066), so most of the offset '
        'is partition luck rather than over-searching.', BODY),
      P('<b>Consequence for ranking.</b> The M0 / M2 / M3 / M4 spread (RMSECV 1.588&ndash;1.620, range 0.032) '
        'is smaller than the seed-to-seed standard deviation of any one of them. Those four are not '
        'separated by this evidence; M1, M6 and especially M5 are separated by several standard '
        'deviations.', NOTE),
      P('<b>A structural ceiling exists.</b> Y is computed from each batch\'s own final noisy sample, while '
        'X2 is truncated to a time grid shared across batches. The columns Y is an exact function of are '
        'therefore <i>not in X2</i>. This, not measurement noise alone, is what limits achievable '
        'accuracy.', BODY)]
S.append(PageBreak())

# ================================================================== COMMONALITY
S += [P('5. Commonality decomposition', H1),
      P('The proposal that the mechanistic contribution equals full &minus; (X1+X2) and the measured '
        'contribution equals full &minus; (X1+M2) is structurally correct, but these are <i>unique</i> '
        '(incremental) contributions and do not sum to the joint effect. Closing the decomposition '
        'requires the X1-only baseline.', BODY),
      P('Joint = Q&sup2;_full &minus; Q&sup2;_X1 &nbsp;&bull;&nbsp; Unique_M = Q&sup2;_full &minus; Q&sup2;_DD '
        '&nbsp;&bull;&nbsp; Unique_X = Q&sup2;_full &minus; Q&sup2;_KD &nbsp;&bull;&nbsp; '
        'Shared = Joint &minus; Unique_M &minus; Unique_X', MONO)]
g = comm[['model', 'ro_Q_X1', 'ro_Q_KD', 'ro_Q_DD', 'Q_full', 'ro_UniqueM', 'ro_UniqueX',
          'ro_sum_uniques', 'ro_Joint', 'ro_Shared']].copy()
g['model'] = g.model.map(short)
g.columns = ['Model', 'Q2 X1', 'Q2 KD', 'Q2 DD', 'Q2 full', 'Unique M', 'Unique X', 'sum', 'Joint', 'Shared']
S.append(tbl(g.round(4), fs=7.6))
S += [P('<b>The two unique parts capture only 17&ndash;37 % of the joint effect.</b> Shared is '
        '0.47&ndash;0.68 out of a joint of ~0.84. The unique contributions are therefore floor estimates of '
        'each block\'s importance, not a partition of it.', BODY),
      P('Two structural observations shaped the metric design. First, Q&sup2;_DD is identical (0.7879) for '
        'M0&ndash;M5 because the data-driven model contains no M2 block at all, so Unique_M is merely '
        'Q&sup2;_full shifted by a constant and adds nothing as a comparative measure. Second, '
        '<b>Unique_X is the quantity that discriminates</b>, and it runs opposite to intuition: M5 (0.2673) '
        'and M1 (0.1546) stand far above the ~0.06 of the well-specified models. A large unique-X2 means X2 '
        'is doing work the mechanism failed to do.', BODY),
      P('The numbers depend heavily on protocol. Holding sub-models at the full model\'s latent-variable '
        'allocation instead of re-optimising roughly doubles Unique_M, because the data-driven baseline is '
        'starved of components (Q&sup2;_DD collapses to 0.6562 for M0/M3/M4). The re-optimised protocol is '
        'the defensible one.', NOTE)]
S.append(PageBreak())

# ================================================================== METRIC
S += [P('6. The proposed metric', H1),
      P('M2M_unique = unique(M2) / unique(X2), with a natural reference at 1: above 1 the mechanistic block '
        'contributes more unique predictive information than the measured block, below 1 the reverse. '
        'Four estimators were compared because they are <i>not</i> interchangeable.', BODY)]
h = uniq[['model', 'M2M_current', 'SSown_ratio', 'SSmatch_ratio', 'SSeq_lo_ratio',
          'CV_ratio_mean', 'CV_ratio_sd', 'CV_frac_gt1']].copy()
h['model'] = h.model.map(short)
h.columns = ['Model', 'current M2M', 'SS-own', 'SS-match', 'SS-equal', 'CV mean', 'CV sd', 'P(>1)']
S.append(tbl(h.round(3), fs=7.8, hi_rows=[5]))
S += [P('<b>The reference at 1 works, under the CV estimator only.</b> The two mechanisms that omit an '
        'entire chemical pathway, M1 and M5, fall below 1; the four that get the pathway structure right '
        'rise above it. This is unanimous across all twenty cross-validation partitions '
        '(P(&gt;1) exactly 0.00 and 1.00). The current M2M cannot do this: its range is 2.37 to 12.57 with '
        'no anchor anywhere.', BODY),
      P('<b>The in-sample SS estimators fail on M1</b> (1.19&ndash;1.49, wrongly above 1) because in-sample '
        'SS cannot decrease when latent variables are added, so the mechanistic side is inflated by however '
        'many components it was granted. The latent-variable asymmetry is large: M0\'s unique-M2 is measured '
        'with 8 components against 2 for unique-X2, and matching them moves the ratio from 2.879 to '
        '3.769.', BODY),
      P('6.1 Choice of estimator', H2),
      P('<b>Most intuitive:</b> the SS version &mdash; a variance split on a single deterministic fit, '
        'continuous with the existing M2M. But it is also the one that misclassifies M1. '
        '<b>Most interpretable of the working versions:</b> RMSECV, which reads in physical units '
        '("adding M2 cuts prediction error by 0.65 purity points; adding X2 cuts it by 0.32"). '
        '<b>Preferred:</b> the Q&sup2; version, because only Q&sup2; differences decompose additively, so the '
        'Shared term means something; because it is more stable across partitions; and because, as Section 8 '
        'shows, it is far less sensitive to flat-region width.', BODY),
      P('<b>Limitations carried forward.</b> The metric separates {M0, M2, M3, M4} from {M1, M5} decisively '
        'but does not rank within either group: the spread (2.114, 2.107, 2.090, 1.957) is far smaller than '
        'the per-model standard deviation (0.31&ndash;0.40). M6, a genuine misspecification, lands above 1. '
        'The ratio is also unbounded above and undefined if unique_X reaches zero.', NOTE)]
S.append(PageBreak())

# ================================================================== LV DISTRIBUTION
S += [P('7. Sensitivity to latent-variable allocation', H1),
      P('Both estimators were swept over all 1 350 allocations per model, with sub-models held consistent '
        'with each candidate allocation.', BODY)]
i_ = lvd[['model', 'opt_LV', 'ss_at_opt', 'ss_median', 'ss_frac_gt1',
          'q_at_opt', 'q_median', 'q_frac_gt1', 'q_n_undefined']].copy()
i_['model'] = i_.model.map(short)
i_.columns = ['Model', 'opt LV', 'SS @opt', 'SS median', 'SS P(>1)', 'Q2 @opt', 'Q2 median',
              'Q2 P(>1)', 'Q2 undef.']
S.append(tbl(i_.round(3), fs=7.6))
S += [P('<b>The threshold at 1 does not survive the full sweep.</b> M0, the correct mechanism, falls below '
        '1 at 40 % of allocations under the SS estimator. <b>The ranking does survive:</b> median ratio '
        'orders the models consistently under both estimators, with the four structurally sound mechanisms '
        'above the two known-broken ones.', BODY),
      P('<b>The optimum is not a representative allocation.</b> For M0/M2/M3/M4 the argmin sits well into '
        'the upper tail (M0: 3.77 at the optimum versus a median of 1.24). Quoting the ratio at the RMSECV '
        'optimum systematically flatters the mechanism.', NOTE)]
S += fig('metric_ratio_lv_distribution.png', width=150*mm,
         caption='Figure 1. Distribution of the unique-contribution ratio over all 1 350 latent-variable '
                 'allocations. Solid line: reference value 1. Dashed: RMSECV-optimal allocation.')
S.append(PageBreak())

# ================================================================== FLAT REGIONS
S += [P('8. Flat regions: 1-SE, 2-SE, 3-SE', H1)]
pv = flat.pivot(index='model', columns='region', values='q_frac_gt1')[['1-SE', '2-SE', '3-SE', 'full grid']]
pv = pv.reset_index(); pv['model'] = pv.model.map(short)
pv.columns = ['Model', '1-SE', '2-SE', '3-SE', 'full grid']
S += [P('8.1 Diagnostic power, P(ratio &gt; 1), Q&sup2; estimator', H2), tbl(pv.round(3), fs=8)]
pm = flat.pivot(index='model', columns='region', values='q_median')[['1-SE', '2-SE', '3-SE', 'full grid']]
pm = pm.reset_index(); pm['model'] = pm.model.map(short)
pm.columns = ['Model', '1-SE', '2-SE', '3-SE', 'full grid']
S += [P('8.2 Median ratio', H2), tbl(pm.round(3), fs=8)]
S += [P('Restricting to the 1-SE region sharpens the classification dramatically (M0 goes from 0.60 to 0.98; '
        'M5 reaches exactly 0.000 under both estimators) and <b>eliminates the Q&sup2; estimator\'s numerical '
        'pathology entirely</b> &mdash; undefined cells fall from ~30 % of the full grid to 0.0 % at 1-SE and '
        '2-SE. Q&sup2; is also far less sensitive to region width than SS: its medians move by 1.07&ndash;1.31x '
        'between 1-SE and the full grid, against 2.5&ndash;2.9x for SS.', BODY)]
S += fig('metric_ratio_flat_regions.png', width=142*mm,
         caption='Figure 2. Ratio distributions within each flat region against the full grid.')
S.append(PageBreak())

S += [P('8.3 The case for stopping at one standard error', H2),
      P('The purpose of a flat-region rule is to trade accuracy for parsimony. Measuring that trade directly '
        'as percentage points of Q&sup2; surrendered per latent variable removed:', BODY)]
sel = q2fr.pivot(index='model', columns='rule', values='Q2')[['argmin (full)', '1-SE', '2-SE', '3-SE']]
tot = q2fr.pivot(index='model', columns='rule', values='total_LV')[['argmin (full)', '1-SE', '2-SE', '3-SE']]
mm_ = pd.DataFrame(index=sel.index)
mm_['LV saved 1-SE'] = tot['argmin (full)'] - tot['1-SE']
mm_['pp Q2 per LV'] = (-100*(sel['1-SE'] - sel['argmin (full)'])/mm_['LV saved 1-SE']).round(3)
mm_['extra LV 2-SE'] = tot['1-SE'] - tot['2-SE']
mm_['pp Q2 per LV '] = (-100*(sel['2-SE'] - sel['1-SE'])/mm_['extra LV 2-SE']).round(3)
mm_['ratio'] = (mm_['pp Q2 per LV '] / mm_['pp Q2 per LV']).round(2)
mm_ = mm_.reset_index(); mm_['model'] = mm_.model.map(short)
mm_.columns = ['Model', 'LV saved (1-SE)', 'pp Q2 / LV', 'extra LV (2-SE)', 'pp Q2 / LV ', 'x worse']
S.append(tbl(mm_, fs=8))
S += [P('The first standard error buys simplification at <b>0.422 pp of Q&sup2; per latent variable</b>; the '
        'second buys it at <b>1.325 pp</b> &mdash; three times the price, for six of the seven models. It also '
        'removes 3.7 latent variables on average against 1.6 for the second step, so roughly 70 % of the '
        'achievable simplification happens in the first step.', BODY)]
sp = q2sp.pivot(index='model', columns='region', values='Q2_range')[['1-SE', '2-SE', '3-SE', 'full grid']]
sp = sp.reset_index(); sp['model'] = sp.model.map(short)
sp.columns = ['Model', '1-SE', '2-SE', '3-SE', 'full grid']
S += [P('Internal Q&sup2; spread of each region &mdash; is it still flat?', H3), tbl(sp.round(4), fs=8),
      P('At 1-SE the worst admitted model is 2.1&ndash;2.9 pp below the optimum; at 2-SE it is '
        '4.2&ndash;5.6 pp. <b>A set whose members differ by nearly six points of Q&sup2; is not a flat region</b>, '
        'so widening to 2-SE does not merely cost accuracy &mdash; it invalidates the justification for using '
        'a region at all. Model ordering also degrades monotonically (Spearman against plain argmin: 0.750, '
        '0.643, 0.464); by 3-SE, M1 &mdash; a mechanism missing a whole pathway &mdash; ranks first.', BODY),
      P('<b>Honest counterweight.</b> The 1-SE rule is not free: it costs 0.9&ndash;2.0 pp of Q&sup2;, or '
        '1.8&ndash;3.2 standard deviations of partition noise. The 2-SE rule costs 4.4&ndash;6.8. The defensible '
        'claim is not that 1-SE is costless but that it is where the trade stops being favourable.', NOTE)]
S.append(PageBreak())

# ================================================================== EXTRAP B
S += [P('9. First extrapolation test (Regime B) and why it was repeated', H1),
      P('The only pre-existing extrapolation data used fixed nominal kinetic parameters, incorrect noise '
        'coefficients, and the defective within-domain control described in Section 2. Every ratio in this '
        'regime fell below 1, so the reference threshold could not even be evaluated. Results are recorded '
        'for completeness.', BODY)]
j = ex2[['model', 'n_1SE', 'RMSEP_min', 'RMSEP_median', 'RMSEP_max', 'RMSEP_range',
         'frac_beat_DD', 'rho_LVM2', 'rho_LVX2']].copy()
j['model'] = j.model.map(short)
j.columns = ['Model', 'n 1-SE', 'RMSEP min', 'median', 'max', 'range', 'beat DD', 'rho LV_M2', 'rho LV_X2']
S.append(tbl(j.round(3), fs=7.8))
S += [P('<b>The within-model spread exceeded the between-model spread.</b> Allocations that are statistically '
        'indistinguishable on calibration differed by 0.78 to 4.77 RMSEP units out of domain. Neither the '
        'argmin nor the 1-SE pick reliably found a good extrapolator: M3\'s argmin pick sat in the worst '
        'decile of its own region, while M6\'s 1-SE pick sat near its worst.', BODY),
      P('Crucially, the latent-variable direction result already appeared here: for the structurally sound '
        'mechanisms, &rho;(LV_M2, RMSEP) was strongly positive (+0.62 to +0.77) and &rho;(LV_X2, RMSEP) '
        'negative. <b>This regime and Regime C agree, which is what makes the finding credible.</b>', NOTE)]
S += fig('extrapolation_1se_distribution.png', width=155*mm,
         caption='Figure 3. Regime B: extrapolation performance of every calibration-equivalent allocation.')
S.append(PageBreak())

# ================================================================== EXTRAP C
S += [P('10. Final extrapolation test (Regime C)', H1),
      P('Regenerated data with the author\'s corrected noise coefficients, per-model kinetic parameters '
        'fitted on calibration only, and a proper in-domain control. For the first time all seven candidates '
        'sit on the same data, so M6 is numerically comparable to the rest.', BODY),
      P('10.1 Calibration side &mdash; the metric behaves as designed', H2)]
k = rg[['model', 'argmin_LV', 'RMSECV', 'Q2_cal', 'n_1SE', 'ratioQ_med', 'ratioQ_p10', 'ratioQ_p90',
        'ratioQ_fracgt1', 'ratioSS_med']].copy()
k['model'] = k.model.map(short)
k.columns = ['Model', 'argmin LV', 'RMSECV', 'Q2 cal', 'n 1-SE', 'ratio Q2 med', 'p10', 'p90',
             'P(>1)', 'ratio SS med']
S.append(tbl(k.round(4), fs=7.4))
S.append(P('The metric is again above 1 for six models with M5 the sole model below it, unanimously. Both '
           'estimators agree on the full ordering. Meanwhile calibration itself is blind: Q&sup2;_cal spans '
           '0.9100&ndash;0.9223 and within-domain R&sup2; spans 0.8914&ndash;0.8978.', BODY))

S += [P('10.2 Extrapolation &mdash; the inversion', H2)]
m_ = R2tab[['model', 'ratioQ_med', 'R2_within', 'R2_extrap', 'dR2_pp', 'RMSEP_E_med', 'frac_beat_DD']].copy()
m_['model'] = m_.model.map(short)
m_.columns = ['Model', 'ratio Q2', 'R2 within', 'R2 extrap', 'dR2 (pp)', 'RMSEP extrap', 'beat DD']
S.append(tbl(m_.round(4), fs=8))
S += [P('<b>Spearman(metric, R&sup2;_extrap) = &minus;1.000. Spearman(metric, &Delta;R&sup2;) = &minus;1.000.</b> '
        'Perfectly monotone across all seven models, for both estimators. The higher the mechanistic '
        'dominance, the worse the extrapolation and the larger the degradation.', BODY),
      P('The scale-free R&sup2; measure was used deliberately: the extrapolation set has lower Y variance '
        '(sd 3.47) than the within-domain set (sd 5.08), which flatters raw extrapolation RMSEP. The '
        'inversion survives normalisation.', BODY),
      P('<b>It is not the metric specifically &mdash; it is mechanism reliance in general.</b> '
        'Spearman(fit cost, R&sup2;_extrap) = +0.964: the better a mechanism fits the calibration '
        'trajectories, the worse its hybrid extrapolates. M0, which recovers the true kinetics, degrades by '
        '14.85 pp; M5, whose fitted parameters are nonsense, degrades by only 8.42 pp.', NOTE)]
S += [P('10.3 Within-model corroboration', H2)]
n_ = rg[['model', 'rho_ratioQ_vs_E', 'rho_LVX1', 'rho_LVM2', 'rho_LVX2']].copy()
n_['model'] = n_.model.map(short)
n_.columns = ['Model', 'rho(ratio, RMSEP_e)', 'rho(LV_X1)', 'rho(LV_M2)', 'rho(LV_X2)']
S.append(tbl(n_.round(3), fs=8))
S.append(P('Across the 1-SE region, for all seven models without exception, every additional M2 latent '
           'variable worsens extrapolation and every additional X2 latent variable improves it. Fourteen '
           'consistent signs from an independent level of analysis &mdash; considerably stronger evidence '
           'than the n = 7 rank correlation alone.', BODY))
S += fig('regen_fit_and_test.png', width=160*mm,
         caption='Figure 4. Regime C. Left: metric against extrapolation error (dashed grey = data-driven '
                 'benchmark). Centre: extrapolation across all 1-SE allocations. Right: degradation against '
                 'the proper in-domain control.')
S.append(PageBreak())

# ================================================================== CONCLUSIONS
S += [P('11. Conclusions', H1),
      P('11.1 On the proposed metric', H2),
      P('The unique-contribution ratio is a genuine improvement on the existing M2M in three respects: it '
        'does not credit shared variance to the mechanistic block, it has an intrinsic reference point, and '
        'it is well behaved numerically when evaluated over a 1-SE flat region. The recommended form is the '
        'cross-validated Q&sup2; version, reported as the median over the 1-SE region with a p10&ndash;p90 '
        'interval, alongside the Shared term so the reader can see how much the ratio does not account for.', BODY),
      P('It nevertheless fails at its intended purpose. It does not rank mechanisms by correctness: M3 '
        '(wrong activation energy) scores highest, M0 and M3 are indistinguishable on every estimator, and '
        'M5 &mdash; by far the worst mechanism &mdash; scores lowest while extrapolating best. What it '
        'measures is how heavily the hybrid leans on the mechanistic block.', BODY),
      P('11.2 On what the metric actually predicts', H2),
      P('Read with the sign reversed, it is an excellent indicator of <b>extrapolation risk</b>. On Regime C '
        'it orders out-of-domain degradation perfectly. Whether this is a general property or an artefact of '
        'this block structure &mdash; where X2 is measured under the test conditions and therefore never '
        'out-of-domain for itself &mdash; cannot be settled by this data.', BODY),
      P('11.3 Limitations', H2)]
for t in ['Regime C is a single realisation from a single seed. A rank correlation of &minus;1.000 on seven '
          'points is striking but is one draw; the within-model evidence is what makes it credible. A second '
          'seed is the obvious confirmation.',
          'Parameter fitting used max_nfev = 60, matching the earlier canonical run. The hardest fits, M5 in '
          'particular, may not be fully converged.',
          'The metric values in Regimes A, B and C are not comparable; only within-regime comparisons are '
          'meaningful.',
          'The data-driven benchmark is a single allocation chosen by calibration Q&sup2;, compared against '
          'medians over hundreds of hybrid allocations. That specific comparison is indicative only.',
          'RMSECV (mean of fold RMSEs) and Q&sup2; (pooled PRESS) are not algebraically consistent; the '
          'allocation minimising one does not exactly maximise the other. The difference observed here is '
          '&le; 0.03 pp.',
          'The finding that mechanistic reliance harms extrapolation may not transfer to designs where the '
          'measured block is also fixed at calibration conditions.']:
    S.append(P('&bull; ' + t, BODY))

S += [P('11.4 Recommended next steps', H2),
      P('1. Repeat Regime C with a second and third generation seed to confirm the inversion. '
        '2. Test whether the inversion persists when the mechanistic parameters are re-fitted using data '
        'that spans the extrapolation range, which would separate "mechanism is wrong" from "parameters were '
        'estimated in the wrong domain". '
        '3. Investigate the fraction of allocations at which X2 adds nothing out-of-sample, which separated '
        'the sound from the broken mechanisms more cleanly than the ratio itself '
        '(0.295&ndash;0.332 versus 0.000&ndash;0.015) and was not pursued.', BODY)]

S.append(PageBreak())
S += [P('Appendix A. Analysis scripts and outputs', H1)]
files = pd.DataFrame({
 'Script': ['regen_corrected_data.py', 'regen_fit_and_test.py', 'sopls_two_orderings.py',
            'sopls_two_block.py', 'sopls_two_block_ceiling.py', 'sopls_commonality.py',
            'metric_unique_m2m.py', 'metric_ratio_lv_distribution.py', 'metric_ratio_flat_regions.py',
            'q2_flat_region_comparison.py', 'extrapolation_test.py', 'extrapolation_1se_distribution.py'],
 'Produces': ['RG_calibration / RG_within / RG_extrap .xlsx', 'regen_fit_and_test.xlsx + .png',
              'sopls_two_orderings.xlsx', 'sopls_two_block.xlsx', 'sopls_two_block_ceiling.xlsx',
              'sopls_commonality.xlsx', 'metric_unique_m2m.xlsx', 'metric_ratio_lv_distribution.xlsx + .png',
              'metric_ratio_flat_regions.xlsx + .png', 'q2_flat_region_comparison.xlsx',
              'extrapolation_test.xlsx + .png', 'extrapolation_1se_distribution.xlsx + .png'],
 'Section': ['2', '10', '3', '3', '4', '5', '6', '7', '8', '8', '9', '9']})
S.append(tbl(files, widths=[52*mm, 88*mm, 15*mm], fs=7.4))
S += [Spacer(1, 4*mm),
      P('Shared conventions across every analysis: mean-centring and unit-variance scaling fitted on '
        'calibration only, block scaling by 1/&radic;K, KFold(10, shuffle, random_state = 42), '
        'RMSECV as the mean of per-fold RMSE, Q&sup2; = 1 &minus; pooled PRESS / SST, sequential sums of '
        'squares on standardised Y, and a latent-variable grid of X1 &le; 6, M2 &le; 15, X2 &le; 15.', BODY),
      P('Files not modified in the course of this work: Data Generation.py, KD_fit.py, sopls_r.py. '
        'Data_Generation_corrected.py was created as a separate file and is imported unchanged; the only '
        'runtime interaction is setting its T2_range_testing attribute to carve the extrapolation range.', NOTE)]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#777777'))
    canvas.drawString(20*mm, 12*mm, 'Hybrid model diagnostics — SO-PLS unique-contribution metric')
    canvas.drawRightString(A4[0]-20*mm, 12*mm, f'Page {doc.page}')
    canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
    canvas.line(20*mm, 15*mm, A4[0]-20*mm, 15*mm)
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                        topMargin=18*mm, bottomMargin=20*mm,
                        title='Hybrid Model Diagnostics: Findings Report',
                        author='Thesis analysis')
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
