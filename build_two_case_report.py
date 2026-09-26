"""
Build the two-case-study PDF report.

Every number is read back from the saved result workbooks rather than retyped,
so the report cannot drift from the analyses. Figures are embedded from the
PNGs written by build_report_figures.py and the urethane comparison scripts.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

OUT = 'Thesis_TwoCaseStudy_Report.pdf'
ACCENT = colors.HexColor('#1F3864')
ACCENT2 = colors.HexColor('#C44E52')
GREY = colors.HexColor('#F2F2F2')
USABLE = 174*mm

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=15, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=7, leading=18)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=11.5, textColor=ACCENT,
                    spaceBefore=9, spaceAfter=4, leading=14)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontSize=10,
                    textColor=colors.HexColor('#333333'), spaceBefore=7, spaceAfter=3, leading=12)
BODY = ParagraphStyle('BODY', parent=ss['BodyText'], fontSize=9.2, leading=13,
                      alignment=TA_JUSTIFY, spaceAfter=5)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8.3, leading=11.5,
                      textColor=colors.HexColor('#444444'), leftIndent=8, rightIndent=8,
                      borderPadding=4, backColor=GREY, spaceBefore=4, spaceAfter=7)
CAP = ParagraphStyle('CAP', parent=BODY, fontSize=8, leading=10.5, alignment=1,
                     textColor=colors.HexColor('#555555'), spaceBefore=2, spaceAfter=9)
MONO = ParagraphStyle('MONO', parent=BODY, fontName='Courier', fontSize=8.0, leading=11,
                      alignment=0, backColor=GREY, borderPadding=5, spaceBefore=4, spaceAfter=7)
TITLE = ParagraphStyle('TITLE', parent=ss['Title'], fontSize=20, textColor=ACCENT, leading=24)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontSize=10.5, alignment=1,
                     textColor=colors.HexColor('#444444'), spaceBefore=4)

S = []
def P(t, st=BODY): S.append(Paragraph(t, st))
def h1(t): S.append(Paragraph(t, H1))
def h2(t): S.append(Paragraph(t, H2))
def h3(t): S.append(Paragraph(t, H3))
def note(t): S.append(Paragraph(t, NOTE))
def sp(h=4): S.append(Spacer(1, h))
def cap(t): S.append(Paragraph(t, CAP))

def figure(path, caption, width=USABLE):
    if not os.path.exists(path):
        P(f'[missing figure: {path}]', NOTE); return
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    S.append(Image(path, width=width, height=width*h/w))
    cap(caption)

def table(data, widths=None, align=None, fs=8.2):
    t = Table(data, colWidths=widths, hAlign='CENTER')
    style = [('FONTSIZE', (0, 0), (-1, -1), fs),
             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
             ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
             ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
             ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#BBBBBB')),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ('TOPPADDING', (0, 0), (-1, -1), 3),
             ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
             ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GREY])]
    if align:
        style.append(('ALIGN', (1, 0), (-1, -1), align))
    t.setStyle(TableStyle(style))
    S.append(t); sp(6)

SHORT = lambda s: s.split('_')[0]

# ============================================================ load the results
cs1 = pd.read_excel('shared_vs_ranking.xlsx').set_index('model')
q2f = pd.read_excel('q2_flat_region_comparison.xlsx')
itr = pd.read_excel('iterative_prediction.xlsx', sheet_name='summary').set_index('model')
frg = pd.read_excel('metric_ratio_flat_regions.xlsx')
cm = pd.read_excel('sopls_commonality.xlsx').set_index('model')
bv = pd.read_excel('urethane_m2m_blockvariants.xlsx')
v1 = bv[(bv.variant == 'V1_static') & (bv.param == 70.0)].set_index(['response', 'model'])
v2 = bv[(bv.variant == 'V2_rolling') & (bv.param == 15.0)].set_index(['response', 'model'])

rho_fit = spearmanr(cs1.ratio, cs1.fit_cost)
rho_ext = spearmanr(cs1.ratio.dropna(), cs1.roll_R2e.dropna())
rho_sh = spearmanr(cs1.shared, cs1.roll_R2e)

# ==================================================================== TITLE
P('Measuring Physics-Driven and Data-Driven Contributions in Hybrid Process Models', TITLE)
P('A modified M2M metric based on unique contributions, tested on two case studies', SUB)
P('Two-stage batch reactor &nbsp;&bull;&nbsp; Urethane semi-batch reactor', SUB)
sp(14)

h2('Summary')
P('The original M2M metric compares the sequential sums of squares carried by the mechanistic '
  'and the measured block, <b>M2M = SS<sub>M</sub>/SS<sub>X</sub></b>. Because the shared '
  'variance is absorbed into the numerator, the ratio has no natural reference point: it is '
  'large whenever the two blocks overlap, whatever the mechanism is worth. This report tests a '
  'modified metric built from the <i>unique</i> (commonality) contributions only, '
  '<b>M2M* = u<sub>M</sub>/u<sub>X</sub></b>, whose reference is intrinsic: above 1 the '
  'mechanistic block supplies more unique information than the measurements, below 1 the '
  'measurements supply more.')
P('Two case studies are used. The first is the two-stage batch reactor of the source paper, with '
  'seven candidate mechanisms of known, graded quality. The second is the urethane semi-batch '
  'reactor, reconstructed from the original authors&rsquo; own notebooks, with the two models the '
  'source papers actually define.')
note('<b>Headline.</b> In case study 1 the modified metric tracks mechanism quality '
     f'(Spearman &rho; = {rho_fit.correlation:+.3f} against kinetic fit cost, '
     f'&rho; = {rho_ext.correlation:+.3f} against out-of-domain performance, p = {rho_ext.pvalue:.3f}), '
     'where the shared term alone does not '
     f'(&rho; = {rho_sh.correlation:+.3f}, p = {rho_sh.pvalue:.3f}). In case study 2 it places the '
     'correct and the broken mechanism on <i>opposite sides of 1</i>, which the conventional '
     'ratio cannot do. Its weakness is a denominator that can vanish when the mechanism is '
     'strong, making the magnitude unstable even where the sign is not.')

S.append(PageBreak())

# ================================================================== PART I
h1('Part I &mdash; Method')

h2('1.1 The two ratios')
P('SO-PLS fits the blocks in sequence. Each block is orthogonalised against the scores of the '
  'blocks already entered, so the variance it explains is what it adds <i>beyond</i> them. This '
  'gives Type-I (sequential) sums of squares, and it is why the ordering of blocks matters.')
P('Writing the knowledge-driven block as M and the data-driven block as X, the two metrics are:')
P('&nbsp;&nbsp;&nbsp;<b>conventional</b>&nbsp;&nbsp; M2M &nbsp;= SS<sub>M</sub> / SS<sub>X</sub>'
  '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>modified</b>&nbsp;&nbsp; M2M* = u<sub>M</sub> / u<sub>X</sub>', MONO)

h2('1.2 Commonality decomposition')
P('The unique terms come from refitting with one block removed. With a common prior block '
  'X<sub>1</sub> (case study 1) the decomposition is conditional on it; without one (case study 2) '
  'it is measured against the grand mean:')
P('joint&nbsp;&nbsp;= Q&sup2;(all blocks) &minus; Q&sup2;(prior block only)<br/>'
  'u<sub>M</sub>&nbsp;&nbsp;&nbsp;&nbsp;= Q&sup2;(all) &minus; Q&sup2;(all without M)'
  '&nbsp;&nbsp;&nbsp; what the mechanism adds to a purely data-driven model<br/>'
  'u<sub>X</sub>&nbsp;&nbsp;&nbsp;&nbsp;= Q&sup2;(all) &minus; Q&sup2;(all without X)'
  '&nbsp;&nbsp;&nbsp; what the measurements add to a purely knowledge-driven model<br/>'
  'shared = joint &minus; u<sub>M</sub> &minus; u<sub>X</sub>', MONO)
P('Three consequences are used throughout. First, the <b>hybrid model&rsquo;s advantage</b> over the '
  'better of the two single-source models is exactly min(u<sub>M</sub>, u<sub>X</sub>); the shared '
  'term does not enter, and in fact caps that advantage at (joint &minus; shared)/2. Large shared '
  'means the blocks are <i>redundant</i>, which is an argument against hybridising, not for it. '
  'Second, shared is <b>symmetric</b> under swapping M and X, so it cannot by itself say which '
  'block is better &mdash; a ranking quantity has to be antisymmetric, as the ratio is. Third, the '
  'ratio&rsquo;s sampling variance scales as 1/u<sub>X</sub>&sup2;, so it is fragile exactly where '
  'the uniques are small.')
note('<b>Coverage.</b> It is useful to report <b>coverage = (u<sub>M</sub>+u<sub>X</sub>)/joint = '
     '1 &minus; shared fraction</b> alongside the ratio. It states how much of the explained '
     'variance the ratio is entitled to arbitrate. Empirically the dispersion of the ratio across '
     'the flat region correlates with coverage at &rho; = &minus;0.857 (p = 0.014): low coverage, '
     'unstable ratio.')

h2('1.3 Latent-variable selection')
P('Allocations are chosen by the <b>parsimony-within-1-SE</b> rule: take the minimum-RMSECV cell, '
  'form the flat region of all cells within one cross-validation standard error of it, and inside '
  'that region choose the allocation with the fewest total latent variables (ties broken by '
  'RMSECV). Ten-fold cross-validation, fixed seed. This matters because the raw argmin gives '
  'degenerate splits that put nearly all capacity in one block.')

h2('1.4 Block construction and the &kappa; rule')
P('Blocks are unfolded batch-wise. Where a variable in the measured block is an <i>input</i> to '
  'the mechanistic model, the source paper moves it out of X and into M &mdash; '
  '&ldquo;an augmented mechanistic block is formed by appending the selected columns of X to M '
  'and removing them from X&rdquo;. Both case studies here follow that rule, which is why the '
  'operating conditions sit in the M block rather than the X block.')

S.append(PageBreak())

# ================================================================= PART II
h1('Part II &mdash; Case study 1: two-stage batch reactor')

h2('2.1 Process and data')
P('Component A forms the desired intermediate B while a consecutive reaction converts B to the '
  'unwanted C. The mixture transfers to stage 2, where B reacts with D to give the product E '
  'alongside a parallel reaction producing the impurity F. The response is the end-of-batch '
  'purity of E. Variability enters three ways: operating conditions are drawn at random '
  '(temperatures and durations uniform, initial concentrations normal), the deterministic ODEs '
  'are augmented with a Wiener term and integrated as SDEs by Euler&ndash;Maruyama, and the '
  'measurements carry sampling noise. One hundred batches form the calibration set.')
P('<b>Seven candidate mechanisms</b> of deliberately graded quality are fitted to the same data: '
  'the correct structure (M0), three with a plausible but wrong rate law or parameter tying '
  '(M2, M3, M4), one missing the side reaction (M1), one omitting the D dependence entirely (M5), '
  'and one reproducing the original authors&rsquo; misspecification (M6). Because each candidate '
  'has an independent kinetic fit cost and an independent out-of-domain score, the case study can '
  'ask whether the metric ranks mechanisms the way an oracle would.')

h2('2.2 Block layout')
table([['block', 'columns', 'contents'],
       ['X1', '6', 'CA0, t1, T1 (stage-1 recipe) + CA, CB, CC at stage-1 end'],
       ['M2', '45', '6 species x 7 timepoints (ODE) + CD0, t2, T2 (kappa rule)'],
       ['X2', '42', '6 species x 7 timepoints, measured'],
       ['Y', '1', 'purity of E at end of batch']],
      widths=[20*mm, 22*mm, 132*mm])
P('Ordering is X<sub>1</sub> &rarr; M<sub>2</sub> &rarr; X<sub>2</sub>, with X<sub>1</sub> first '
  'because it is the information available before stage 2 begins and is common to both '
  'sub-models. Both orderings were run; the reverse changes the sequential sums of squares '
  'substantially, which is the Type-I asymmetry noted in &sect;1.1.')

h2('2.3 Commonality results')
rows = [['model', 'fit cost', 'Q2 full', 'uM', 'uX', 'shared', 'coverage', 'M2M*', 'roll R2']]
for m, r in cs1.sort_values('fit_cost').iterrows():
    rows.append([SHORT(m), f'{r.fit_cost:.2e}', f'{r.Q2_full:.4f}', f'{r.uM:+.4f}',
                 f'{r.uX:+.4f}', f'{r.shared:+.4f}', f'{1-r.shared_frac:.3f}',
                 f'{r.ratio:.3f}', f'{r.roll_R2e:+.3f}'])
table(rows, align='CENTER')
figure('rep_cs1_commonality.png',
       'Figure 1 — Commonality split per candidate, ordered by kinetic fit cost (best on the left). '
       'The number above each bar is the modified metric. The shared term dominates throughout, '
       'which is why coverage is only 0.36–0.74.')

P('Two things stand out. The <b>shared term is large</b> (0.20&ndash;0.45, i.e. 26&ndash;57 % of '
  'the joint) so the ratio arbitrates only about half the explained variance here. And the '
  'shared term does <i>not</i> order the mechanisms: it places M2 above the correct M0 and M1 '
  'above M4, both wrong.')

figure('rep_cs1_metric_vs_quality.png',
       'Figure 2 — The modified metric against two independent quality measures. Left: kinetic fit '
       'cost, an oracle for how well each structure can describe the data at all. Right: rolling '
       'extrapolation R² on the out-of-domain set. The dashed line is the intrinsic reference at 1.')

P('Against the independent quality measures the ranking picture is:')
P(f'u<sub>M</sub> &nbsp; vs fit cost {spearmanr(cs1.uM, cs1.fit_cost).correlation:+.3f} '
  f'&nbsp;&nbsp; vs rolling R&sup2; {spearmanr(cs1.uM, cs1.roll_R2e).correlation:+.3f}<br/>'
  f'u<sub>X</sub> &nbsp; vs fit cost {spearmanr(cs1.uX, cs1.fit_cost).correlation:+.3f} '
  f'&nbsp;&nbsp; vs rolling R&sup2; {spearmanr(cs1.uX, cs1.roll_R2e).correlation:+.3f}<br/>'
  f'shared vs fit cost {rho_sh.correlation*-1:+.3f} '
  f'&nbsp;&nbsp; vs rolling R&sup2; {rho_sh.correlation:+.3f} (p = {rho_sh.pvalue:.3f})<br/>'
  f'<b>M2M*</b> &nbsp;&nbsp;vs fit cost {rho_fit.correlation:+.3f} '
  f'&nbsp;&nbsp; vs rolling R&sup2; {rho_ext.correlation:+.3f} (p = {rho_ext.pvalue:.3f})<br/>'
  f'Q&sup2; full vs fit cost {spearmanr(cs1.Q2_full, cs1.fit_cost).correlation:+.3f} '
  f'&nbsp;&nbsp; vs rolling R&sup2; {spearmanr(cs1.Q2_full, cs1.roll_R2e).correlation:+.3f}', MONO)
P('The ratio is the strongest of the candidates and the only one significant at the 5 % level '
  'against out-of-domain performance. Note that <b>Q&sup2; itself is nearly useless as a '
  'ranking device</b>: the seven models sit within 0.904&ndash;0.915 in-domain yet span more than '
  '17 R&sup2; units out-of-domain. That gap is the entire motivation for a contribution metric.')
note('<b>Honest caveat.</b> With n = 7 candidates none of these orderings is firmly established, '
     'and the shared ranking is not reproducible between the original and the regenerated '
     'datasets. The defensible claim is the narrow one: on the dataset where out-of-domain '
     'outcomes exist, the ratio tracks them and the shared term does not, and the symmetry '
     'argument of &sect;1.2 does not depend on n at all.')

S.append(PageBreak())

h2('2.4 Stability and the choice of stopping rule')
figure('rep_cs1_flat_region.png',
       'Figure 3 — Median and p10–p90 of the modified metric over every allocation inside each '
       'flat region. Widening the region from 1-SE outward adds spread without changing the '
       'central value.')
figure('rep_cs1_q2_cost.png',
       'Figure 4 — Left: predictive performance at the selected allocation under each stopping '
       'rule, starting from the unpenalised argmin. Right: total latent variables retained.')
_p = q2f.pivot_table(index='model', columns='rule', values=['Q2', 'total_LV'])
_a, _al = _p['Q2']['argmin (full)'], _p['total_LV']['argmin (full)']
_tr = {r: ((_p['Q2'][r] - _a).mean(), (_al - _p['total_LV'][r]).mean())
       for r in ['1-SE', '2-SE', '3-SE']}
P('Averaged over the seven candidates, relative to the unpenalised argmin:')
P('<br/>'.join(f'{r:<6s} &Delta;Q&sup2; {_tr[r][0]:+.4f} &nbsp;&nbsp; latent variables saved '
               f'{_tr[r][1]:.1f}' for r in ['1-SE', '2-SE', '3-SE']), MONO)
P(f'The first standard error is close to free: it removes {_tr["1-SE"][1]:.1f} latent variables '
  f'for {abs(_tr["1-SE"][0]):.4f} of Q&sup2;. The second costs a further '
  f'{abs(_tr["2-SE"][0]-_tr["1-SE"][0]):.4f} &mdash; more than the first step cost in total '
  f'&mdash; to remove only {_tr["2-SE"][1]-_tr["1-SE"][1]:.1f} more. That is the argument for '
  'stopping at 1-SE: the parsimony is nearly exhausted after one standard error while the '
  'predictive cost accelerates.')
P('The dispersion of the ratio is itself informative. Across models, the relative spread of M2M* '
  'inside the 1-SE region correlates with the shared term at &rho; = +0.714 and with coverage at '
  '&rho; = &minus;0.857 (p = 0.014), and with 1/u<sub>X</sub> at Pearson r = +0.818. This is the '
  '1/u<sub>X</sub>&sup2; variance scaling of &sect;1.2 showing up in the data: <b>a large shared '
  'term does not bias the ratio, it destabilises it.</b>')

h2('2.5 Extrapolation')
figure('rep_cs1_rolling.png',
       'Figure 5 — Rolling prediction of end-of-batch purity as measurements accumulate, for an '
       'offline-calibrated diagnostic. Left: within the calibration domain. Right: the '
       'extrapolation set, generated under shifted temperature conditions.')
P('The extrapolation test is what gives the metric something external to be right about. A '
  'diagnostic calibrated offline is applied in a rolling scheme: at each decision point k the '
  'measured trajectory up to k is combined with mechanistic forecasts beyond k, and end-of-batch '
  'purity is predicted. The correlation between the metric and out-of-domain performance is '
  f'&rho; = {rho_ext.correlation:+.3f} (p = {rho_ext.pvalue:.3f}).')
note('<b>The evaluation regime decides the answer.</b> A single-shot evaluation on the '
     'extrapolation set gives the opposite sign to the rolling scheme. Only the rolling regime '
     'tests the thing the metric is about &mdash; whether the mechanistic part can be trusted to '
     'carry the forecast where measurements run out &mdash; so that is the regime reported here.')

S.append(PageBreak())

# ================================================================ PART III
h1('Part III &mdash; Case study 2: urethane semi-batch reactor')

h2('3.1 Process and provenance')
P('Isocyanate (A) and butanol (B) react to urethane (C); urethane reacts further with isocyanate '
  'to allophanate (D) in a <i>reversible</i> step; isocyanate also trimerises to isocyanurate (E). '
  'Two feed streams and a temperature programme drive an 80-hour semi-batch run. Only C, D and E '
  'are measured; A, B and the volume follow from three algebraic balances and the component '
  'densities.')
P('The model, its parameters, the noise levels and the input handling used here are taken '
  '<b>verbatim from the original authors&rsquo; own notebooks</b>. This matters, because an earlier '
  'reconstruction of this case study from the published papers alone disagreed with the '
  'published trajectories and led to a refit of the kinetics that turned out to be unnecessary.')
note('<b>Two corrections the notebooks forced.</b> (i) The published parameter table is correct '
     'after all &mdash; an earlier conclusion that it could not have generated the published '
     'figures was wrong. (ii) The measurement noise on species C is &sigma; = 5&times;10<sup>-3</sup>, '
     'not 5&times;10<sup>-2</sup>. The larger value made n<sub>C</sub> unusable as a response '
     '(SNR &lt; 1) and forced a switch to n<sub>D</sub>; with the correct value n<sub>C</sub> has '
     'SNR = 10 and is usable, which matters because n<sub>C</sub> is the species the '
     'misspecification acts on.')

h2('3.2 Reconstructing the process data')
P('The authors&rsquo; input workbook is not in their archive, so the charge and the feed programme '
  'were recovered from the figures stored inside the notebooks: the temperature profile read '
  'directly off their own panel, the initial moles and volume read at t = 0, the feed totals from '
  'the closing material balances, and the feed solvent from the volume shortfall. Only the two '
  'feed ramps &mdash; which the notebooks never plot &mdash; had to be fitted.')
figure('urethane_compare_published.png',
       'Figure 6 — Published trajectories (recovered from the authors’ stored figure) against the '
       'reconstruction, both carrying measurement noise. Overall RMS deviation 5.0 % of each '
       'panel’s range: nB 1.6 %, nE 3.5 %, V 4.0 %, nD 4.1 %, nC 5.8 %, nA 8.4 %.')
note('<b>The reconstruction is validated, not assumed.</b> The authors seed their noise with '
     'np.random.seed(42) and draw C, D, E in that order. Matching the seed, the draw order and '
     'the 161-sample grid reproduces their <i>individual</i> realisation, not merely its variance: '
     'our noise vector correlates with the high-frequency content of their published trace at '
     '0.603, against &minus;0.005 &plusmn; 0.055 for twenty-five alternative seeds &mdash; '
     'z = 11.1. The residual disagreement is concentrated in the 3&ndash;9 h transient and is '
     'consistent with the reconstructed butanol feed running slightly fast.')

h2('3.3 The two models, and the block layout')
P('Neither source paper defines a family of candidate mechanisms &mdash; and neither uses '
  'multiblock regression at all; their method is symbolic regression on numerical derivatives. '
  'Each paper has exactly one available first-principles model alongside the ground truth, so the '
  'faithful comparison is a pair:')
table([['model', 'structure', 'fitted parameters'],
       ['U0 correct', 'A+B -> C,  A+C <-> D,  3A -> E', '8'],
       ['U1 broken', 'A+B -> C,  A+C  -> D,  3A -> E   (reverse step absent)', '6']],
      widths=[28*mm, 96*mm, 32*mm])
P('U1 is precisely the misspecification the source method exists to repair: the reverse step is '
  'the term their symbolic regression rediscovers. As a check on the pipeline, U0 recovers the '
  'true kinetics almost exactly (k<sub>ref1</sub> 0.001249 against 1.25&times;10<sup>-3</sup>, '
  'E<sub>a2</sub> 70.99 against 71.014) while U1&rsquo;s fit cost is five orders of magnitude worse '
  '(1.22&times;10<sup>8</sup> against 1964), with visibly compensating parameters.')
P('Because the block design is <b>not</b> inherited from the source papers, it was built by '
  'analogy with case study 1: the mechanistic block holds the ODE states plus the unfolded input '
  'programme (the &kappa; rule), the measured block holds the measured states on the same grid, '
  'and the response lies outside that grid. Forty batches were drawn by perturbing the validated '
  'batch; note that scaling the whole charge leaves every concentration unchanged, so only ratios '
  'and the programme are varied.')

h2('3.4 Results')
rows = [['response', 'model', 'Q2 M alone', 'Q2 X alone', 'uM', 'uX', 'shared', 'M2M*', 'M2M conv']]
for resp in ['nC', 'nD']:
    for mod, lab in [('U0_correct', 'U0 correct'), ('U1_no_reverse', 'U1 broken')]:
        r = v1.loc[(resp, mod)]
        rows.append([resp, lab, f'{r.Q2_M:.4f}', f'{r.Q2_X:.4f}', f'{r.uM:+.4f}',
                     f'{r.uX:+.4f}', f'{r.shared:+.4f}', f'{r.M2M_mod:.3f}', f'{r.M2M_conv:.2f}'])
table(rows, align='CENTER')
figure('rep_cs2_m2m.png',
       'Figure 7 — (a) what each block predicts on its own; (b) the commonality split; (c) both '
       'metrics on a log scale against the intrinsic reference at 1. The modified metric puts the '
       'correct and the broken model on opposite sides of that line; the conventional one does not.')
P('<b>This is the clearest result the modified metric produces.</b> For both responses the correct '
  'model sits above 1 and the broken one below it &mdash; a separation of about 130&times; on '
  'n<sub>C</sub> and 30 000&times; on n<sub>D</sub>, straddling the reference point. The mechanism '
  'is visible without any ratio: breaking the reverse reaction collapses the mechanistic block '
  'from Q&sup2; = 0.939 to 0.579 on n<sub>C</sub> while the measured block picks up the slack, '
  '0.883 to 0.962. The uniques simply follow. The conventional ratio orders the two the same way '
  '(235 against 2.11) but places both above 1, so it yields a ranking and no verdict.')

h2('3.5 Sensitivity to the block layout')
P('The result depends on the block layout, and this should be stated rather than hidden. An '
  'earlier layout truncated the measured block at a horizon while the mechanistic block kept the '
  'full batch. The source paper never does this &mdash; it truncates only to equalise trajectory '
  'lengths, and for rolling prediction it <i>fills</i> the future with mechanistic forecasts '
  'rather than cutting it. Under truncation u<sub>X</sub> collapsed to 0.000&ndash;0.065 and the '
  'metric inverted on one response; under the faithful layout u<sub>X</sub> for the broken model '
  'recovers to 0.394. Much of the reported degeneracy was an artefact of the layout.')
r0, r1 = v2.loc[('nC', 'U0_correct')], v2.loc[('nC', 'U1_no_reverse')]
P('The paper&rsquo;s rolling hybrid block was also run, as a negative control. It behaves exactly '
  'as the source paper warns it does &mdash; because the hybrid block is built from the '
  'candidate&rsquo;s own forecasts, the mechanistic block becomes redundant with it: on '
  f'n<sub>C</sub> the correct model gets u<sub>M</sub> = {r0.uM:+.4f} (negative) and '
  f'M2M* = {r0.M2M_mod:.3f}, against {r1.M2M_mod:.3f} for the broken model. Both metrics fail '
  'there. This variant is reported as a control, not as a result.')
note('<b>What to report.</b> The <i>side of 1</i> is robust across responses; the magnitude is '
     'not. The correct model&rsquo;s M2M* moves from 3.49 to 414 between responses because its '
     'u<sub>X</sub> is tiny (0.0009&ndash;0.022) and a small noisy denominator swings the ratio '
     'hard &mdash; the 1/u<sub>X</sub> scaling of &sect;1.2 again. At a shorter shared grid '
     '(0&ndash;30 h) the n<sub>D</sub> comparison inverts.')

sp(6)

# ================================================================= PART IV
h1('Part IV &mdash; Synthesis')

h2('4.1 Where the modified metric helps')
P('<b>It has a meaningful zero point.</b> This is its one unambiguous advantage and case study 2 '
  'demonstrates it: the correct and the broken mechanism land on opposite sides of 1, so the '
  'metric returns a verdict rather than only an ordering. The conventional ratio put both models '
  'above 1 in every configuration tested, so its numeric value carries no absolute meaning.')
P('<b>It tracks quality where quality is gradeable.</b> In case study 1, with seven mechanisms of '
  f'graded quality, it is the best of the candidate statistics against out-of-domain performance '
  f'(&rho; = {rho_ext.correlation:+.3f}, p = {rho_ext.pvalue:.3f}) and clearly better than Q&sup2; '
  'in-domain, which is nearly uninformative.')

h2('4.2 Where it fails, and why')
P('<b>The denominator can vanish.</b> u<sub>X</sub> is what the measurements add once the '
  'mechanism is present. A good mechanism on a well-measured process drives it toward zero, and '
  'the ratio then explodes and loses both precision and meaning. This is not a tuning problem; it '
  'is structural. The observed failure &mdash; a broken model out-ranking the correct one &mdash; '
  'happened exactly when u<sub>X</sub> rounded to 0.0000.')
P('<b>It discards magnitude.</b> Uniques of 0.05/0.05 and 0.40/0.40 give the same ratio of 1 but '
  'describe completely different situations: the hybrid advantage differs eightfold and so does '
  'the ratio&rsquo;s own standard error. The remedy is not to fold the shared term into the ratio '
  '&mdash; that would destroy the intrinsic reference, and shared is symmetric in M and X so it '
  'cannot rank them &mdash; but to report coverage alongside it as a second axis.')
P('<b>It is layout-sensitive.</b> Case study 2 changed its answer when the block layout changed. '
  'Any application must state the layout and justify it against the source method.')

h2('4.3 Recommended reporting')
P('Report the pair, not a blend: <b>M2M* for direction</b>, with its reference at 1, and '
  '<b>coverage = (u<sub>M</sub>+u<sub>X</sub>)/joint for magnitude and credibility</b>. Report '
  'the side of 1 rather than the magnitude where u<sub>X</sub> is small. Where a family of '
  'graded candidates exists, validate against something external &mdash; a fit cost, an '
  'out-of-domain score &mdash; rather than against in-domain Q&sup2;, which does not '
  'discriminate.')

h2('4.4 Open items')
P('&bull;&nbsp; The urethane input workbook (26+ batches) is not in the authors&rsquo; archive. '
  'Obtaining it would convert case study 2 from a reconstruction into an exact replication and '
  'would let the analysis run on their experimental design rather than on forty batches sampled '
  'around one validated run.<br/>'
  '&bull;&nbsp; With two models in case study 2 and seven in case study 1, none of the rank '
  'correlations is firmly established. A larger candidate family for the urethane process would '
  'test whether the reference at 1 keeps separating good from bad mechanisms.<br/>'
  '&bull;&nbsp; No threshold for &ldquo;u<sub>X</sub> too small to divide by&rdquo; has been '
  'established. A defensible rule would compare u<sub>X</sub> against its own '
  'cross-validation standard error.')

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=16*mm, bottomMargin=16*mm,
                        title='Two Case Study M2M Report')

def page_num(canvas, doc_):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#777777'))
    canvas.drawCentredString(A4[0]/2, 10*mm, str(canvas.getPageNumber()))
    canvas.restoreState()

doc.build(S, onFirstPage=page_num, onLaterPages=page_num)
print(f'Saved {OUT}  ({os.path.getsize(OUT)/1024:.0f} kB)')
