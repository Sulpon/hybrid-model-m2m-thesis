"""Build the detailed outline PDF for Chapters 3-5."""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)

OUT = 'Thesis_Chapters_3to5_Outline.pdf'
ACC = colors.HexColor('#1F3864')
GREY = colors.HexColor('#F2F2F2')
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=15, textColor=ACC,
                    spaceBefore=12, spaceAfter=6, leading=18)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=11.5, textColor=ACC,
                    spaceBefore=10, spaceAfter=3, leading=14)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontSize=9.8,
                    textColor=colors.HexColor('#333333'), spaceBefore=6,
                    spaceAfter=2, leading=12)
BODY = ParagraphStyle('BODY', parent=ss['BodyText'], fontSize=9, leading=12.4,
                      alignment=TA_JUSTIFY, spaceAfter=3)
BUL = ParagraphStyle('BUL', parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
SUB = ParagraphStyle('SUB', parent=BODY, leftIndent=22, bulletIndent=14,
                     fontSize=8.4, leading=11, textColor=colors.HexColor('#444444'),
                     spaceAfter=1.5)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8.3, leading=11.3,
                      textColor=colors.HexColor('#444444'), leftIndent=8,
                      rightIndent=8, borderPadding=4, backColor=GREY,
                      spaceBefore=4, spaceAfter=6)
SRC = ParagraphStyle('SRC', parent=BODY, fontName='Courier', fontSize=7.6,
                     leading=10, leftIndent=10, textColor=colors.HexColor('#555555'),
                     spaceAfter=5)
TITLE = ParagraphStyle('TITLE', parent=ss['Title'], fontSize=20, textColor=ACC, leading=24)
SUBT = ParagraphStyle('SUBT', parent=ss['Normal'], fontSize=11, alignment=1,
                      textColor=colors.HexColor('#444444'), leading=15)

def P(t, s=BODY): return Paragraph(t, s)
def B(t, s=BUL): return Paragraph(t, s, bulletText='•')
def S(t): return Paragraph(t, SUB, bulletText='–')
def src(t): return Paragraph('source: ' + t, SRC)

def tbl(rows, widths, fs=7.6, header=True):
    t = Table(rows, colWidths=widths, hAlign='LEFT')
    st = [('FONTSIZE', (0, 0), (-1, -1), fs),
          ('LEADING', (0, 0), (-1, -1), fs + 2.2),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#BBBBBB')),
          ('TOPPADDING', (0, 0), (-1, -1), 2.2),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F8FA')])]
    if header:
        st += [('BACKGROUND', (0, 0), (-1, 0), ACC),
               ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')]
    t.setStyle(TableStyle(st))
    return t

F = []
# ===================================================== title
F += [Spacer(1, 40*mm),
      P('Detailed Outline: Chapters 3&ndash;5', TITLE), Spacer(1, 5*mm),
      P('Measuring physics-driven and data-driven contributions<br/>in hybrid process models', SUBT),
      Spacer(1, 12*mm)]
F.append(tbl([['Chapter 3', 'Investigated Processes and Data Generation'],
              ['Chapter 4', 'Results and Discussion (six sections)'],
              ['Chapter 5', 'Conclusions and Outlook']],
             [28*mm, 120*mm], fs=9, header=False))
F += [Spacer(1, 10*mm),
      P('<b>Scope.</b> This outline covers only what has actually been produced. Results are quoted '
        'where they exist so each section has something concrete to be built around. Items that are '
        'still missing are marked <b>GAP</b> and collected again at the end. Section numbering follows '
        'the six-section Chapter 4 agreed earlier; the overall discussion moves to Chapter 5.', NOTE),
      PageBreak()]

# ===================================================== chapter 3
F += [P('Chapter 3. Investigated Processes and Data Generation', H1),
      P('Two simulated processes. The first carries the main development; the second tests whether '
        'the methodology transfers. Chapter 2 stays general, so every process-specific choice belongs '
        'here.', BODY)]

F += [P('3.1 Two-stage batch reactor', H2)]
F += [P('3.1.1 Process description', H3),
      B('Reaction network: stage 1 A &rarr; B &rarr; C; stage 2 B + 2D &rarr; E and C + 2D &rarr; F.'),
      B('Competition for the shared reagent D &mdash; the mechanism that makes the misspecifications bite.'),
      B('Operating variables and their ranges; what is measured at each stage and when.')]

F += [P('3.1.2 Correct and misspecified model variants', H3),
      B('Seven candidates M0&ndash;M6: correct; no side reaction; first order in D; wrong activation '
        'energy Ea4; lumped E/F; no D dependence; the source paper&rsquo;s own misspecification.'),
      B('For each: the modified equation, and what kind of error it represents (missing pathway, '
        'wrong order, wrong parameter, lumping).'),
      S('State which are structural and which parametric &mdash; this distinction reappears in 4.4.')]

F += [P('3.1.3 Kinetic parameter estimation', H3),
      B('<b>GAP in the current outline.</b> The mechanistic block is built from parameters '
        '<i>fitted to calibration data</i>, not literature values. This is load-bearing: the '
        'fitted-versus-nominal choice changed results materially.'),
      B('Objective, solver, bounds, starting values, convergence settings; fitted on calibration only.'),
      B('Report the fitted parameters per candidate and the residual cost as a mechanism-quality ranking.')]

F += [P('3.1.4 Data generation', H3),
      B('SDE formulation, integrator, step size, diffusion coefficients.'),
      B('Discretisation and measurement noise; noise scale fixed from calibration batches only.'),
      B('<b>Dataset design</b> &mdash; currently missing from the outline and required by 4.5:'),
      S('calibration, T2 ~ U(330, 370) K, n = 100'),
      S('in-domain control, same distribution, independent draw, n = 100'),
      S('extrapolation, T2 ~ U(370, 382) K, n = 100'),
      B('Why an in-domain control is necessary: without one, degradation cannot be separated from '
        'baseline difficulty. An earlier control drawn only from the top of the calibration range '
        'made extrapolation appear to <i>outperform</i> in-domain prediction.')]

F += [P('3.1.5 Data blocks', H3),
      B('X1, M2, X2, Y: contents, dimensions, physical meaning.'),
      B('Stage-2 initial conditions for the mechanistic block come from upstream measured quantities, '
        '<b>not</b> from the first sample of the trajectory being predicted. The alternative makes the '
        'measured block a copy of M and collapses the iterative procedure of 4.5.'),
      B('Trajectory unfolding; truncation to a common length; the response taken at each batch&rsquo;s '
        'own endpoint.'),
      B('Note the structural ceiling this creates: the columns Y is an exact function of are not in X2.')]

F += [P('3.2 Urethane manufacturing process', H2)]
F += [P('3.2.1 Process description', H3),
      B('Semi-batch reactor, two feed vessels; A + B &rarr; C, A + C &#8652; D, 3A &rarr; E.'),
      B('Competition for the shared reagent A &mdash; the analogue of D in 3.1.'),
      B('Manipulated inputs f<sub>v1</sub>, f<sub>v2</sub>, T; measured species n<sub>C</sub>, '
        'n<sub>D</sub>, n<sub>E</sub>.')]

F += [P('3.2.2 Correct and misspecified model variants', H3),
      B('<b>GAP in the current outline</b> &mdash; 3.2 has no counterpart to 3.1.2, so the case study '
        'has no candidate mechanisms and cannot carry the methodology.'),
      B('Seven candidates U0&ndash;U6: correct reversible; irreversible second reaction (the source '
        'papers&rsquo; own model); no isocyanurate reaction; first order in A; lumped rate constant; '
        'constant volume; temperature-independent kinetics.')]

F += [P('3.2.3 Data generation and provenance', H3),
      B('Ground-truth and first-principles models; the mismatch is exactly the missing reverse step.'),
      B('<b>Provenance matters here</b> &mdash; the case study is assembled from three sources:'),
      S('the seven initial molar charges are not tabulated anywhere; they are recovered by inverting '
        'the seven design variables of the original optimum-experimental-design paper'),
      S('the control profiles are read from a figure, with a stated reading uncertainty'),
      S('the two source papers conflict on whether the stated noise is a variance or a standard '
        'deviation; the conflict is resolved by a signal-to-noise argument'),
      B('Design: 50 calibration + 10 validation batches, 90 h, sampling every 30 min.'),
      B('<b>GAP:</b> there is no extrapolation design for this process &mdash; the published validation '
        'batches are drawn from the same input ranges as calibration. Either design one, or state '
        'explicitly that case study 2 tests the calibration-side diagnostic only.')]

F += [P('3.2.4 Data blocks', H3),
      B('M, X, Y; the KD inputs are moved into M by the causal-ordering rule.'),
      B('The mechanistic block spans the full batch while the measured block stops at a decision '
        'point &mdash; the single-decision-point form of the procedure in 2.6.'),
      B('<b>Response selection.</b> Final n<sub>C</sub> is unusable: its measurement noise exceeds its '
        'entire batch-to-batch spread (SNR 0.37&ndash;0.99), and every model returns negative Q&sup2;. '
        'n<sub>D</sub> has SNR &asymp; 143 and is the species the structural mismatch acts on.')]

F += [P('3.3 Verification of the generated data', H2),
      B('<b>GAP in the current outline.</b> This is what makes both case studies credible and '
        'currently appears nowhere.'),
      B('Case study 1: reproduction of the reference latent-variable allocations {4, 6, 2} and {3, 3, 6}.'),
      B('Case study 2, three independent checks:'),
      S('atom balances on A and B close to machine precision'),
      S('the reverse rate constant implied by the equilibrium relation agrees with the value another '
        'study discovered from data alone, to 0.03&ndash;0.07 standard deviations'),
      S('the mechanistic mean absolute error for species C reproduces the published value to within '
        '1.5 % in calibration and 5 % in validation'),
      B('Where replication is not exact, say why &mdash; the two unstated choices are the residual '
        'weighting in parameter estimation and whether time enters the data-driven model.'),
      src('regen_corrected_data.py, urethane_case_study.py, urethane_replicate.py'),
      PageBreak()]

# ===================================================== chapter 4
F += [P('Chapter 4. Results and Discussion', H1),
      P('Six sections. The arc: do the methods work &rarr; how is information split &rarr; is M2M '
        'stable &rarr; does the modification help &rarr; does it predict extrapolation &rarr; does it '
        'transfer.', BODY)]

F += [P('4.1 Baseline multiblock model comparison', H2),
      P('<i>Purpose: establish that all three methods are adequate here, and justify SO-PLS as the '
        'vehicle.</i>', BODY),
      B('MB-PLS, SO-PLS, SMB-PLS; latent-variable selection by RMSECV; reproduction of the reference '
        'allocations.'),
      B('Comparable predictive accuracy across methods; SO-PLS adopted for its variance-decomposition '
        'property, which the other two do not admit.'),
      B('Remark: SMB-PLS reduces to SO-PLS on this data &mdash; the block score is exactly parallel to '
        'the super-score, so the three-way shared/unique split is degenerate. Derivation to an appendix.'),
      src('sopls_two_orderings.py, sopls_two_block.py')]

F += [P('4.2 Decomposition of mechanistic and measured contributions', H2),
      P('<i>Purpose: this is the motivation for the modified ratio in 2.4, and should say so explicitly.</i>', BODY),
      B('Sequential sums of squares under both block orderings and the resulting asymmetry.'),
      S('for the correct model, SS<sub>M</sub> falls from 71.72 to 16.43 when M is placed last, while '
        'SS<sub>X</sub> rises from 5.71 to 61.58'),
      S('roughly 56&ndash;67 % of SS<sub>Y</sub> is jointly explainable and unattributable by any '
        'single ordering'),
      B('Unique versus shared contributions via block removal; the commonality identity '
        'Joint = Unique<sub>M</sub> + Unique<sub>X</sub> + Shared.'),
      S('the two unique parts capture only 17&ndash;37 % of the joint effect; Shared is 0.47&ndash;0.68 '
        'out of a joint of about 0.84'),
      S('the data-driven baseline is identical across candidates because it contains no mechanistic '
        'block &mdash; note what this implies for cross-model comparison'),
      B('Species-level removal: which species carry the contribution.'),
      B('Protocol sensitivity: re-optimising the sub-models versus holding them at the full '
        'model&rsquo;s allocation roughly doubles the unique mechanistic contribution. State which is used.'),
      src('sopls_commonality.py, X2_removed_analysis.py, shared_variance_forward_reverse.py')]

F += [P('4.3 Sensitivity of M2M to latent-variable allocation', H2),
      P('<i>Purpose: show the point estimate is not trustworthy, and define what replaces it.</i>', BODY),
      B('M&aring;ge plot and the full allocation grid; M2M varies by orders of magnitude across '
        'allocations of near-identical RMSECV.'),
      B('Definition and size of the 1-SE flat region; region sizes and the standard error per model.'),
      B('Distribution of M2M within the region; regional median versus point estimate; the optimum is '
        'not a representative allocation.'),
      B('Why the rule stops at one standard error:'),
      S('the first standard error buys simplification at about 0.42 percentage points of Q&sup2; per '
        'latent variable removed; the second costs about 1.33 &mdash; three times the price for less '
        'than half the benefit'),
      S('at two standard errors the internal Q&sup2; spread of the region roughly doubles, so the '
        'premise that members are equivalent no longer holds'),
      S('state the cost honestly: the 1-SE rule is not free, it costs 1.8&ndash;3.2 standard '
        'deviations of partition noise'),
      B('Stability across cross-validation partitions.'),
      src('lv_allocation_analysis.py, metric_ratio_lv_distribution.py, metric_ratio_flat_regions.py, '
          'q2_flat_region_comparison.py, sopls_two_block_ceiling.py')]

F += [P('4.4 Original versus modified M2M ratio', H2),
      P('<i>Purpose: the contribution. Report what it does and what it does not do.</i>', BODY),
      B('Both ratios across the seven candidates; SS- and Q&sup2;-based estimators compared.'),
      B('The reference value 1: which mechanisms fall either side and how decisively.'),
      S('mechanisms omitting a whole reaction pathway fall below 1; the structurally sound ones rise '
        'above it, unanimously across cross-validation partitions'),
      B('Where the modification does not help:'),
      S('it does not separate the correct model from the wrong-activation-energy variant'),
      S('it does not rank mechanisms by correctness &mdash; what it measures is how heavily the hybrid '
        'leans on the mechanistic block'),
      B('Numerical behaviour: unbounded above, undefined when the denominator reaches zero. Report the '
        'fraction of allocations where this occurs and how the flat region suppresses it.'),
      src('metric_unique_m2m.py, M2M_vs_AMR_shared_comparison.py')]

F += [P('4.5 Extrapolation validation', H2),
      P('<i>Purpose: does a calibration-only diagnostic predict out-of-domain behaviour?</i>', BODY),
      B('Datasets and the in-domain control; what &ldquo;extrapolation&rdquo; means here.'),
      B('<b>4.5.1 Single-decision-point test</b> &mdash; full measured trajectory available.'),
      B('<b>4.5.2 Iterative (rolling) prediction</b> &mdash; at each decision point the unobserved part '
        'of the trajectory is replaced by mechanistic predictions.'),
      S('report performance against decision point for both regimes'),
      S('misspecified mechanisms recover as observations accumulate; sound ones are stable'),
      B('<b>4.5.3 Diagnostics versus extrapolation performance</b> &mdash; rank correlation across the '
        'candidates, for the point estimate and the regional median.'),
      S('the regional median outperforms the point estimate'),
      S('calibration fit alone predicts nothing, so the diagnostic adds real information'),
      B('<b>The two evaluation modes disagree</b>, and the reason is identifiable: at the fully '
        'observed endpoint the mechanistic block is redundant, so a model leaning on it looks '
        'marginally worse. Only the rolling regime tests what the metric is about. Report the '
        'single-decision-point result as a negative control that motivates the rolling design.'),
      src('extrapolation_test.py, extrapolation_1se_distribution.py, regen_fit_and_test.py, '
          'regen_fit_and_test_paperIC.py, iterative_prediction.py')]

F += [P('4.6 Application to the urethane manufacturing process', H2),
      P('<i>Purpose: does the methodology transfer? Partly &mdash; and the failure is informative.</i>', BODY),
      B('<b>4.6.1 Replication of published results</b> &mdash; parameter estimates and error metrics '
        'against the source study; where agreement is exact and where it is not.'),
      B('<b>4.6.2 M2M analysis</b> across the seven candidate mechanisms, at several decision points.'),
      B('<b>4.6.3 Degeneracy of the modified ratio.</b> The measured block adds almost nothing beyond '
        'the mechanistic block, so the denominator collapses and the ratio becomes undefined &mdash; '
        'including for the correct mechanism, the worst place to lose it.'),
      S('the original SS-based ratio remains defined and still ranks mechanism quality'),
      B('<b>Scope condition.</b> The determining factor is whether the mechanistic block can forecast '
        'beyond the measurement window. In case study 1 it cannot, and the unique contribution of the '
        'measured block is healthy; here it can, and that contribution vanishes. State this as a '
        'condition of applicability rather than as a limitation.'),
      src('urethane_case_study.py, urethane_replicate.py, urethane_m2m.py'),
      PageBreak()]

# ===================================================== chapter 5
F += [P('Chapter 5. Conclusions and Outlook', H1),
      P('5.1 Summary of findings', H2),
      B('What the original M2M ratio does and does not measure.'),
      B('What the modified ratio adds: an intrinsic reference point, and no credit for shared variance.'),
      B('What it costs: a denominator that can vanish.'),
      P('5.2 Methodological recommendations', H2),
      B('Report the ratio as a regional median with an interval, not as a point estimate.'),
      B('Report the shared component alongside it, so the reader sees what the ratio does not account for.'),
      B('Pin down the sub-model allocation protocol in the definition.'),
      B('Evaluate extrapolation with the rolling procedure; the fully observed endpoint is not a test '
        'of mechanistic contribution.'),
      P('5.3 Scope conditions and limitations', H2),
      B('The metric applies where the measured block retains a non-trivial unique contribution.'),
      B('Rank correlations over a handful of candidate models have very low power; say so.'),
      B('Results rest on single realisations unless repeated across generation seeds.'),
      B('Absolute values are not comparable across data regimes; only within-regime comparisons hold.'),
      P('5.4 Outlook', H2),
      B('Repeat across generation seeds to confirm the extrapolation result.'),
      B('Re-estimate mechanistic parameters on data spanning the extrapolation range, to separate '
        '&ldquo;wrong mechanism&rdquo; from &ldquo;parameters estimated in the wrong domain&rdquo;.'),
      B('Investigate the fraction of allocations at which the measured block adds nothing '
        'out-of-sample, which separated sound from broken mechanisms more cleanly than the ratio itself '
        'and was not pursued.'),
      B('Bounded reformulations of the ratio that avoid the vanishing denominator.')]

F += [P('Open decisions and gaps', H1),
      P('Collected from above. Each needs either work or an explicit statement in the text.', BODY)]
gaps = [
    ['3.1.3 / 3.2.3', 'Parameter estimation has no subsection', 'Write it; state the objective, solver and that fitting uses calibration only'],
    ['3.1.4', 'Dataset split not described', 'Add calibration / in-domain control / extrapolation'],
    ['3.2.2', 'No candidate mechanisms for urethane', 'Add U0-U6, mirroring 3.1.2'],
    ['3.2.3', 'No extrapolation design for urethane', 'Design one, or state that case study 2 is calibration-only'],
    ['3.3', 'No verification section', 'Add; the replication checks already exist'],
    ['4.4', 'Protocol for sub-model allocation', 'Fix one convention and state it'],
    ['4.5', 'Single seed only', 'Repeat across generation seeds, or state as a limitation'],
    ['Ch. 2', 'LV selection rule stated inconsistently', 'Argmin or parsimony within 1-SE - decide once'],
]
F.append(tbl([['Section', 'Gap', 'Action']] + gaps, [24*mm, 58*mm, 78*mm], fs=7.6))

F += [Spacer(1, 5*mm),
      P('<b>One caution on reuse.</b> Several numbers quoted in this outline come from specific data '
        'regimes that are not interchangeable &mdash; real versus regenerated data, fitted versus '
        'nominal kinetic parameters. Before a number goes into the thesis, check which regime produced '
        'it against the source file named under each section.', NOTE)]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#777777'))
    canvas.drawString(20*mm, 12*mm, 'Detailed outline - Chapters 3 to 5')
    canvas.drawRightString(A4[0]-20*mm, 12*mm, f'Page {doc.page}')
    canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
    canvas.line(20*mm, 15*mm, A4[0]-20*mm, 15*mm)
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                        topMargin=18*mm, bottomMargin=20*mm,
                        title='Detailed Outline: Chapters 3-5')
doc.build(F, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
