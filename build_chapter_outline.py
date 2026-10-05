"""
Detailed outline for Chapters 2-6.

Chapters 2 and 3 are written to be case-study agnostic: they define the metric
and the procedure in generic block notation (M, X, Y) and name no process. The
case studies appear for the first time in Chapter 4.

Every numeric claim listed for Chapter 5 is taken from a saved workbook, named
in the entry, so the outline can be checked against the analyses.
"""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, KeepTogether)

OUT = 'Thesis_Chapter_Outline.pdf'
NAVY = colors.HexColor('#1F3864')
GREEN = colors.HexColor('#2E7D32')
RED = colors.HexColor('#C0392B')
GREY = colors.HexColor('#F2F2F2')
AMBER = colors.HexColor('#B06A00')

ss = getSampleStyleSheet()
CH = ParagraphStyle('CH', parent=ss['Heading1'], fontSize=16, textColor=colors.white,
                    spaceBefore=2, spaceAfter=2, leading=19, leftIndent=6)
SEC = ParagraphStyle('SEC', parent=ss['Heading2'], fontSize=11.5, textColor=NAVY,
                     spaceBefore=10, spaceAfter=3, leading=14)
BODY = ParagraphStyle('BODY', parent=ss['BodyText'], fontSize=9, leading=12.4,
                      alignment=TA_JUSTIFY, spaceAfter=3)
LBL = ParagraphStyle('LBL', parent=BODY, fontSize=8.2, leading=11.2, leftIndent=10,
                     spaceAfter=2)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8.2, leading=11, textColor=
                      colors.HexColor('#444444'), leftIndent=8, rightIndent=8,
                      borderPadding=4, backColor=GREY, spaceBefore=4, spaceAfter=6)
TITLE = ParagraphStyle('TITLE', parent=ss['Title'], fontSize=19, textColor=NAVY, leading=23)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontSize=10.5, alignment=1,
                     textColor=colors.HexColor('#444444'), spaceBefore=4)

S = []


def chapter(num, title, words=None):
    t = Table([[Paragraph(f'Chapter {num} &nbsp;&mdash;&nbsp; {title}', CH)]],
              colWidths=[174*mm])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), NAVY),
                           ('TOPPADDING', (0, 0), (-1, -1), 5),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    S.append(Spacer(1, 6)); S.append(t); S.append(Spacer(1, 5))
    if words:
        S.append(Paragraph(f'<i>Target length ~{words}. '
                           f'{"Case studies are NOT named in this chapter." if num in (2, 3) else ""}</i>',
                           ParagraphStyle('x', parent=BODY, fontSize=8.3,
                                          textColor=colors.HexColor('#666666'))))


def sec(num, title):
    S.append(Paragraph(f'{num} &nbsp;{title}', SEC))


def say(text):
    S.append(Paragraph(text, BODY))


def item(kind, text, colour=None):
    cmap = {'VISUAL': NAVY, 'CLAIM': GREEN, 'TABLE': NAVY,
            'TO MAKE': AMBER, 'CAUTION': RED, 'SOURCE': colors.HexColor('#555555')}
    c = colour or cmap.get(kind, NAVY)
    S.append(Paragraph(f'<font color="{c.hexval()}"><b>{kind}</b></font> &nbsp; {text}', LBL))


def note(t):
    S.append(Paragraph(t, NOTE))


# ============================================================== title page
S.append(Paragraph('Thesis Chapter Outline', TITLE))
S.append(Paragraph('Chapters 2 to 6 &mdash; structure, visuals and the claim to make in each section', SUB))
S.append(Spacer(1, 10))
note('<b>Two structural decisions are built into this outline.</b> '
     '(i) The modified M2M ratio is promoted to its own chapter, placed '
     'immediately after Chapter 1 so that it follows directly from &sect;1.6, '
     'where the conventional ratio is introduced. (ii) Chapters 2 and 3 are '
     'written entirely in generic block notation (M, X, Y) and name no process, '
     'so both case studies inherit them unchanged. The case studies are '
     'introduced for the first time in Chapter 4.')
S.append(Spacer(1, 4))
rows = [['', 'Chapter', 'Role', 'Length'],
        ['1', 'Mathematical background', 'unchanged; ends at §1.6 the M2M ratio', '~2,900'],
        ['2', 'A Modified M2M Ratio', 'NEW &mdash; the contribution', '~3,200'],
        ['3', 'Methodology', 'generic procedure', '~2,400'],
        ['4', 'Investigated Processes and Data Generation', 'both case studies', '~3,000'],
        ['5', 'Results and Discussion', 'all findings', '~4,500'],
        ['6', 'Conclusions', 'contribution, limits, future work', '~1,200']]
t = Table([[Paragraph(f'<b>{c}</b>' if i == 0 else c, BODY) for c in r]
           for i, r in enumerate(rows)],
          colWidths=[10*mm, 56*mm, 76*mm, 22*mm])
t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#BBBBBB')),
                       ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                       ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                       ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GREY]),
                       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                       ('TOPPADDING', (0, 0), (-1, -1), 3),
                       ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
S.append(t)
S.append(PageBreak())

# ================================================================ CHAPTER 2
chapter(2, 'A Modified M2M Ratio', '3,200 words')

sec('2.1', 'Limitations of the conventional ratio')
say('Open from &sect;1.6. The conventional ratio is formed from a sequential '
    'decomposition in which the mechanistic block enters first, so the variance '
    'that both blocks could explain is credited entirely to the numerator. '
    'State the two consequences: the ratio grows whenever the blocks overlap, '
    'independently of how good the mechanism is; and it carries no reference '
    'value, so a single number cannot be interpreted on its own.')
item('CLAIM', 'A metric usable only by comparison cannot certify one model &mdash; '
     'which is what an engineer deciding whether to trust a mechanism needs.')
item('TO MAKE', '<b>Fig. 2.1</b> &mdash; set diagram of joint = u<sub>M</sub> + '
     'u<sub>X</sub> + shared, with the conventional numerator shaded to show it '
     'absorbing the shared region. One generic schematic, no data.')

sec('2.2', 'Commonality decomposition of the block contributions')
say('Define the four quantities by equation and in words. State that they are '
    'obtained by refitting the model with one block removed and differencing, '
    'and that when a prior block is present the decomposition is conditional on it. '
    'This section absorbs the block-level half of the current §2.3; the '
    'species-level half moves to Chapter 3.')
item('TO MAKE', '<b>Table 2.1</b> &mdash; the four quantities: symbol, definition '
     'as a Q² difference, and plain-language interpretation.')

sec('2.3', 'Definition of the modified ratio and its intrinsic reference')
say('Give M2M* = u<sub>M</sub>/u<sub>X</sub>. State precisely what the reference '
    'at 1 means: it is the point at which the two blocks supply equal unique '
    'information, so the threshold is a property of the decomposition rather '
    'than a convention. Say what a value above 1 does and does not license.')
item('CLAIM', 'The reference is intrinsic, not calibrated: no second model and '
     'no tuning constant is required to interpret a single value.')

sec('2.4', 'Why the shared term is excluded')
say('Three independent arguments, in this order.')
item('', '<b>(i) Structural.</b> shared is unchanged when M and X are swapped. '
     'A quantity that ranks one block against the other must be antisymmetric '
     'under that swap; the ratio inverts, shared does not. It is therefore '
     'incapable in principle of expressing the comparison.')
item('', '<b>(ii) Informational.</b> When the joint explained variance and '
     'u<sub>M</sub> are stable across candidates, shared reduces to a constant '
     'minus u<sub>X</sub> &mdash; so it carries nothing the ratio does not already '
     'use, and uses it without an orientation. Forward-reference the demonstration '
     'to §5.4.')
item('', '<b>(iii) Decision-theoretic.</b> The advantage of the hybrid over the '
     'better single-source model is min(u<sub>M</sub>, u<sub>X</sub>), which is '
     'bounded by (joint − shared)/2. A large shared term therefore argues '
     '<i>against</i> hybridising, not for it.')
item('CLAIM', 'Dropping the shared term costs no ranking information and buys the '
     'reference point. It is reported alongside the ratio, never inside it.')

sec('2.5', 'Estimation: cross-validated Q² rather than sums of squares')
say('In-sample sequential sums of squares increase monotonically with latent '
    'variables, so a ratio built from them drifts with the allocation. '
    'Cross-validated Q² penalises added components and pins the ratio down. '
    'State that u<sub>M</sub> and u<sub>X</sub> are therefore out-of-sample gains '
    'and that the metric cannot be inflated by adding components.')
item('', 'Also fix the <b>sub-model protocol</b>: each reduced model is '
     're-optimised on its own allocation grid rather than inheriting the full '
     "model's. Justify it &mdash; each sub-model should be given its best chance, "
     'or the difference is not a fair measure of what the removed block added.')
item('CLAIM', 'Forward-reference §5.6: under the sums-of-squares estimator the '
     'ratio for a correct mechanism crosses the reference value within the set of '
     'statistically equivalent allocations, so the threshold does not survive. '
     'Under cross-validation it does.')

sec('2.6', 'Latent-variable allocation and the stability of the ratio')
say('The optimal allocation is not unique. Refer forward to §3.3 for the 1-SE '
    'flat region and state the consequence here: the metric should be reported '
    'over that region rather than at a single point. Give the variance '
    'propagation var(M2M*) ≈ (1/u<sub>X</sub>²)·[var(u<sub>M</sub>) + '
    'M2M*²·var(u<sub>X</sub>)] and draw the conclusion that precision is '
    'controlled by the denominator.')
item('', 'Introduce averaging over repeated cross-validation partitions, and '
     'report P(M2M* &gt; 1) across them as the primary summary &mdash; it is the '
     'quantity the threshold actually licenses, and it is far more stable than '
     'the magnitude.')

sec('2.7', 'Properties and scope conditions')
say('State the limits openly; this section protects the contribution rather than '
    'weakening it.')
item('', '<b>Scale-freedom.</b> The ratio depends only on the proportion of the '
     'uniques, so u<sub>M</sub>/u<sub>X</sub> = 0.05/0.05 and 0.40/0.40 give the '
     'same value while describing very different models. Define '
     'coverage = (u<sub>M</sub>+u<sub>X</sub>)/joint as the companion statistic.')
item('CAUTION', 'State explicitly that coverage is <b>not</b> a quality criterion '
     'and must not be maximised jointly with the ratio &mdash; the two oppose each '
     'other, because a good mechanism drives u<sub>X</sub> down. Evidence in §5.10.')
item('', '<b>Degeneracy.</b> u<sub>X</sub> → 0 when the mechanism can forecast the '
     'window in which the response is measured; the ratio then diverges and only '
     'its side of 1 remains meaningful.')
item('', '<b>Design requirements.</b> Sample size relative to block width, and '
     'the requirement that the response lie outside the window spanned by both blocks.')
item('TO MAKE', '<b>Table 2.2</b> &mdash; applicability checklist: condition, why '
     'it matters, what to report when it fails.')
S.append(PageBreak())

# ================================================================ CHAPTER 3
chapter(3, 'Methodology', '2,400 words')
note('Everything here is stated for an arbitrary set of candidate mechanistic '
     'models and an arbitrary process. No case study is named. Both case studies '
     'then reference this chapter without restating it.')

sec('3.1', 'Data preprocessing')
say('Keep the current §2.1 essentially as written &mdash; autoscaling, block '
    'scaling by 1/√K for MB-PLS, and train-fold-only scaling to prevent leakage. '
    'Add one sentence on how near-constant columns are handled, since mechanistic '
    'blocks can contain them.')

sec('3.2', 'Implementation of the multiblock models')
say('Keep the current §2.2, with the three algorithm boxes and the comparison '
    'table. This is the section to expand if Chapter 3 reads thin.')
item('CAUTION', 'Fix the inconsistency: the current text says latent variables '
     'were chosen by minimising RMSECV. Change to the parsimonious member of the '
     '1-SE region and cross-reference §3.3, or the chapter contradicts itself.')
item('TABLE', '<b>Table 3.1</b> &mdash; comparison of MB-PLS, SO-PLS and SMB-PLS '
     '(exists).')

sec('3.3', 'Latent-variable selection and the flat region')
say('Current §2.5. Define the grid search, the Måge plot, the 1-SE threshold and '
    'the parsimonious selection rule within the region. State the rule once, '
    'precisely, since Chapters 2 and 5 both depend on it.')
item('TO MAKE', '<b>Fig. 3.1</b> &mdash; generic Måge-plot schematic: RMSECV '
     'against total components, flat band marked, argmin and parsimonious choices '
     'labelled. Draw it with unlabelled axes so it stays case-study free.')

sec('3.4', 'Species-level contribution analysis')
say('The species-removal half of the current §2.3. Keep it here: it is a '
    'diagnostic procedure applied to the models, not part of the definition of '
    'the metric.')
item('VISUAL', '<b>Fig. 3.2</b> &mdash; species-level removal scheme (exists as '
     'current Figure 2.1).')

sec('3.5', 'Extrapolation validation')
say('Current §2.6. Describe the rolling scheme generically: at decision point k '
    'the measured trajectory up to k is combined with knowledge-driven forecasts '
    'beyond it, and the end-of-batch response is predicted. Define the performance '
    'measure and the rank-correlation comparison against the diagnostics.')
item('VISUAL', '<b>Fig. 3.3</b> &mdash; extrapolation procedure schematic (exists '
     'as current Figure 2.3).')
item('CAUTION', 'State here that the diagnostic is computed on calibration data '
     'only and the extrapolation set is never used to construct it. This is the '
     'sentence that makes the validation in §5.8 admissible.')

sec('3.6', 'Software and reproducibility')
say('Short. Custom implementations in Python with NumPy, SciPy and scikit-learn; '
    'fixed cross-validation seeds; each case study reproducible from a single '
    'script. One paragraph, no figures.')
S.append(PageBreak())

# ================================================================ CHAPTER 4
chapter(4, 'Investigated Processes and Data Generation', '3,000 words')

sec('4.1', 'Selection of the case studies')
say('State the design logic before either process appears: one case study with a '
    'family of candidate mechanisms of graded, known quality, to test whether the '
    'metric ranks as an oracle would; and one taken from independent published '
    'work, to test it on a misspecification the literature itself considers '
    'realistic rather than one constructed here.')

sec('4.2', 'Two-stage batch reactor')
say('4.2.1 Process description &mdash; the two stages, the desired and impurity '
    'routes, the response. 4.2.2 Data generation &mdash; random operating '
    'conditions, the stochastic differential equation formulation, measurement '
    'noise, number of batches. 4.2.3 Candidate model variants. 4.2.4 Data blocks.')
item('VISUAL', '<b>Fig. 4.1</b> &mdash; reaction scheme for both stages.')
item('TO MAKE', '<b>Table 4.1</b> &mdash; the seven candidates: what was altered '
     'in each, the resulting rate laws, and the independent kinetic fit cost. '
     'Colour or group by whether the reaction network survives.')
item('CLAIM', 'The candidates span parameter-level errors, rate-law errors and '
     'structural deletions, and each carries an independent quality grade, so the '
     'metric can be checked against an oracle rather than against intuition.')
item('TABLE', '<b>Table 4.2</b> &mdash; data blocks: X₁ 100×6, M₂ 100×45, '
     'X₂ 100×42, Y 100×1 (exists in the current draft).')

sec('4.3', 'Urethane manufacturing process')
say('4.3.1 Process description and the reversible step. 4.3.2 Provenance: what '
    'was taken from the original authors and what had to be reconstructed, stated '
    'explicitly. 4.3.3 Validation of the reconstruction. 4.3.4 The two candidate '
    'models defined by the source papers. 4.3.5 Data blocks.')
item('VISUAL', '<b>Fig. 4.2</b> &mdash; reaction scheme. <b>Fig. 4.3</b> &mdash; '
     'published trajectories against the reconstruction '
     '(<font face="Courier">cs2_compare_published.png</font>).')
item('CLAIM', 'Reconstruction validated at 5.0 % RMS deviation across six panels; '
     "and the authors' individual noise realisation is reproduced, not merely its "
     'variance &mdash; correlation 0.60 against −0.005 ± 0.055 for 25 alternative '
     'seeds, z = 11.1.')
item('TABLE', '<b>Table 4.3</b> &mdash; data blocks: M 40×906, X 40×423, Y 40×1 '
     '(<font face="Courier">deck_cs2_blocks_table.png</font>).')
item('CAUTION', 'State plainly that the authors\' input workbook was unavailable, '
     'so the batches are sampled around one validated run rather than being their '
     'experimental design. This is a reconstruction, not a replication.')

sec('4.4', 'Comparison of the two designs')
say('A short closing table contrasting the two: number of blocks, presence of a '
    'prior block, sample-to-variable ratio, source of the misspecification, and '
    'whether an independent quality grade exists. This sets up which results in '
    'Chapter 5 are expected to be weaker and why.')
item('TO MAKE', '<b>Table 4.4</b> &mdash; design comparison. Flag the 2:1 versus '
     '1:23 sample-to-variable contrast here; §5.10 returns to it.')
S.append(PageBreak())

# ================================================================ CHAPTER 5
chapter(5, 'Results and Discussion', '4,500 words')
note('Order is deliberate: establish that the problem is real (5.1), show the '
     'decomposition (5.2), show the metric works (5.3), explain <i>why</i> it '
     'works (5.4), show it is stable (5.5-5.6), compare with the conventional '
     'ratio (5.7), validate externally (5.8), replicate independently (5.9), and '
     'state the limits (5.10).')

sec('5.1', 'In-domain performance does not separate the candidates')
say('Open with the motivating negative result. Report the predictive performance '
    'of every candidate and show that it is effectively flat.')
item('CLAIM', 'All seven candidates reach Q² = 87.8 – 90.4 %, a span of 2.5 '
     'percentage points, while differing by more than 17 R² units out of domain. '
     'In-domain fit cannot rank mechanisms &mdash; which is the entire reason a '
     'contribution metric is needed.')
item('SOURCE', '<font face="Courier">check_lv_rule.xlsx</font>, '
     '<font face="Courier">case_study_1_rolling.xlsx</font>')

sec('5.2', 'Commonality decomposition of the candidates')
say('Present the full decomposition before any ratio is formed, so the reader '
    'sees the raw quantities first.')
item('VISUAL', '<b>Fig. 5.1</b> &mdash; stacked commonality bars per candidate '
     '(<font face="Courier">rep_cs1_commonality.png</font>).')
item('TABLE', '<b>Table 5.1</b> &mdash; Q² full, Q² without M, Q² without X, '
     'u<sub>M</sub>, u<sub>X</sub>, shared, coverage for each candidate.')
item('CLAIM', 'The shared term dominates throughout (56 – 80 % of the jointly '
     'explained variance), so coverage is 0.20 – 0.44 and the ratio arbitrates a '
     'minority of the model. Say this here rather than letting a reader discover it.')

sec('5.3', 'The modified ratio separates sound from broken mechanisms')
say('The headline result. Report the ratio with its dispersion across repeated '
    'cross-validation partitions and the probability of exceeding the reference.')
item('VISUAL', '<b>Fig. 5.2</b> &mdash; ratio per candidate with error bars and '
     'the reference line at 1, coloured by whether the mechanism is sound '
     '(<font face="Courier">deck_cs1_ratio.png</font>).')
item('TABLE', '<b>Table 5.2</b> &mdash; the seven-candidate table: Q² full, '
     'Q² no-M, Q² no-X, u<sub>M</sub>, u<sub>X</sub>, M2M*, ± sd, P(&gt;1). '
     '(<font face="Courier">check_lv_rule.xlsx</font>)')
item('CLAIM', 'Under the parsimonious 1-SE rule the four sound mechanisms score '
     '1.67 – 2.03 with P(&gt;1) = 1.00, and the three broken ones 0.36 – 1.08. '
     'Report the argmin column too (1.96 – 2.11 against 0.38 – 1.16) and state '
     'that the conclusion does not depend on the selection rule.')
item('CAUTION', 'Be explicit that one broken candidate sits just above the '
     'reference (1.08, P(&gt;1) = 0.75). Do not round it away &mdash; it is the '
     'honest boundary case and it is better volunteered than found.')

sec('5.4', 'Where the discriminating power comes from')
say('This section converts the result from an observation into an explanation, '
    'and it is what makes the contribution defensible rather than empirical.')
item('VISUAL', '<b>Fig. 5.3</b> &mdash; u<sub>M</sub> and u<sub>X</sub> side by '
     'side per candidate (<font face="Courier">deck_cs1_uniques.png</font>).')
item('CLAIM', 'u<sub>M</sub> is nearly constant across candidates (10 – 13 %); it '
     'is u<sub>X</sub> that moves, from about 6 % when the mechanism is sound to '
     '16 – 27 % when it is not. The metric detects a failing mechanism by '
     'observing that the measurements are forced to compensate.')
item('CLAIM', 'Tie this to the construction in §4.2.3: the threshold falls exactly '
     'where the reaction <i>network</i> breaks, not where parameters are merely '
     'wrong. Candidates that keep both reaction channels score near 2; those that '
     'delete a channel or remove a dependence fall to 1 or below.')
item('CLAIM', 'Deliver the empirical half of §2.4 here: shared and u<sub>X</sub> '
     'are rank-equivalent (Spearman = −1.000) when u<sub>M</sub> is stable, and the '
     'equivalence breaks (−0.286) when u<sub>M</sub> varies nine times more. Shared '
     'is u<sub>X</sub> without an orientation, which is why it is not used.')

sec('5.5', 'Stability across statistically equivalent allocations')
say('Show that the result does not depend on which member of the flat region is '
    'chosen.')
item('VISUAL', '<b>Fig. 5.4</b> &mdash; Måge plot: full RMSECV landscape with the '
     'flat region marked, beside the same region coloured by the ratio, with the '
     'argmin and parsimonious allocations labelled '
     '(<font face="Courier">deck_m0_flat_region.png</font>).')
item('CLAIM', 'Across all 233 allocations inside the 1-SE band the ratio spans '
     '1.59 – 1.82, a 14 % band, and the two candidate selections differ by 4 % '
     '(1.60 against 1.67). The allocation choice does not change the conclusion.')
item('CLAIM', 'Note that the sub-model allocation matters more than the full-model '
     'allocation (about ±0.5 against ±0.1). This justifies fixing the sub-model '
     'protocol in §2.5 and is a result in its own right.')
item('VISUAL', '<b>Fig. 5.5</b> &mdash; Q² cost against parsimony gained for each '
     'stopping rule (<font face="Courier">rep_cs1_q2_cost.png</font>).')
item('CLAIM', 'Averaged over the candidates, the first standard error removes 3.7 '
     'latent variables for 0.014 of Q²; the second costs a further 0.019 to remove '
     'only 1.6 more. Parsimony is nearly exhausted after one standard error while '
     'the predictive cost accelerates &mdash; the justification for stopping at 1-SE.')

sec('5.6', 'Estimator choice: cross-validated Q² against sums of squares')
say('The methodological result that justifies §2.5. Present it as a comparison, '
    'not as an assertion.')
item('VISUAL', '<b>Fig. 5.6</b> &mdash; the same flat region coloured by the '
     'sums-of-squares metrics, three panels '
     '(<font face="Courier">deck_m0_flat_region_ss.png</font>).')
item('TABLE', '<b>Table 5.3</b> &mdash; the seven candidates under the '
     'sums-of-squares estimator (<font face="Courier">table_ss_estimator.xlsx</font>).')
item('CLAIM', 'The sums-of-squares estimator ranks correctly but loses the '
     'threshold: a candidate with an entire reaction deleted scores 1.49 and 1.02, '
     'above the reference, on both selection rules. Under cross-validation the same '
     'candidate scores 0.70 and 0.67.')
item('CLAIM', 'Worse, six of the seven candidates have flat regions that straddle '
     'the reference, including all four sound ones: roughly 10 % of statistically '
     'equivalent allocations would declare a sound mechanism data-driven. '
     'Spread 0.69 – 6.80 against 1.59 – 1.82 for cross-validation.')
item('CLAIM', 'Conclude: the modified ratio requires cross-validated Q². In-sample '
     'sums of squares reward components monotonically, so the ratio drifts with the '
     'allocation and the reference washes out.')

sec('5.7', 'Conventional against modified ratio')
say('Direct comparison on the same candidates, same allocations.')
item('VISUAL', '<b>Fig. 5.7</b> &mdash; the two ratios side by side with the '
     'reference line (<font face="Courier">deck_cs1_vs.png</font>).')
item('CLAIM', 'Both order the candidates correctly. The conventional ratio places '
     'every candidate above 1 (2.4 – 12.6), so it yields an ordering and no verdict; '
     'the modified ratio places sound and broken mechanisms on opposite sides of a '
     'threshold that was not fitted to the data.')
item('CAUTION', 'Do not overclaim here. On ranking alone the conventional ratio is '
     'not worse. The contribution is interpretability, and saying so plainly is '
     'stronger than implying superiority on every axis.')

sec('5.8', 'External validation against out-of-domain performance')
say('The section that gives the metric something outside itself to be right about.')
item('VISUAL', '<b>Fig. 5.8</b> &mdash; rolling prediction error against decision '
     'point, within domain and extrapolated '
     '(<font face="Courier">rep_cs1_rolling.png</font>). '
     '<b>Fig. 5.9</b> &mdash; the ratio against kinetic fit cost and against '
     'extrapolation performance '
     '(<font face="Courier">rep_cs1_metric_vs_quality.png</font>).')
item('TABLE', '<b>Table 5.4</b> &mdash; Spearman correlations of u<sub>M</sub>, '
     'u<sub>X</sub>, shared, coverage, the ratio and Q² against both the kinetic '
     'fit cost and the extrapolation performance.')
item('CLAIM', 'The ratio tracks out-of-domain performance at ρ = +0.79 to +0.82 '
     '(p ≈ 0.03) and is the most stable of the candidate statistics across the two '
     'data generations. In-domain Q² reaches only ρ = +0.32 (p = 0.48).')
item('CLAIM', 'u<sub>X</sub> alone is comparably predictive (ρ = −0.86 to −0.96) '
     'but has no scale; shared is strong on one data generation (+0.96) and fails '
     'on the other (+0.43). State this honestly &mdash; the ratio is chosen for '
     'stability and interpretability, not because it wins every single correlation.')
item('CAUTION', 'Add the sentence that the evaluation regime decides the answer: a '
     'single-shot evaluation gives the opposite sign to the rolling scheme, and only '
     'the rolling scheme tests what the metric is about.')

sec('5.9', 'Independent case study')
say('Second process, two models defined by the source literature rather than '
    'constructed here.')
item('VISUAL', '<b>Fig. 5.10</b> &mdash; three panels: each block alone, the '
     'commonality split, and both metrics on a log scale against the reference '
     '(<font face="Courier">deck_cs2.png</font>).')
item('TABLE', '<b>Table 5.5</b> &mdash; results table in the same format as '
     'Table 5.2 (<font face="Courier">table_cs2_results.xlsx</font>).')
item('CLAIM', 'Breaking the reverse step collapses the mechanistic block from '
     'Q² = 0.95 to 0.59 while the measured block rises from 0.91 to 0.97. The '
     'correct model sits above the reference and the broken one below, on both '
     'responses, with P(&gt;1) = 0.00 for the broken model in every replicate.')
item('CAUTION', 'Report P(&gt;1) = 0.95 and 0.80 for the <i>correct</i> model and '
     'explain it rather than hiding it: u<sub>X</sub> is 0.03 – 1.65 %, so the '
     'magnitude is unstable even where the side of the threshold is not. This is '
     'the §2.7 degeneracy appearing exactly where predicted.')
item('CLAIM', 'Report the sample-size sensitivity: at 200 batches both responses '
     'reach P(&gt;1) = 1.00 for the correct model and 0.00 for the broken one, and '
     'the relative spread falls to 0.25 – 0.29. '
     '(<font face="Courier">check_n_batches.xlsx</font>)')

sec('5.10', 'Scope conditions and limitations')
say('Collect every limitation in one place, with evidence. A chapter that states '
    'its own limits precisely is harder to attack than one that leaves them to be '
    'found.')
item('CLAIM', '<b>Coverage is not a quality criterion.</b> Across the candidates '
     'coverage correlates with the kinetic fit cost at ρ = +0.93 (p = 0.003) and '
     'with the ratio at ρ = −0.82 &mdash; the worst mechanism has the highest '
     'coverage. Selecting on both jointly would systematically choose worse models.')
item('CLAIM', '<b>Block layout matters.</b> Truncating the measured block while the '
     'mechanistic block keeps the full horizon drives u<sub>X</sub> to zero and can '
     'invert the ranking. Report the layout that was used and why.')
item('CLAIM', '<b>Insensitivity checks that passed.</b> Giving the mechanistic block '
     'the full input programme rather than the truncated one changes no conclusion '
     '(<font face="Courier">check_input_horizon.xlsx</font>).')
item('CAUTION', 'State the statistical limits: seven and two candidates, so rank '
     'correlations are indicative rather than established; and the second case study '
     'is a reconstruction.')
S.append(PageBreak())

# ================================================================ CHAPTER 6
chapter(6, 'Conclusions', '1,200 words')
sec('6.1', 'Summary of the contribution')
say('One paragraph per claim, in the order they were established: the ratio built '
    'on unique contributions has an intrinsic reference; it requires cross-validated '
    'Q²; it separates sound from broken mechanisms on two independent processes; '
    'and it tracks out-of-domain performance where in-domain fit does not.')
sec('6.2', 'Practical recommendation')
say('State the reporting rule compactly: report the ratio with coverage beside it, '
    'report the side of the reference rather than the magnitude when u<sub>X</sub> '
    'is small, and state the block layout.')
item('TO MAKE', '<b>Table 6.1</b> &mdash; one-page practitioner summary: what to '
     'compute, what to report, when the metric does not apply.')
sec('6.3', 'Limitations')
say('Short and direct; point to §5.10 rather than repeating it.')
sec('6.4', 'Future work')
say('Obtaining the original input data for the second case study to convert the '
    'reconstruction into a replication; a defensible rule for when u<sub>X</sub> is '
    'too small to divide by, based on its own cross-validation standard error; '
    'experimental design to reduce block redundancy and raise coverage; and a larger '
    'candidate family to test the threshold more severely.')

S.append(Spacer(1, 10))
note('<b>Figures still to be produced:</b> Fig. 2.1 (decomposition schematic), '
     'Table 2.1 (quantity definitions), Table 2.2 (applicability checklist), '
     'Fig. 3.1 (generic Måge schematic), Table 4.1 (candidate construction), '
     'Table 4.4 (design comparison), Table 6.1 (practitioner summary). '
     'Everything else named above already exists.')

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=15*mm, bottomMargin=15*mm,
                        title='Thesis Chapter Outline')


def page_num(canvas, doc_):
    canvas.saveState(); canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#777777'))
    canvas.drawCentredString(A4[0]/2, 9*mm, str(canvas.getPageNumber()))
    canvas.restoreState()


doc.build(S, onFirstPage=page_num, onLaterPages=page_num)
print(f'Saved {OUT}  ({os.path.getsize(OUT)/1024:.0f} kB)')
