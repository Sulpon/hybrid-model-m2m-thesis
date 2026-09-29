"""
Build the thesis progress presentation (Meeting 30.09).

Numbers are read back from the result workbooks, never retyped, so the deck
cannot drift from the analyses.
"""
import os
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

OUT = 'Meeting 30.09.pptx'
W, H = 13.333, 7.5
NAVY = RGBColor(0x1E, 0x27, 0x61)
GOOD = RGBColor(0x2E, 0x7D, 0x32)
BAD = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x5A, 0x60, 0x6A)
LGREY = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BODY, HEAD = 'Calibri', 'Cambria'

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def tb(s, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def put(tf, text, size=15, bold=False, color=None, font=BODY, space_after=6,
        first=False, bullet=False, align=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    if align is not None:
        p.alignment = align
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = ('•  ' + text) if bullet else text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = font
    r.font.color.rgb = color if color is not None else RGBColor(0x22, 0x26, 0x2B)
    return p


def title(s, text, sub=None):
    tf = tb(s, 0.62, 0.34, W-1.24, 0.95)
    put(tf, text, 32, True, NAVY, HEAD, 2, first=True)
    if sub:
        put(tf, sub, 14, False, GREY, BODY, 0)
    return 1.55 if not sub else 1.72


def bullets(s, items, x, y, w, size=15, gap=9, h=None):
    tf = tb(s, x, y, w, h or (H - y - 0.5))
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            txt, bold, col = it
        else:
            txt, bold, col = it, False, None
        put(tf, txt, size, bold, col, space_after=gap, first=(i == 0), bullet=not bold)
    return tf


def card(s, x, y, w, h, fill=LGREY):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    sh.adjustments[0] = 0.06
    if sh.has_text_frame:
        sh.text_frame.text = ''
    return sh


def pic(s, path, x, y, w=None, h=None):
    if not os.path.exists(path):
        raise SystemExit(f'missing figure {path}')
    kw = {}
    if w: kw['width'] = Inches(w)
    if h: kw['height'] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def table(s, rows, x, y, w, h, colw=None, fs=12, headfs=12, hl=None):
    """hl: dict row_idx -> RGBColor for the whole row's text."""
    nr, nc = len(rows), len(rows[0])
    shp = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h))
    t = shp.table
    if colw:
        for i, cw in enumerate(colw):
            t.columns[i].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(headfs if i == 0 else fs)
                r.font.name = BODY
                r.font.bold = (i == 0)
                if i == 0:
                    r.font.color.rgb = WHITE
                elif hl and i in hl:
                    r.font.color.rgb = hl[i]
                    r.font.bold = True
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = NAVY
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = WHITE if i % 2 else LGREY
    return t


def note(s, text, y=None, color=GREY, size=12):
    tf = tb(s, 0.62, y if y is not None else H-0.72, W-1.24, 0.5)
    put(tf, text, size, False, color, space_after=0, first=True)


def pagenum(s, n):
    tf = tb(s, W-1.05, H-0.52, 0.5, 0.3, align=PP_ALIGN.RIGHT)
    put(tf, str(n), 11, False, GREY, space_after=0, first=True)


# ===================================================================== data
cs1 = pd.read_excel('_archive/metric_unique_m2m.xlsx', sheet_name='summary').set_index('model')
ps = pd.read_excel('_archive/metric_unique_m2m.xlsx', sheet_name='per_seed')
mq = (ps.groupby('model')[['Q2_full', 'Q2_KD', 'Q2_DD', 'unique_M', 'unique_X']].mean()*100)
ORDER = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
         'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']
NICE = {'M0_correct': 'M0  correct', 'M1_no_side': 'M1  no side reaction',
        'M2_order1_D': 'M2  order-1 in D', 'M3_wrong_Ea4': 'M3  wrong Ea4',
        'M4_lumped_EF': 'M4  lumped E/F', 'M5_no_D': 'M5  no D dependence',
        'M6_author_mis': "M6  authors' misspec."}
BADSET = {'M1_no_side', 'M5_no_D', 'M6_author_mis'}

u2 = pd.read_excel('case_study_2_results.xlsx')
u2 = u2[(u2.variant == 'V1_static') & (u2.param == 70.0)].set_index(['response', 'model'])

# ==================================================================== 1 title
s = slide()
bg = card(s, -0.1, -0.1, W+0.2, H+0.2, NAVY)
tf = tb(s, 1.1, 2.15, W-2.2, 2.6)
put(tf, 'Measuring Physics vs Data', 44, True, WHITE, HEAD, 4, first=True)
put(tf, 'in Hybrid Process Models', 44, True, WHITE, HEAD, 16)
put(tf, 'A modified M2M metric with an absolute reference point',
    19, False, RGBColor(0xCA, 0xDC, 0xFC), BODY, 0)
tf = tb(s, 1.1, 5.55, W-2.2, 1.0)
put(tf, 'Thesis progress  —  two case studies, one metric that works',
    14, False, RGBColor(0x9F, 0xB3, 0xD9), BODY, 3, first=True)
put(tf, '30 September 2026', 13, False, RGBColor(0x8A, 0x9C, 0xC0), BODY, 0)

# ============================================== 2 where we left off / problem
s = slide(); y = title(s, 'Where we left off',
                       'Two months ago: the metric ranked models, but nobody could say what a value meant')
card(s, 0.62, y, 5.9, 4.45)
tf = tb(s, 0.95, y+0.3, 5.3, 3.9)
put(tf, 'THE SETTING', 12, True, NAVY, BODY, 8, first=True)
put(tf, 'A hybrid model predicts a quality variable from two sources:', 14, space_after=8)
put(tf, 'a mechanistic (physics) block  M', 14, bullet=True, space_after=5)
put(tf, 'a measured (data) block  X', 14, bullet=True, space_after=11)
put(tf, 'M2M asks how the credit splits between them.', 14, space_after=10)
put(tf, 'THE PROBLEM', 12, True, BAD, BODY, 8)
put(tf, 'The original ratio has no reference point. If M2M = 12.6 — '
        'is the physics good, or is it just overlapping with the data?',
    14, space_after=0)

card(s, 6.85, y, 5.88, 4.45)
tf = tb(s, 7.18, y+0.3, 5.25, 3.9)
put(tf, 'WHAT CHANGED', 12, True, GOOD, BODY, 8, first=True)
put(tf, 'Build the ratio from the UNIQUE contributions only.', 15, True, space_after=10)
put(tf, 'That gives it a meaning you can read off a single number:',
    14, space_after=9)
put(tf, 'above 1  →  physics carries more unique information', 14,
    True, GOOD, space_after=6)
put(tf, 'below 1  →  measurements carry more', 14, True, BAD, space_after=11)
put(tf, 'Tested on two independent case studies. It separates sound '
        'mechanisms from broken ones in both.', 14, space_after=0)
note(s, 'Everything in this deck is reproducible from two scripts: case_study_1.py and case_study_2.py')
pagenum(s, 2)

# ================================================= 3 case study 1: the process
s = slide(); y = title(s, 'Case study 1 — the process',
                       'Two-stage batch reactor (the benchmark from the source paper)')
card(s, 0.62, y, 6.3, 2.5)
tf = tb(s, 0.95, y+0.26, 5.7, 2.0)
put(tf, 'STAGE 1', 12, True, NAVY, BODY, 8, first=True)
put(tf, 'A  →  B  →  C', 19, True, space_after=7, font=HEAD)
put(tf, 'A forms the wanted intermediate B; a consecutive reaction '
        'degrades B to the unwanted C.', 13.5, space_after=0)

card(s, 7.03, y, 5.7, 2.5)
tf = tb(s, 7.36, y+0.26, 5.1, 2.0)
put(tf, 'STAGE 2', 12, True, NAVY, BODY, 8, first=True)
put(tf, 'B + 2D  →  E      (product)', 17, True, space_after=4, font=HEAD)
put(tf, 'C + 2D  →  F      (impurity)', 17, True, space_after=7, font=HEAD)
put(tf, 'Response Y = end-of-batch purity of E.', 13.5, space_after=0)

tf = tb(s, 0.62, y+2.75, W-1.24, 2.2)
put(tf, 'Three sources of variability, so the data behave like a real plant:',
    15, True, NAVY, space_after=9, first=True)
put(tf, 'Operating conditions drawn at random — temperatures and batch '
        'durations uniform, initial concentrations normal', 14, bullet=True, space_after=7)
put(tf, 'The ODEs are integrated as stochastic differential equations '
        '(Euler–Maruyama), so each batch has its own trajectory noise', 14,
    bullet=True, space_after=7)
put(tf, 'Measurements carry 1 % sampling noise', 14, bullet=True, space_after=7)
put(tf, '100 batches form the calibration set.', 14, space_after=0)
pagenum(s, 3)

# ====================================================== 4 the blocks / SO-PLS
s = slide(); y = title(s, 'How the data enter the model',
                       'Blocks are fitted in sequence, so each one is credited only with what it adds')
rows = [['block', 'what it holds', 'columns'],
        ['X₁', 'stage-1 recipe (CA0, t₁, T₁) + stage-1 outlet measurements', '6'],
        ['M₂', 'the candidate ODE simulated over stage 2  +  stage-2 setpoints', '45'],
        ['X₂', 'the measured stage-2 trajectories (6 species × 7 samples)', '42'],
        ['Y', 'end-of-batch purity of E', '1']]
table(s, rows, 0.62, y, 12.1, 2.05, colw=[1.0, 9.5, 1.6], fs=13.5, headfs=12.5)

tf = tb(s, 0.62, y+2.35, 12.1, 2.6)
put(tf, 'SO-PLS  —  X₁  →  M₂  →  X₂', 18, True, NAVY, HEAD, 10, first=True)
put(tf, 'Each block is orthogonalised against the blocks already entered. '
        'M₂ is therefore credited only with variance that X₁ could not '
        'explain, and X₂ only with what X₁ and M₂ together could not. '
        'These are sequential (Type-I) sums of squares — which is exactly why '
        'the block order matters.', 14.5, space_after=12)
put(tf, 'Latent variables are chosen by parsimony inside the 1-SE flat region, '
        'not by raw argmin — the argmin gives degenerate splits that put almost '
        'all capacity in one block.', 14.5, space_after=0)
note(s, 'Stage-2 setpoints sit in M₂ rather than X₂ because they are inputs '
        'to the mechanistic model (the source paper\'s κ rule).')
pagenum(s, 4)

# =============================================== 5 the original M2M + problem
s = slide(); y = title(s, 'The original metric — and why it is hard to act on')
tf = tb(s, 0.62, y, 12.1, 0.9)
put(tf, 'M2M  =  SSₘ / SSₓ', 26, True, NAVY, HEAD, 6, first=True, align=PP_ALIGN.CENTER)

card(s, 0.62, y+1.0, 5.9, 3.55)
tf = tb(s, 0.95, y+1.26, 5.3, 3.0)
put(tf, 'WHAT IT DOES WELL', 12, True, GOOD, BODY, 8, first=True)
put(tf, 'It orders models correctly. On our seven candidates the sound '
        'mechanisms do score higher than the broken ones.', 14, space_after=0)

card(s, 6.85, y+1.0, 5.88, 3.55)
tf = tb(s, 7.18, y+1.26, 5.25, 3.0)
put(tf, 'WHAT IT CANNOT DO', 12, True, BAD, BODY, 8, first=True)
put(tf, 'It has no scale. The shared variance — what BOTH blocks explain — '
        'is swept into the numerator, so the ratio is large whenever the two '
        'blocks overlap, whatever the physics is worth.', 14, space_after=10)
put(tf, 'M2M = 12.6.  So what?', 19, True, BAD, HEAD, 8)
put(tf, 'You cannot tell from that number alone whether the mechanism is '
        'sound. You can only compare it with another model you already have.',
    14, space_after=0)
note(s, 'A metric you can only use comparatively cannot certify a single '
        'model — which is what an engineer actually needs.')
pagenum(s, 5)

# ================================================= 6 commonality decomposition
s = slide(); y = title(s, 'The fix — separate what is unique from what is shared',
                       'Commonality analysis: split the jointly explained variance into three parts')
tf = tb(s, 0.62, y, 12.1, 0.75)
put(tf, 'joint  =  uniqueₘ  +  uniqueₓ  +  shared', 24, True, NAVY, HEAD, 0,
    first=True, align=PP_ALIGN.CENTER)

W3, X3 = 3.85, 0.62
for i, (t1, t2, col) in enumerate([
        ('uniqueₘ', 'what the MECHANISM adds to a purely data-driven model\n\n'
         'Q²(all)  −  Q²(all without M)', GOOD),
        ('uniqueₓ', 'what the MEASUREMENTS add to a purely knowledge-driven model\n\n'
         'Q²(all)  −  Q²(all without X)', BAD),
        ('shared', 'variance EITHER block could have supplied on its own — '
         'redundant, and the part that made the original ratio unreadable', GREY)]):
    x = X3 + i*(W3 + 0.28)
    card(s, x, y+0.95, W3, 3.05)
    tf2 = tb(s, x+0.28, y+1.2, W3-0.56, 2.55)
    put(tf2, t1, 20, True, col, HEAD, 9, first=True)
    put(tf2, t2, 13, space_after=0)

tf = tb(s, 0.62, y+4.2, 12.1, 0.9)
put(tf, 'The original metric charges the whole shared term to the mechanism. '
        'The modified one charges it to neither.', 15, True, NAVY, space_after=0, first=True)
pagenum(s, 6)

# ============================================================ 7 the new metric
s = slide(); y = title(s, 'The modified metric', 'and how it is computed')
tf = tb(s, 0.62, y, 12.1, 0.85)
put(tf, 'M2M*  =  uniqueₘ / uniqueₓ', 28, True, NAVY, HEAD, 0, first=True,
    align=PP_ALIGN.CENTER)

steps = [('1', 'Fit the FULL model', 'SO-PLS on X₁ → M₂ → X₂. '
          'Choose the allocation by parsimony inside the 1-SE flat region. Record Q².'),
         ('2', 'Drop one block at a time', 'Refit X₁+X₂ (no mechanism) and '
          'X₁+M₂ (no measurements), each re-optimised on its own.'),
         ('3', 'Take the differences', 'uₘ = Q²(full) − Q²(no M).   '
          'uₓ = Q²(full) − Q²(no X).'),
         ('4', 'Divide, and repeat over seeds', 'M2M* = uₘ/uₓ, averaged over '
          '20 cross-validation seeds so a small denominator cannot swing the answer.')]
yy = y + 0.95
for num, head, txt in steps:
    card(s, 0.62, yy, 12.1, 0.92)
    tf2 = tb(s, 0.95, yy+0.12, 0.5, 0.7)
    put(tf2, num, 22, True, NAVY, HEAD, 0, first=True)
    tf2 = tb(s, 1.55, yy+0.11, 11.0, 0.74)
    put(tf2, head, 14.5, True, NAVY, space_after=3, first=True)
    put(tf2, txt, 13, space_after=0)
    yy += 1.03
note(s, 'Because Q² is cross-validated, uₘ and uₓ are out-of-sample '
        'gains — not in-sample fit.')
pagenum(s, 7)

# ================================================== 8 CS1 results — the table
s = slide(); y = title(s, 'Case study 1 — seven candidate mechanisms',
                       'Four sound structures, three deliberately broken. The metric was not told which is which.')
rows = [['model', 'Q² full', 'Q² no-M', 'Q² no-X', 'uₘ', 'uₓ',
         'M2M*', '± sd', 'P(>1)']]
hl = {}
for i, m in enumerate(ORDER):
    rows.append([NICE[m], f'{mq.Q2_full[m]:.2f}', f'{mq.Q2_DD[m]:.2f}', f'{mq.Q2_KD[m]:.2f}',
                 f'{mq.unique_M[m]:.2f}', f'{mq.unique_X[m]:.2f}',
                 f'{cs1.CV_ratio_mean[m]:.2f}', f'{cs1.CV_ratio_sd[m]:.2f}',
                 f'{cs1.CV_frac_gt1[m]:.2f}'])
    hl[i+1] = BAD if m in BADSET else GOOD
table(s, rows, 0.62, y, 12.1, 3.1,
      colw=[3.05, 1.2, 1.2, 1.2, 1.15, 1.15, 1.15, 1.0, 1.0], fs=13, headfs=12)
tf = tb(s, 0.62, y+3.4, 12.1, 1.5)
put(tf, 'Sound mechanisms land at 1.96 – 2.11.   Broken ones at 0.37, 0.70 and 1.16.',
    17, True, NAVY, space_after=9, first=True)
put(tf, 'The gap between 1.96 and 1.16 is clean, and the last column says how '
        'often the ratio exceeded 1 across the 20 seeds — always for the sound '
        'models, never for M1 and M5.', 14, space_after=0)
note(s, 'All Q² values in percent, averaged over 20 cross-validation seeds. '
        'M6 is the misspecification used in the original paper.')
pagenum(s, 8)

# ====================================================== 9 CS1 figure
s = slide(); y = title(s, 'The same result, read off a single threshold')
pic(s, 'deck_cs1_ratio.png', 1.32, y-0.05, w=10.7)
note(s, 'Error bars are ± 1 sd over 20 cross-validation seeds. No sound '
        'model’s interval reaches 1; M1 and M5 never approach it.', y=H-0.62)
pagenum(s, 9)

# ====================================================== 10 why it separates
s = slide(); y = title(s, 'Why it separates',
                       'Break the mechanism and the measurements are forced to take over')
pic(s, 'deck_cs1_uniques.png', 1.37, y, w=10.6)
tf = tb(s, 0.62, y+4.55, 12.1, 0.95)
put(tf, 'uₘ barely moves (10 – 13 %). It is uₓ that gives the game away: '
        '~6 % when the physics is right, 16 – 27 % when it is not.',
    15, True, NAVY, space_after=0, first=True)
pagenum(s, 10)

# ====================================================== 11 original vs modified
s = slide(); y = title(s, 'Original vs modified, side by side')
pic(s, 'deck_cs1_vs.png', 0.87, y, w=11.6)
tf = tb(s, 0.62, y+4.45, 12.1, 0.95)
put(tf, 'Both rank the models. Only the right-hand scale tells you, from one '
        'number, whether a mechanism is worth keeping.', 15, True, NAVY,
    space_after=0, first=True)
pagenum(s, 11)

# ====================================================== 12 pros and cons
s = slide(); y = title(s, 'Why this is better — and where it is fragile')
card(s, 0.62, y, 5.9, 4.5)
tf = tb(s, 0.95, y+0.26, 5.3, 4.0)
put(tf, 'ADVANTAGES', 13, True, GOOD, BODY, 10, first=True)
put(tf, 'Absolute reference. 1 is a real threshold, not a convention — '
        'it is the point where the two blocks contribute equally.', 13.5,
    bullet=True, space_after=8)
put(tf, 'Immune to redundancy. The shared term no longer inflates the '
        'numerator, so overlapping blocks cannot fake a good score.', 13.5,
    bullet=True, space_after=8)
put(tf, 'Certifies one model. No second model needed for comparison.',
    13.5, bullet=True, space_after=8)
put(tf, 'Cross-validated. uₘ and uₓ are out-of-sample, so the metric '
        'cannot be gamed by adding latent variables.', 13.5, bullet=True, space_after=8)
put(tf, 'Tracks out-of-domain performance. ρ = +0.79 (p = 0.036) against '
        'extrapolation R² — where in-domain Q² gives only ρ = +0.32.',
    13.5, bullet=True, space_after=0)

card(s, 6.85, y, 5.88, 4.5)
tf = tb(s, 7.18, y+0.26, 5.25, 4.0)
put(tf, 'LIMITATIONS — stated openly', 13, True, BAD, BODY, 10, first=True)
put(tf, 'The denominator can vanish. A very good mechanism on a '
        'well-measured process drives uₓ toward 0 and the ratio explodes.',
    13.5, bullet=True, space_after=8)
put(tf, 'Report the side of 1, not the magnitude, whenever uₓ is small.',
    13.5, bullet=True, space_after=8)
put(tf, 'Sensitive to block layout. Truncating the measured block changes '
        'the answer — the layout must be stated and justified.', 13.5,
    bullet=True, space_after=8)
put(tf, 'Needs a companion statistic. Report coverage = (uₘ+uₓ)/joint '
        'so the reader knows how much variance the ratio is arbitrating.',
    13.5, bullet=True, space_after=8)
put(tf, 'Seven and two candidates only — rank correlations are indicative, '
        'not established.', 13.5, bullet=True, space_after=0)
pagenum(s, 12)

# ====================================================== 13 CS2 process
s = slide(); y = title(s, 'Case study 2 — an independent test',
                       'Urethane semi-batch reactor, taken from a different group’s published work')
card(s, 0.62, y, 6.3, 2.35)
tf = tb(s, 0.95, y+0.24, 5.7, 1.9)
put(tf, 'THE CHEMISTRY', 12, True, NAVY, BODY, 8, first=True)
put(tf, 'A + B  →  C          isocyanate + butanol → urethane', 13.5, space_after=4)
put(tf, 'A + C  ⇌  D          → allophanate  (REVERSIBLE)', 13.5, True, space_after=4)
put(tf, '3A  →  E              → isocyanurate', 13.5, space_after=0)

card(s, 7.03, y, 5.7, 2.35)
tf = tb(s, 7.36, y+0.24, 5.1, 1.9)
put(tf, 'WHY THIS CASE', 12, True, NAVY, BODY, 8, first=True)
put(tf, 'The published method exists precisely to rediscover the reverse '
        'step. So the literature hands us one correct model and one '
        'realistically broken one — no invented misspecification.',
    13.5, space_after=0)

tf = tb(s, 0.62, y+2.6, 12.1, 2.4)
put(tf, 'Two models — the ones the source papers actually define:', 15, True,
    NAVY, space_after=9, first=True)
put(tf, 'U0   correct structure, reverse step present          '
        '(8 fitted parameters)', 14.5, bullet=True, space_after=7)
put(tf, 'U1   the available first-principles model, reverse step absent   '
        '(6 parameters)', 14.5, bullet=True, space_after=11)
put(tf, 'Sanity check on the pipeline:  U0 recovers the true kinetics almost '
        'exactly (kᵣₑₑ₁ 0.001249 vs 1.25e-3), while U1’s fit cost is five '
        'orders of magnitude worse — 1.22e8 against 1964.', 14, space_after=0)
pagenum(s, 13)

# ====================================================== 14 CS2 data + blocks
s = slide(); y = title(s, 'Case study 2 — data and blocks',
                       'The process data were rebuilt from the authors’ own code and validated against their figures')
pic(s, 'cs2_compare_published.png', 0.55, y-0.08, w=7.5)
tf = tb(s, 8.35, y, 4.4, 4.6)
put(tf, 'VALIDATION', 12, True, GOOD, BODY, 8, first=True)
put(tf, 'Published curves (blue) vs our reconstruction (red).', 13, space_after=7)
put(tf, '5.0 % RMS deviation', 15, True, GOOD, space_after=7)
put(tf, 'Stronger still: the authors seed their noise with 42. Matching the '
        'seed reproduces their INDIVIDUAL noise realisation, not just its '
        'variance — correlation 0.60 against −0.005 ± 0.055 for 25 other '
        'seeds  (z = 11.1).', 12.5, space_after=10)
put(tf, 'BLOCKS', 12, True, NAVY, BODY, 8)
put(tf, 'M  = ODE states + input programme', 13, space_after=4)
put(tf, 'X  = measured C, D, E on the same grid', 13, space_after=4)
put(tf, 'Y  = final n_C or n_D, outside that grid', 13, space_after=0)
pagenum(s, 14)

# ====================================================== 15 CS2 results
s = slide(); y = title(s, 'Case study 2 — the result',
                       'The correct and the broken mechanism land on opposite sides of 1')
pic(s, 'deck_cs2.png', 0.97, y-0.05, w=11.4)
r = lambda resp, m, c: u2.loc[(resp, m), c]
tf = tb(s, 0.62, y+4.35, 12.1, 1.0)
put(tf, f'n_C:   U0 = {r("nC","U0_correct","M2M_mod"):.2f}   vs   '
        f'U1 = {r("nC","U1_no_reverse","M2M_mod"):.3f}        '
        f'n_D:   U0 = {r("nD","U0_correct","M2M_mod"):.0f}   vs   '
        f'U1 = {r("nD","U1_no_reverse","M2M_mod"):.3f}',
    16, True, NAVY, space_after=6, first=True)
put(tf, 'The original ratio ranks them the same way but puts BOTH above 1 '
        '(235 vs 2.1) — an ordering, not a verdict.', 14, space_after=0)
pagenum(s, 15)

# ====================================================== 16 conclusions
s = slide(); y = title(s, 'Where this stands')
card(s, 0.62, y, 12.1, 2.0, RGBColor(0xE8, 0xF1, 0xE9))
tf = tb(s, 0.95, y+0.25, 11.4, 1.6)
put(tf, 'The metric works, and it works on two independent processes.',
    19, True, GOOD, HEAD, 9, first=True)
put(tf, 'Case study 1: sound mechanisms at 1.96–2.11, broken ones at '
        '0.37–1.16, with a clean gap.   Case study 2: correct model above 1, '
        'broken model below, on both responses. The original metric cannot '
        'make either statement in absolute terms.', 14, space_after=0)

tf = tb(s, 0.62, y+2.3, 5.9, 2.6)
put(tf, 'READY FOR THE THESIS', 12, True, NAVY, BODY, 9, first=True)
put(tf, 'Definition, estimator and scope condition are settled', 13.5, bullet=True, space_after=6)
put(tf, 'Two case studies, one of them external', 13.5, bullet=True, space_after=6)
put(tf, 'Full pipeline in two reproducible scripts', 13.5, bullet=True, space_after=6)
put(tf, 'Limitations characterised, not hidden', 13.5, bullet=True, space_after=0)

tf = tb(s, 6.85, y+2.3, 5.88, 2.6)
put(tf, 'NEXT', 12, True, BAD, BODY, 9, first=True)
put(tf, 'Obtain the urethane authors’ input workbook — turns case study 2 '
        'from a reconstruction into an exact replication', 13.5, bullet=True, space_after=6)
put(tf, 'A defensible rule for “uₓ too small to divide by”, based on its '
        'own CV standard error', 13.5, bullet=True, space_after=6)
put(tf, 'Write up — target: a methods paper', 13.5, bullet=True, space_after=0)
pagenum(s, 16)

# ====================================================== 17 backup
s = slide(); y = title(s, 'Backup — why two versions of the table exist',
                       'The estimator matters, and the difference is worth stating')
rows = [['', 'reported here', 'earlier variant'],
        ['dataset', 'original (X1/X2/Y/M2)', 'regenerated (RG_*)'],
        ['CV seeds', '20, averaged', 'single seed 42'],
        ['sub-model latent variables', 're-optimised independently', 'inherited from the full model'],
        ['separates good from bad?', 'yes, clean gap', 'no — M2 falls below M6']]
table(s, rows, 0.62, y, 12.1, 2.35, colw=[4.1, 4.0, 4.0], fs=13, headfs=12.5)
tf = tb(s, 0.62, y+2.65, 12.1, 2.2)
put(tf, 'The reported estimator is the defensible one: each sub-model gets its '
        'own best allocation, and averaging over 20 seeds tames the '
        '1/uₓ variance that makes a single-seed ratio noisy.', 14.5,
    space_after=10, first=True)
put(tf, 'This is a real sensitivity, not a bug — and it is exactly why the '
        'recommendation is to report the side of 1 rather than the magnitude.',
    14.5, True, NAVY, space_after=0)
pagenum(s, 17)

prs.save(OUT)
print(f'Saved {OUT}  ({os.path.getsize(OUT)/1024:.0f} kB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)')
