# Hybrid model M2M — a modified contribution metric

Thesis code for measuring how much a hybrid process model owes to its
physics and how much to its data, and for testing a **modified M2M metric**
built from *unique* (commonality) contributions instead of raw sequential
sums of squares.

## The idea

The original metric compares the sequential sums of squares carried by the
mechanistic block **M** and the measured block **X**:

```
M2M  = SS_M / SS_X
```

Because the variance the two blocks *share* is absorbed into the numerator,
this ratio has no natural reference point — it is large whenever the blocks
overlap, whatever the mechanism is actually worth. The modified metric uses
only the unique terms:

```
M2M* = u_M / u_X          u_M = Q2(all) - Q2(all without M)
                          u_X = Q2(all) - Q2(all without X)
```

so its reference is intrinsic: **above 1** the mechanism supplies more unique
information than the measurements, **below 1** the measurements do.

## What the two case studies show

| | case study 1 — two-stage reactor | case study 2 — urethane |
|---|---|---|
| candidates | 7 mechanisms of graded quality | 2 (correct vs missing reverse step) |
| blocks | X1 → M2 → X2 | M → X |
| finding | M2M\* tracks out-of-domain performance (ρ = +0.79, p = 0.036); the shared term alone does not (ρ = +0.43, p = 0.34) | M2M\* puts the correct and broken model on **opposite sides of 1** (3.49 vs 0.027); the conventional ratio puts both above 1 |

The weakness is structural and is reported rather than hidden: `u_X` is what
the measurements add *once the mechanism is present*, so a good mechanism on
a well-measured process drives it toward zero and the ratio loses precision
and meaning. Report the **side of 1**, not the magnitude, and report
`coverage = (u_M + u_X)/joint` alongside it.

Full write-up: **`Thesis_TwoCaseStudy_Report.pdf`** (9 pages, 7 figures).

## Layout

Two scripts, one per case study, each self-contained and stage-based.

```
case_study_1.py       two-stage batch reactor, 7 candidate mechanisms
    commonality         u_M / u_X / shared and the modified ratio      [dataset C]
    flatregion          ratio distribution over the 1/2/3-SE regions   [dataset A]
    q2cost              what each stopping rule costs in Q2            [dataset A]
    rolling             iterative prediction + domain extrapolation    [dataset C]

case_study_2.py       urethane semi-batch reactor, 2 models
    replicate           reproduce the authors' published mole profiles
    compare             published vs reconstruction + noise-seed test
    feeds               re-derive the unpublished feed programme (slow)
    m2m                 the M2M comparison, both block layouts

build_report_figures.py   figures for the report
build_two_case_report.py  builds Thesis_TwoCaseStudy_Report.pdf
regen_corrected_data.py / Data_Generation_corrected.py
                          regenerate the case study 1 RG_*.xlsx data
```

Case study 1 deliberately carries **two datasets**, because both are still in
use and they do not agree on the ranking of the seven mechanisms — that
disagreement is itself a reported finding:

| | source | used by |
|---|---|---|
| A "original" | `X1/X2/Y/M2.xlsx` (+ `_mis` for M6), parameters hard-coded | `flatregion`, `q2cost` |
| C "regenerated" | `RG_*.xlsx`, corrected sigma and paper-style stage-2 ICs | `commonality`, `rolling` |

## Reproducing

```bash
python case_study_1.py                 # commonality + flatregion + q2cost
python case_study_1.py rolling         # extrapolation test (~90 s)
python case_study_2.py                 # replicate + compare + m2m
python build_report_figures.py         # figures
python build_two_case_report.py        # the PDF
```

Results are written as `case_study_*.xlsx`. The report reads its numbers back
from those workbooks at build time rather than having them retyped, so it
cannot drift from the analyses.

Requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `openpyxl`,
`reportlab`, `pillow`. The urethane scripts integrate with `scipy.solve_ivp`
rather than GEKKO; the authors' DAE is index-1 with an explicit algebraic
part, so the substitution is exact.

## Provenance and known gaps

The urethane process model, its parameters and its noise levels are taken from
the original authors' notebooks and are reproduced here from published values —
their notebooks and the source papers are **not** included in this repository.

Two things were corrected against those notebooks: the measurement noise on
species C is `5e-3`, not `5e-2`; and the published parameter table is correct,
contrary to an earlier conclusion here that it could not have produced the
published figures.

Still open:

- The authors' input workbook (26+ batches) is not in their archive. The 40
  batches used here are sampled around a single validated run, so case study 2
  is a reconstruction rather than an exact replication.
- No threshold has been established for "`u_X` too small to divide by". A
  defensible rule would compare `u_X` against its own cross-validation
  standard error.
