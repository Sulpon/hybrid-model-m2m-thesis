"""
Digitise the authors' stored "Baseline Simulation Results" figure.

The notebooks in "Urethane case study.zip" ship their output figures but not
the data behind them, and the input workbook they read is not in the archive.
To compare our reconstruction against theirs over the whole batch -- rather
than against the handful of points that can be read by eye -- the published
curves are recovered from the PNG itself.

The figure is a plain matplotlib render, which makes this reliable rather
than approximate: the traces are exactly C0 (31,119,180), the grid is exactly
(176,176,176), and the axes spines are solid black runs. Panels are located
from the spines, the pixel-to-data map comes from the gridlines, and the
calibration is then checked against quantities that are known independently
(all three of nC, nD, nE start at 0; nA0, nB0 and V0 are read at t=0).

Writes urethane_paper_curves.npz with one (t, y) pair per panel.
"""
import numpy as np
from PIL import Image

PNG = r'urethane_authors_tmp/out/urethane_C_c0_1.png'
C0 = np.array([31, 119, 180])
GRID = 176

# panel order as laid out by the authors, with the y tick values printed on each
PANELS = [
    ('A', 0, 0, [0.00, 0.01, 0.02, 0.03, 0.04]),
    ('B', 0, 1, [0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175]),
    ('C', 0, 2, [0.00, 0.05, 0.10, 0.15, 0.20]),
    ('D', 1, 0, [0.00, 0.05, 0.10, 0.15, 0.20]),
    ('E', 1, 1, [0.00000, 0.00005, 0.00010, 0.00015, 0.00020,
                 0.00025, 0.00030, 0.00035]),
    ('V', 1, 2, [3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]),
]
XTICKS = [0.0, 20.0, 40.0, 60.0, 80.0]


def _runs(idx, gap=3):
    """Collapse neighbouring indices into one representative each."""
    if len(idx) == 0:
        return []
    out, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(int(np.mean(cur))); cur = [v]
    out.append(int(np.mean(cur)))
    return out


def find_panels(a):
    dark = a.sum(2) < 250
    cols = _runs(np.nonzero(dark.sum(0) > 200)[0])
    rows = _runs(np.nonzero(dark.sum(1) > 200)[0])
    xs = [(cols[i], cols[i+1]) for i in range(0, len(cols)-1, 2)]
    ys = [(rows[i], rows[i+1]) for i in range(0, len(rows)-1, 2)]
    return xs, ys


def calibrate(a, box, yticks):
    x0, x1, y0, y1 = box
    sub = a[y0:y1+1, x0:x1+1]
    g = (np.abs(sub.astype(int) - GRID).max(2) < 12)
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    gx = _runs(np.nonzero(g.sum(0) > 0.55*h)[0])
    gy = _runs(np.nonzero(g.sum(1) > 0.55*w)[0])
    return gx, gy


def fit_map(pix, vals):
    """Least-squares affine pixel -> data."""
    pix = np.asarray(pix, float); vals = np.asarray(vals, float)
    A = np.vstack([pix, np.ones_like(pix)]).T
    m, c = np.linalg.lstsq(A, vals, rcond=None)[0]
    return m, c


def extract(path=PNG, verbose=True):
    a = np.asarray(Image.open(path).convert('RGB'))
    xs, ys = find_panels(a)
    if verbose:
        print(f'image {a.shape[1]}x{a.shape[0]}  panel cols {xs}  rows {ys}')
    out = {}
    for name, r, c, yticks in PANELS:
        x0, x1 = xs[c]; y0, y1 = ys[r]
        gx, gy = calibrate(a, (x0, x1, y0, y1), yticks)
        if verbose:
            print(f'  panel {name}: box x[{x0},{x1}] y[{y0},{y1}]  '
                  f'{len(gx)} x-gridlines, {len(gy)} y-gridlines '
                  f'(expected {len(XTICKS)}, {len(yticks)})')
        if len(gx) != len(XTICKS) or len(gy) != len(yticks):
            raise RuntimeError(f'panel {name}: gridline count mismatch')
        mx, cx = fit_map([x0 + v for v in gx], XTICKS)
        # image rows increase downward, ticks listed bottom-up
        my, cy = fit_map([y0 + v for v in gy], list(reversed(yticks)))

        sub = a[y0:y1+1, x0:x1+1].astype(int)
        mask = np.abs(sub - C0).sum(2) < 90
        tt, yy = [], []
        for j in range(mask.shape[1]):
            rows_ = np.nonzero(mask[:, j])[0]
            if len(rows_) == 0:
                continue
            tt.append(mx*(x0 + j) + cx)
            yy.append(my*(y0 + rows_.mean()) + cy)
        out[name] = (np.asarray(tt), np.asarray(yy))
    return out


if __name__ == '__main__':
    cur = extract()
    print('\ncalibration check against independently known values')
    known = {'A': ('n_A(0)', 0.028), 'B': ('n_B(0)', 0.170), 'C': ('n_C(0)', 0.0),
             'D': ('n_D(0)', 0.0), 'E': ('n_E(0)', 0.0), 'V': ('V(0)', 2.5e-5)}
    for k, (lab, exp) in known.items():
        t, y = cur[k]
        v = y[np.argmin(np.abs(t - 0.0))]
        print(f'  {lab:8s} extracted {v: .5g}   expected {exp: .5g}')
    print()
    for k, (t, y) in cur.items():
        print(f'  n{k}: {len(t)} points, t {t.min():.1f}-{t.max():.1f}, '
              f'y {y.min():.4g}..{y.max():.4g}')
    np.savez('urethane_paper_curves.npz', **{k: np.vstack(v) for k, v in cur.items()})
    print('\nSaved urethane_paper_curves.npz')
