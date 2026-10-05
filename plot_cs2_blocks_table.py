"""
Case study 2 data-blocks table, styled to match the case study 1 table.

Dimensions are computed, not typed: the grid, the horizon and the batch count
come from case_study_2, so the table cannot drift from the analysis.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import case_study_2 as U

FILL = '#FCEFEC'
RULE = '#C9A9A2'
TEXT = '#1A1A1A'
S = 70.0                      # the shared M/X grid end used for the headline result


def build_rows():
    TG = U.TGRID
    keep = TG <= S
    n, nt, N = int(keep.sum()), len(TG), U.N_BATCH
    nM, nX = 3*n + 3*nt, 3*n
    rows = [
        ('Block', 'Size', 'Meaning'),
        ('M', f'{N} × {nM}',
         f'The mechanistic block: KD-model trajectories (3 species × {n}\n'
         f'timepoints, 0–{S:.0f} h) plus the input programme f$_{{v1}}$, '
         f'f$_{{v2}}$, T (3 × {nt})'),
        ('X', f'{N} × {nX}',
         f'The measured block: sensor readings for C, D, E\n'
         f'(3 species × {n} timepoints, 0–{S:.0f} h)'),
        ('Y', f'{N} × 1',
         f'Final n$_D$, measured at {TG[-1]:.0f} h — outside the block grid'),
    ]
    return rows, dict(n=n, nt=nt, N=N, nM=nM, nX=nX)


def main():
    rows, info = build_rows()
    heights = [0.58] + [1.02 if '\n' in r[2] else 0.62 for r in rows[1:]]
    total = sum(heights)
    W = 13.6
    xs = [0.0, 1.35, 3.30, W]          # column boundaries

    fig, ax = plt.subplots(figsize=(W, total + 0.5))
    ax.set_xlim(0, W); ax.set_ylim(0, total)
    ax.axis('off')

    y = total
    for i, (blk, size, mean) in enumerate(rows):
        h = heights[i]
        ax.add_patch(Rectangle((0, y-h), W, h, facecolor=FILL, edgecolor='none'))
        if i == 0:
            ax.plot([0, W], [y, y], color=RULE, lw=1.6)
        ax.plot([0, W], [y-h, y-h], color=RULE, lw=0.9)
        fs = 17 if i == 0 else 16.5
        wt = 'normal'
        for j, txt in enumerate((blk, size, mean)):
            ax.text(xs[j] + 0.22, y - h/2, txt, ha='left', va='center',
                    fontsize=fs, color=TEXT, fontweight=wt, linespacing=1.45)
        y -= h

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    fig.savefig('deck_cs2_blocks_table.png', dpi=200,
                facecolor='white', bbox_inches='tight', pad_inches=0.06)
    print('block dimensions (computed):')
    print(f'  shared M/X grid : 0-{S:.0f} h  ->  {info["n"]} of {info["nt"]} timepoints')
    print(f'  M : {info["N"]} x {info["nM"]}   = 3x{info["n"]} states + 3x{info["nt"]} inputs')
    print(f'  X : {info["N"]} x {info["nX"]}   = 3x{info["n"]} states')
    print(f'  Y : {info["N"]} x 1')
    print('\nSaved deck_cs2_blocks_table.png')


if __name__ == '__main__':
    main()
