"""
Render the Section 3.1 (Data Preprocessing) equations as a readable image + PDF.

Uses matplotlib mathtext, so no LaTeX installation is required.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY = '#1F3864'
GREY = '#555555'

fig = plt.figure(figsize=(9.6, 13.2))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

T = lambda y, s, **k: ax.text(k.pop('x', 0.07), y, s, va='center',
                              fontsize=k.pop('fs', 13), color=k.pop('c', '#111111'), **k)

y = 0.965
T(y, 'Section 3.1   Data Preprocessing', fs=19, c=NAVY, fontweight='bold')
y -= 0.030
T(y, 'Equations, with every symbol defined', fs=12, c=GREY, style='italic')

# ------------------------------------------------------------------ notation
y -= 0.055
T(y, 'Notation', fs=14, c=NAVY, fontweight='bold')
rows = [
    (r'$n = 1,\dots ,N$', 'observation (batch)'),
    (r'$i = 1,\dots ,B$', 'predictor block'),
    (r'$j = 1,\dots ,K_i$', 'column within block $i$'),
    (r'$K_i$', 'number of columns in block $i$'),
    (r'$N_{tr}$', 'number of observations in the training fold'),
    (r'$x_{nj}^{(i)}$', 'raw entry: row $n$, column $j$, block $i$'),
    (r'$\tilde{x}_{nj}^{(i)}$', 'the same entry after preprocessing'),
]
for a, b in rows:
    y -= 0.027
    ax.text(0.10, y, a, fontsize=13.5, va='center')
    ax.text(0.30, y, b, fontsize=12, va='center', color='#333333')

# ------------------------------------------------- (3.1) scaling parameters
y -= 0.055
T(y, 'Scaling parameters, estimated on the training fold only', fs=13, c=NAVY,
  fontweight='bold')
y -= 0.060
ax.text(0.14, y,
        r'$\mu_j^{(i)}=\frac{1}{N_{tr}}\sum_{n\in\mathrm{train}}'
        r'x_{nj}^{(i)}$', fontsize=20, va='center')
y -= 0.070
ax.text(0.14, y,
        r'$\sigma_j^{(i)}=\sqrt{\frac{1}{N_{tr}-1}\sum_{n\in\mathrm{train}}'
        r'\left(x_{nj}^{(i)}-\mu_j^{(i)}\right)^{2}}$', fontsize=19, va='center')
ax.text(0.95, y + 0.035, '(3.1)', fontsize=13, va='center', ha='right', color=GREY)

# ------------------------------------------------------------- (3.2) guard
y -= 0.075
T(y, 'Guard on near-constant columns', fs=13, c=NAVY, fontweight='bold')
y -= 0.050
ax.text(0.12, y, r'$\sigma_j^{(i)} \leftarrow 1 \qquad \mathrm{if} \qquad '
                 r'\sigma_j^{(i)} < \varepsilon$', fontsize=20, va='center')
ax.text(0.95, y, '(3.2)', fontsize=13, va='center', ha='right', color=GREY)
y -= 0.032
ax.text(0.12, y, r'$\varepsilon$ is a numerical tolerance ($10^{-12}$ here). '
                 'Such a column is centred but not rescaled.',
        fontsize=11.5, va='center', color='#333333')

# ------------------------------------------- (3.3) autoscaling + block scaling
y -= 0.055
T(y, 'Autoscaling and block scaling', fs=13, c=NAVY, fontweight='bold')
y -= 0.072
ax.text(0.20, y, r'$\tilde{x}_{nj}^{(i)}=\frac{x_{nj}^{(i)}-\mu_j^{(i)}}'
                 r'{\sigma_j^{(i)}\,\sqrt{K_i}}$', fontsize=24, va='center')
ax.text(0.95, y, '(3.3)', fontsize=13, va='center', ha='right', color=GREY)
y -= 0.048
ax.text(0.12, y, 'The first factor equalises the variance of the columns; '
                 'the second equalises the weight of the blocks.',
        fontsize=11.5, va='center', color='#333333')
y -= 0.026
ax.text(0.12, y, r'Applied with the same $\mu_j^{(i)}$ and $\sigma_j^{(i)}$ to '
                 r'the training AND the held-out observations.',
        fontsize=11.5, va='center', color='#333333')

# ------------------------------------------------------------ (3.4) response
y -= 0.058
T(y, 'Response scaling and back-transformation', fs=13, c=NAVY, fontweight='bold')
y -= 0.070
ax.text(0.14, y, r'$\tilde{y}_{n}=\frac{y_{n}-\mu_{y}}{\sigma_{y}}$',
        fontsize=22, va='center')
ax.text(0.52, y, r'$\hat{y}_{n}=\sigma_{y}\,\hat{u}_{n}+\mu_{y}$',
        fontsize=22, va='center')
ax.text(0.95, y, '(3.4)', fontsize=13, va='center', ha='right', color=GREY)
y -= 0.048
ax.text(0.12, y, r'$\hat{u}_n$ is the model prediction of the scaled response; '
                 r'$\mu_y,\ \sigma_y$ come from the training fold.',
        fontsize=11.5, va='center', color='#333333')
y -= 0.026
ax.text(0.12, y, r'RMSECV and $Q^{2}$ are therefore reported in the units of the response.',
        fontsize=11.5, va='center', color='#333333')

# ---------------------------------------------------------------- footnote
y -= 0.062
ax.add_patch(plt.Rectangle((0.06, y-0.072), 0.89, 0.094,
                           facecolor='#F2F4F9', edgecolor='none'))
ax.text(0.085, y-0.002, 'What is, and is not, a scaling parameter',
        fontsize=12.5, va='center', color=NAVY, fontweight='bold')
ax.text(0.085, y-0.030,
        r'Estimated from data: $\mu_j^{(i)}$, $\sigma_j^{(i)}$, $\mu_y$, $\sigma_y$. '
        'These must come from the training fold only.',
        fontsize=11.5, va='center')
ax.text(0.085, y-0.055,
        r'Not estimated: $\sqrt{K_i}$ depends only on the width of the block, '
        'so it cannot leak and needs no fold restriction.',
        fontsize=11.5, va='center')

fig.savefig('eq_preprocessing.png', dpi=200, facecolor='white')
fig.savefig('eq_preprocessing.pdf', facecolor='white')
print('Saved eq_preprocessing.png and eq_preprocessing.pdf')
