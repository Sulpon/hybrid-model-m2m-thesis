"""
TASK 3/4 verification: does the ACTUAL sdeint.itoint numerical implementation
in KD_fit.py, run with sigma_A=0.006, produce the variance-vs-time relationship
predicted analytically (Var[A(t)] = sigma_A^2 * t), and what sigma_A value
(estimated three ways from the author's real residual curve) actually
reproduces the author's data when run through the real simulator (not just
the closed-form formula)? Read-only, no files modified.
"""
import numpy as np
import sdeint

t_grid = np.array([0, 15, 30, 45, 60, 75, 90], dtype=float)   # seconds, matches the 7 shared timepoints
author_var = np.array([0, 0.86, 1.44, 2.45, 3.45, 4.58, 5.58])

# ---------------------------------------------------------------------------
# TASK 3: estimate sigma_A three ways from the author's observed variance curve
# ---------------------------------------------------------------------------
# (1) slope through the origin
a_origin = np.sum(t_grid * author_var) / np.sum(t_grid ** 2)
sigma_origin = np.sqrt(a_origin)

# (2) ordinary least squares (free intercept)
A = np.vstack([t_grid, np.ones_like(t_grid)]).T
slope_ols, intercept_ols = np.linalg.lstsq(A, author_var, rcond=None)[0]
sigma_ols = np.sqrt(max(slope_ols, 0))

# (3) endpoint estimate
a_endpoint = author_var[-1] / t_grid[-1]
sigma_endpoint = np.sqrt(a_endpoint)

print("===== TASK 3: sigma_A estimates from author's observed variance curve =====")
print(f"(1) through-origin slope   : a={a_origin:.5f}  sigma_A={sigma_origin:.4f}")
print(f"(2) OLS (free intercept)   : slope={slope_ols:.5f}  intercept={intercept_ols:.4f}  sigma_A={sigma_ols:.4f}")
print(f"(3) endpoint (t7 only)     : a={a_endpoint:.5f}  sigma_A={sigma_endpoint:.4f}")
print(f"mean of the three estimates: {np.mean([sigma_origin, sigma_ols, sigma_endpoint]):.4f}")

# ---------------------------------------------------------------------------
# TASK 4: simulate dA = 0*dt + sigma_A*dW using the EXACT same sdeint.itoint
# call pattern as KD_fit.py's stage2_drift/stage2_noise (index 0 = species A),
# with N replicates, and read off the empirical variance curve at t_grid.
# ---------------------------------------------------------------------------
def simulate_A_only(sigma_A, N=2000, seed=0):
    rng = np.random.default_rng(seed)
    t_fine = np.linspace(0, 90, 1000)     # matches KD_fit.py's np.linspace(0, t2_seconds, 1000) resolution
    out = np.zeros((N, len(t_grid)))
    for i in range(N):
        np.random.seed(rng.integers(0, 2**31 - 1))   # sdeint uses the global np.random state
        sol = sdeint.itoint(lambda y, t: np.array([0.0]),
                             lambda y, t: np.array([[sigma_A]]),
                             np.array([0.0]), t_fine)
        out[i] = np.interp(t_grid, t_fine, sol[:, 0])
    return out.var(axis=0, ddof=1)

print("\n===== TASK 4: empirical variance curve from the ACTUAL sdeint.itoint implementation =====")
var_current = simulate_A_only(0.006, N=2000, seed=1)
print(f"sigma_A = 0.006 (current KD_fit.py value):")
print(f"  simulated Var(A) at t=0..90s: {np.round(var_current, 6)}")
print(f"  analytic prediction sigma^2*t: {np.round(0.006**2 * t_grid, 6)}")

sigma_candidate = float(np.mean([sigma_origin, sigma_ols, sigma_endpoint]))
var_corrected = simulate_A_only(sigma_candidate, N=2000, seed=2)
print(f"\nsigma_A = {sigma_candidate:.4f} (estimated from author data):")
print(f"  simulated Var(A) at t=0..90s: {np.round(var_corrected, 3)}")
print(f"  analytic prediction sigma^2*t: {np.round(sigma_candidate**2 * t_grid, 3)}")
print(f"  author's actual observed curve: {author_var}")

# direct solve: what sigma_A, plugged into the REAL simulator, minimizes squared
# error against the author's curve (grid search, cross-check of the closed form)
print("\n===== Grid search over sigma_A using the real simulator (cross-check) =====")
best_sigma, best_err = None, np.inf
for s in np.arange(0.15, 0.35, 0.01):
    v = simulate_A_only(s, N=500, seed=3)
    err = np.sum((v - author_var) ** 2)
    if err < best_err:
        best_err, best_sigma = err, s
print(f"best-fit sigma_A over grid search: {best_sigma:.3f}  (sse={best_err:.3f})")
