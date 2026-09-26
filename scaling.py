import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


X1_train = pd.read_excel("X1.xlsx").values
X2_train = pd.read_excel("X2.xlsx").values
M2_train = pd.read_excel("M2.xlsx").values
Y_train = pd.read_excel("Y.xlsx").values

# =========================
# AUTOSCALE (fit ONLY on training)
# =========================
def scaling(X):
    X = X.copy()
    for i in range(X.shape[1]):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        X[:,i] = X[:,i] - mean
        X[:,i] = X[:,i]/std
    return X
X1_train_scaled = scaling(X1_train)
X2_train_scaled = scaling(X2_train)
M2_train_scaled = scaling(M2_train)
Y_train_scaled = scaling(Y_train)
print(Y_train_scaled.shape)
Y_train_scaled = Y_train_scaled.reshape(-1, 1)
print(Y_train_scaled.shape)

# =========================
# BLOCK NORMALIZATION (MBPLS!)
# =========================
def normalize_blocks(X1, M2, X2):
    X1n = X1 / np.sqrt(X1.shape[1])
    M2n = M2 / np.sqrt(M2.shape[1])
    X2n = X2 / np.sqrt(X2.shape[1])
    return X1n, M2n, X2n

X1n, M2n ,X2n = normalize_blocks(X1_train_scaled, M2_train_scaled, X2_train_scaled)
X1_only = X1n
X1_M2 = np.hstack([X1n, M2n])
X1_M2_X2 = np.hstack([X1n, M2n, X2n])
X_blocks = [X1n, M2n, X2n]

# =========================
# CROSS-VALIDATION (RMSECV)
# =========================
max_lv = 20
rmsecv = []
cv = KFold(n_splits=10,shuffle=True,random_state=42)

for lv in range (1,max_lv+1):
    pls = PLSRegression(n_components=lv)
    Y_cv = cross_val_predict(pls,X1_M2_X2, Y_train_scaled, cv=cv)
    rmse = np.sqrt(mean_squared_error(Y_train_scaled, Y_cv))
    rmsecv.append(rmse)
    print(f"RMSECV_predict: {rmse}")

# =========================
# BEST LV (MIN RMSECV)
# =========================
best = np.min(rmsecv)
threshold = best * 1.05   # 5% rule
chosen_lv = np.where(rmsecv <= threshold)[0][0] + 1
lvs = np.arange(1, max_lv + 1)

def mb_pls(X_blocks, Y, LV):
    S = len(X_blocks)
    # Initial PLS to get Y scores (super-score)
    pls_init = PLSRegression(n_components=LV)
    pls_init.fit(np.hstack(X_blocks), Y)
    U = pls_init.y_scores_  # Y latent variables

    # Initialize storage
    block_weights_all = []
    block_scores_all = []
    super_scores_all = []
    super_weights_all = []

    X_blocks_current = [X.copy() for X in X_blocks]
    Y_current = Y.copy()
    q_all = []
    for lv in range(LV):
        block_weights = []
        block_scores = []

        # Current Y score for latent variable r
        pls_tmp = PLSRegression(n_components=1)
        pls_tmp.fit(np.hstack(X_blocks_current), Y_current)
        u = pls_tmp.y_scores_

        # Step 1: Compute block weights, scores, and loadings
        for s in range (S):
            Xs = X_blocks_current[s]
            ws = (Xs.T @ u) / (u.T @ u)
            ws /= np.linalg.norm(ws)
            block_weights.append(ws)

            # Block score
            ts = Xs @ ws
            block_scores.append(ts)

        # Step 2: Super-score
        T_super = np.hstack(block_scores)
        w_super = (T_super.T @ u) / (u.T @ u)
        t = T_super @ w_super

        # Store results
        block_weights_all.append(block_weights)
        block_scores_all.append(block_scores)
        super_scores_all.append(t)
        super_weights_all.append(w_super)

        # Step 3: Deflate X blocks and Y
        for s in range(S):
            ps = X_blocks_current[s].T @ t / (t.T @ t)
            X_blocks_current[s] = X_blocks_current[s] - t @ ps.T
        q = (Y_current.T @ t) / (t.T @ t)
        q = np.array(q).reshape(-1, 1)
        q_all.append(q)
        Y_current = Y_current - t @ q.T

    return {
        "block_weights": block_weights_all,
        "block_scores": block_scores_all,
        "super_scores": super_scores_all,
        "super_weights": super_weights_all,
        "q_loadings": q_all
    }

results = mb_pls(X_blocks, Y_train_scaled, LV=chosen_lv)
# =========================================================
# RECONSTRUCT Y PREDICTIONS
# =========================================================

T = np.hstack(results["super_scores"])

Q = np.hstack(results["q_loadings"])
print(Q.shape)
# Predicted scaled Y
Y_pred = T @ Q.T

# =========================================================
# PERFORMANCE METRICS
# =========================================================

rmse = np.sqrt(mean_squared_error(Y_train_scaled, Y_pred))

r2 = r2_score(Y_train_scaled, Y_pred)

print("\n=== MY MB-PLS PERFORMANCE ===")
print(f"RMSE: {rmse:.4f}")
print(f"R2   : {r2:.4f}")

plt.figure(figsize=(4, 3))
plt.plot(lvs,rmsecv,marker="o",color="black",linewidth=1.5)
# highlight best LV
plt.scatter(chosen_lv,rmsecv[chosen_lv - 1],color="red",s=60,label=f"Best LV = {chosen_lv}")
plt.xlabel("Latent Variables (LV)", fontsize = 20)
plt.ylabel("RMSECV", fontsize = 20)
plt.xticks(lvs)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

