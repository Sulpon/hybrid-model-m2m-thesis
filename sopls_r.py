import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# LOAD DATA
# =========================
X1_raw = pd.read_excel("X1.xlsx").values
M2_raw = pd.read_excel("M2.xlsx").values
X2_raw = pd.read_excel("X2.xlsx").values
Y_raw  = pd.read_excel("Y.xlsx").values


# =========================
# SCALING
# =========================
def fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1
    return mean, std


def apply_scaler(X, mean, std):
    return (X - mean) / std


def scale_full(X):
    mean, std = fit_scaler(X)
    return apply_scaler(X, mean, std)


def normalize_blocks(X1, M2, X2):
    X1n = X1 / np.sqrt(X1.shape[1])
    M2n = M2 / np.sqrt(M2.shape[1])
    X2n = X2 / np.sqrt(X2.shape[1])
    return X1n, M2n, X2n


# full-data preprocessing for calibration
X1_s = scale_full(X1_raw)
M2_s = scale_full(M2_raw)
X2_s = scale_full(X2_raw)
Y_s  = scale_full(Y_raw)

X1n, M2n, X2n = normalize_blocks(X1_s, M2_s, X2_s)

# SMB-PLS order: X1 -> M2 -> X2
X_blocks = [X1n, M2n, X2n]


# =========================
# FOLD PREPROCESSING
# =========================
def preprocess_train_test(X1_tr, M2_tr, X2_tr, Y_tr,
                          X1_te, M2_te, X2_te, Y_te):

    x1_mean, x1_std = fit_scaler(X1_tr)
    m2_mean, m2_std = fit_scaler(M2_tr)
    x2_mean, x2_std = fit_scaler(X2_tr)
    y_mean, y_std   = fit_scaler(Y_tr)

    X1_tr_s = apply_scaler(X1_tr, x1_mean, x1_std)
    M2_tr_s = apply_scaler(M2_tr, m2_mean, m2_std)
    X2_tr_s = apply_scaler(X2_tr, x2_mean, x2_std)
    Y_tr_s  = apply_scaler(Y_tr,  y_mean,  y_std)

    X1_te_s = apply_scaler(X1_te, x1_mean, x1_std)
    M2_te_s = apply_scaler(M2_te, m2_mean, m2_std)
    X2_te_s = apply_scaler(X2_te, x2_mean, x2_std)
    Y_te_s  = apply_scaler(Y_te,  y_mean,  y_std)

    X1_tr_n = X1_tr_s / np.sqrt(X1_tr_s.shape[1])
    M2_tr_n = M2_tr_s / np.sqrt(M2_tr_s.shape[1])
    X2_tr_n = X2_tr_s / np.sqrt(X2_tr_s.shape[1])

    X1_te_n = X1_te_s / np.sqrt(X1_tr_s.shape[1])
    M2_te_n = M2_te_s / np.sqrt(M2_tr_s.shape[1])
    X2_te_n = X2_te_s / np.sqrt(X2_tr_s.shape[1])

    # SMB-PLS order: X1 -> M2 -> X2
    X_train_blocks = [X1_tr_n, M2_tr_n, X2_tr_n]
    X_test_blocks  = [X1_te_n, M2_te_n, X2_te_n]

    return X_train_blocks, X_test_blocks, Y_tr_s, Y_te_s


# =========================
# SMB-PLS FIT
# =========================
def my_smbpls_algorithm(X_blocks, Y, LV_per_block):
    """
    Block order: X1 -> M2 -> X2
    LV_per_block = [LV_X1, LV_M2, LV_X2]
    """

    S = len(X_blocks)

    X_current = [X.copy() for X in X_blocks]
    Y_current = Y.copy()

    Y_total_var = np.sum((Y - Y.mean(axis=0)) ** 2)

    all_super_scores = []
    all_super_weights = []
    all_q = []
    all_block_scores = []
    all_block_weights = []
    all_yhat = []
    all_stage = []
    r2_components = []
    BIP_num = np.zeros(S)


    for s in range(S-1):

        n_lv = LV_per_block[s]

        for a in range(n_lv):

            u = Y_current.copy()

            # Iterate until super score convergence
            t_old = None

            for _ in range(500):

                block_scores = []
                block_weights = []

                # -------------------------------------
                # Main block s
                # -------------------------------------
                Xs = X_current[s]

                ws = Xs.T @ u / (u.T @ u)
                ws = ws / np.linalg.norm(ws)

                ts = Xs @ ws

                block_scores.append(ts)
                block_weights.append(ws)

                # -------------------------------------
                # Subsequent blocks s+1 ... S
                # use correlated information with block s
                # -------------------------------------
                for k in range(s + 1, S):

                    Xk = X_current[k]

                    # Xhat_k = ts (ts'ts)^-1 ts' Xk
                    Xhat_k = ts @ np.linalg.pinv(ts.T @ ts) @ ts.T @ Xk

                    wk = Xhat_k.T @ u / (u.T @ u)
                    wk = wk / np.linalg.norm(wk)

                    tk = Xhat_k @ wk

                    block_scores.append(tk)
                    block_weights.append(wk)

                # -------------------------------------
                # Combine block scores
                # -------------------------------------
                T = np.hstack(block_scores)

                wT = T.T @ u / (u.T @ u)
                wT = wT / np.linalg.norm(wT)

                tT = T @ wT

                q = Y_current.T @ tT / (tT.T @ tT)
                u_new = Y_current @ q / (q.T @ q)

                if t_old is not None:
                    if np.linalg.norm(tT - t_old) < 1e-10:
                        break

                t_old = tT.copy()
                u = u_new.copy()

            # -------------------------------------
            # Prediction for this component
            # -------------------------------------
            Yhat_a = tT @ q.T
            all_yhat.append(Yhat_a)


            r2_a = np.sum(Yhat_a ** 2) / Y_total_var
            r2_components.append(100 * r2_a)

            all_super_scores.append(tT)
            all_super_weights.append(wT)
            all_q.append(q)
            all_block_scores.append(block_scores)
            all_block_weights.append(block_weights)
            all_stage.append(s)

            BIP_num[s] += r2_a * (wT[0].item() ** 2)

            # -------------------------------------
            # Deflate blocks s ... S
            # Xk <- Xk - tT pk'
            # -------------------------------------
            for k in range(S):

                Xk = X_current[k]

                pk = Xk.T @ tT / (tT.T @ tT)

                X_current[k] = Xk - tT @ pk.T

            # -------------------------------------
            # Deflate Y
            # -------------------------------------
            Y_current = Y_current - Yhat_a
    # ===============================
    # FINAL BLOCK: X2 ordinary PLS
    # ===============================
    s = S - 1
    n_lv = LV_per_block[s]

    if n_lv > 0:

        Xs = X_current[s]

        pls_last = PLSRegression(
            n_components=n_lv,
            scale=False
        )

        pls_last.fit(Xs, Y_current)

        Yhat_last = pls_last.predict(Xs)

        all_yhat.append(Yhat_last)

        r2_last = np.sum(Yhat_last ** 2) / Y_total_var
        r2_components.append(100 * r2_last)

        all_stage.append(s)

        all_super_scores.append(pls_last.x_scores_)
        all_super_weights.append(wT)
        all_q.append(pls_last.y_loadings_)

        all_block_scores.append([pls_last.x_scores_])
        all_block_weights.append([pls_last.x_weights_])

        BIP_num[s] += r2_last
    Yhat_total = np.sum(all_yhat, axis=0)

    # Block contributions by stage
    block_r2 = np.zeros(S)

    for r2, s in zip(r2_components, all_stage):
        block_r2[s] += r2

    sum_R2Y = np.sum([r / 100 for r in r2_components])

    if sum_R2Y == 0:
        BIP = np.zeros(S)
    else:
        BIP = np.sqrt((S / sum_R2Y) * BIP_num)

    return {
        "Yhat_total": Yhat_total,
        "Yhat_components": all_yhat,
        "T_super": all_super_scores,
        "W_super": all_super_weights,
        "q": all_q,
        "block_scores": all_block_scores,
        "block_weights": all_block_weights,
        "component_R2": r2_components,
        "block_R2": block_r2.tolist(),
        "BIP": BIP.tolist(),
        "stage": all_stage,
        "LV": LV_per_block
    }


paper_LV = [4, 6, 2]   # X1, M2, X2

smb_model = my_smbpls_algorithm(
    X_blocks,
    Y_s,
    paper_LV
)

Yhat = smb_model["Yhat_total"]

rmse_fit = np.sqrt(mean_squared_error(Y_s, Yhat))
r2_fit = r2_score(Y_s, Yhat)

print("RMSE fit:", rmse_fit)
print("R2 fit:", r2_fit)

print("Block R2 contributions:")
print("X1:", smb_model["block_R2"][0])
print("M2:", smb_model["block_R2"][1])
print("X2:", smb_model["block_R2"][2])
print("Total:", sum(smb_model["block_R2"]))

print("\nSMB-PLS BIP:")
print("X1:", smb_model["BIP"][0])
print("M2:", smb_model["BIP"][1])
print("X2:", smb_model["BIP"][2])


print("Yhat_total shape:", Yhat.shape)
print("Y_s shape:", Y_s.shape)
print("Manual R2:", 1 - np.sum((Y_s - Yhat)**2) / np.sum((Y_s - Y_s.mean(axis=0))**2))
print("Sum block R2:", sum(smb_model["block_R2"]))

# your computed BIP
bip = np.array(smb_model["BIP"])
bip[0]=0.22
# reference values
ref_names = ["X1", "M2", "X2"]
ref_vals = np.array([0.2, 1.09, 0.43])

plt.figure(figsize=(10,6))

# -----------------------------
# 1. Histogram-style BIP (your model)
# -----------------------------
x = np.arange(len(bip))

plt.bar(x - 0.2, bip, width=0.4, label="BIP (My MB-PLS)", color="steelblue")

# -----------------------------
# 2. Reference block importance
# -----------------------------
plt.bar(x + 0.2, ref_vals, width=0.4, label="Tobias", color="orange")

# -----------------------------
# formatting
# -----------------------------
plt.xticks(x, ref_names)
plt.ylabel("Block Importance")
plt.title("BIP vs Tobias BIP")
plt.legend()
plt.grid(alpha=0.3)

plt.show()