import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from mbpls.mbpls import MBPLS
import pyphi.calc as phi
import pyphi.plots as pp


X1_train = pd.read_excel("X1_mat.xlsx").values
X2_train = pd.read_excel("X2_matrix.xlsx").values
M2_train = pd.read_excel("M2_matrix.xlsx").values
Y_train = pd.read_excel("Y_mat.xlsx").values

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
q2cv = []
r2cv = []
cv = KFold(n_splits=10,shuffle=True,random_state=42)
Y_total_var = np.sum((Y_train_scaled - Y_train_scaled.mean(axis=0)) ** 2)
for lv in range (1,max_lv+1):
    pls = PLSRegression(n_components=lv)
    Y_cv = cross_val_predict(pls,X1_M2_X2, Y_train_scaled, cv=cv)
    rmse = np.sqrt(mean_squared_error(Y_train_scaled, Y_cv))
    press = np.sum((Y_train_scaled - Y_cv) ** 2)

    q2 = 1 - press / Y_total_var

    r2 = r2_score(Y_train_scaled,Y_cv)

    rmsecv.append(rmse)
    q2cv.append(q2)
    r2cv.append(r2)

    print(
        f"LV={lv:<2} "
        f"RMSECV={rmse:.4f} "
        f"Q2={q2:.4f} "
        f"R2={r2:.4f}"
    )

# =========================
# BEST LV (MIN RMSECV)
# =========================
best = np.min(rmsecv)
threshold = best * 1.05   # 5% rule
chosen_lv = np.where(rmsecv <= threshold)[0][0] + 1
lvs = np.arange(1, max_lv + 1)

best_q2 = q2cv[chosen_lv - 1]


# =========================================================
# YOUR MB-PLS IMPLEMENTATION
# =========================================================
def my_mbpls(X_blocks, Y, LV):
    S = len(X_blocks)

    # Initialize storage
    block_weights_all = []
    block_scores_all = []
    super_scores_all = []
    super_weights_all = []
    q_all = []
    Y_predict = []
    R2 = []
    eig_all = []

    X_blocks_current = [X.copy() for X in X_blocks]
    Y_current = Y.copy()
    Y_mean = Y.mean(axis = 0)
    Y_total_var = np.sum((Y_current - Y_mean) ** 2)

    TSS_blocks = [np.sum(Xb ** 2) for Xb in X_blocks]
    TSS_X_total = np.sum(TSS_blocks)
    R2X_all = []
    R2Xb_all = []

    BIP_num = np.zeros(S)  # numerator accumulator per block

    for lv in range(LV):
        block_weights = []
        block_scores = []
        u = Y_current
        r2xb_lv = []

        # Step 1: Compute block weights
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
        w_super /= np.linalg.norm(w_super)
        t = (T_super @ w_super).reshape(-1, 1)
        eig = (t.T @ t).item() / (t.shape[0] - 1)
        eig_all.append(eig)
        q = (Y_current.T @ t) / (t.T @ t)
        q = np.array(q).reshape(-1, 1)

        # Store results
        block_weights_all.append(block_weights)
        block_scores_all.append(block_scores)
        super_scores_all.append(t)
        super_weights_all.append(w_super)
        q_all.append(q)

        # Step 3: Deflate X blocks and Y
        # Y prediction
        Ypred = t @ q.T
        Y_predict.append(Ypred)
        # Deflate Y
        Y_current = Y_current - Ypred
        # R²
        R = 1 - np.sum((Y - Ypred) ** 2) / Y_total_var
        R2.append(R)
        for s in range(S):
            pb = X_blocks_current[s].T @ t / (t.T @ t)
            Xb_reconstructed = t @ pb.T
            r2xb = np.sum(Xb_reconstructed ** 2) / TSS_blocks[s]
            r2xb_lv.append(r2xb)
            ps = X_blocks_current[s].T @ t / (t.T @ t)
            X_blocks_current[s] = X_blocks_current[s] - t @ ps.T
            BIP_num[s] += R * (w_super[s].item() ** 2)
        # full R2X weighted
        R2X = np.sum([r2xb_lv[s] * TSS_blocks[s] for s in range(S)]) / TSS_X_total
        R2X_all.append(R2X)  # single value per LV
        R2Xb_all.append(r2xb_lv)  # per block values per LV

    sum_R2Y = np.sum(R2)
    BIP = np.sqrt((S / sum_R2Y) * BIP_num)
    return {
        "block_weights": block_weights_all,
        "block_scores": block_scores_all,
        "super_scores": super_scores_all,
        "super_weights": super_weights_all,
        "q_loadings": q_all,
        "Y_predict": Y_predict,
        "R2": R2,
        "Eig": eig_all,
        "R2X": R2X_all,
        "R2Xb": R2Xb_all,
        "BIP": BIP.tolist()
    }


# =========================================================
# FIT YOUR MODEL
# =========================================================
#LV = chosen_lv
LV=20

my_model = my_mbpls(X_blocks, Y_train_scaled, LV)
r2 = my_model["R2"]
# cumulative R2Y
cum_r2 = []
running = 0

for r in r2:
    running += r
    cum_r2.append(running)
results_table = pd.DataFrame({
    "LV": np.arange(1, max_lv + 1),
    "RMSECV": rmsecv,
    "Q2": q2cv
})

results_table["R2Y"] = cum_r2

results_table["RMSECV"] = results_table["RMSECV"].round(3)
results_table["Q2"] = (100*results_table["Q2"]).round(2)
results_table["R2Y"] = (100 * results_table["R2Y"]).round(2)

print("\n===== MB-PLS MODEL SELECTION =====")
print(results_table.to_string(index=False))
print("This is my r2: ", np.sum(r2))
#print("This is EIG: ", my_model["Eig"])
print("This is my r2x: ", np.sum(my_model["R2X"]))
block_importance_my = [w**2 for w in my_model['super_weights']]
bip = my_model['BIP']
print("BIP per block:")
for i, b in enumerate(bip):
    print(f"  Block {i+1}: {b:.4f}")
print(np.where(rmsecv <= best)[0][0] + 1)
print(chosen_lv)
# =========================================================
# FIT mbpls Pyphi PACKAGE
# =========================================================
n = X1_train.shape[0]
obs_ids = [f"obs_{i}" for i in range(n)]

X1_df = pd.DataFrame(X1_train, columns=[f"X1_{i}" for i in range(X1_train.shape[1])])
X2_df = pd.DataFrame(X2_train, columns=[f"X2_{i}" for i in range(X2_train.shape[1])])
M2_df = pd.DataFrame(M2_train, columns=[f"M2_{i}" for i in range(M2_train.shape[1])])
Y_df  = pd.DataFrame(Y_train,  columns=[f"Y_{i}"  for i in range(Y_train.shape[1])])

# Insert observation ID as first column (required by pyphi)
X1_df.insert(0, "ObsID", obs_ids)
X2_df.insert(0, "ObsID", obs_ids)
M2_df.insert(0, "ObsID", obs_ids)
Y_df.insert(0,  "ObsID", obs_ids)

Xdata = {'X1': X1_df, 'M2': M2_df, 'X2': X2_df}

mbpls_obj = phi.mbpls(Xdata, Y_df, LV)


X_concat_raw = np.hstack([X1_train, M2_train, X2_train])
col_names = ([f"X1_{i}" for i in range(X1_train.shape[1])] +
             [f"M2_{i}" for i in range(M2_train.shape[1])] +
             [f"X2_{i}" for i in range(X2_train.shape[1])])

X_pd = pd.DataFrame(X_concat_raw, columns=col_names)
X_pd.insert(0, "ObsID", obs_ids)

pred = phi.pls_pred(X_pd, mbpls_obj)
Yhat_pyphi = pred['Yhat']


# =========================================================
# MODEL FITTING
# =========================================================
# 1. your model — already fitted
my_model = my_mbpls(X_blocks, Y_train_scaled, LV)

# 2. pls_concat
pls_concat = PLSRegression(n_components=LV)
pls_concat.fit(X1_M2_X2, Y_train_scaled)

# 3. DTU
dtu_model = MBPLS(n_components=LV, method='NIPALS')
dtu_model.fit(X_blocks, Y_train_scaled)

# =========================================================
# R2Y COMPARISON
# =========================================================
# your model
R2Y_my = my_model['R2']

# pls_concat — compute per LV
R2Y_concat = []
Y_total_var = np.sum((Y_train_scaled - Y_train_scaled.mean(axis=0))**2)
T_concat = pls_concat.x_scores_
Q_concat = pls_concat.y_loadings_
for a in range(LV):
    t = T_concat[:, [a]]
    q = Q_concat[:, [a]]
    Ypred = t @ q.T
    R2Y_concat.append(np.sum(Ypred**2) / Y_total_var)

# DTU
R2Y_dtu = list(dtu_model.explained_var_y_)

# pyphi
R2Y_pyphi = np.array(mbpls_obj['r2ypv']).ravel().tolist()

print("\n===== R2Y COMPARISON =====")
print(f"{'LV':<6} {'my_mbpls':<12} {'pls_concat':<12} {'DTU':<12} {'pyphi':<12}")
cumY_my = cumY_concat = cumY_dtu = cumY_pyphi = 0
for i in range(LV):
    cumY_my     += R2Y_my[i]
    cumY_concat += R2Y_concat[i]
    cumY_dtu    += R2Y_dtu[i]
    cumY_pyphi  += R2Y_pyphi[i]
    print(f"LV{i+1:<4} {R2Y_my[i]:<12.4f} {R2Y_concat[i]:<12.4f} {R2Y_dtu[i]:<12.4f} {R2Y_pyphi[i]:<12.4f}")
print(f"{'sum':<6} {cumY_my:<12.4f} {cumY_concat:<12.4f} {cumY_dtu:<12.4f} {cumY_pyphi:<12.4f}")

# =========================================================
# R2X COMPARISON
# =========================================================
# your model
R2X_my = my_model['R2X']

# pls_concat — compute per LV
X_total_var = np.sum(X1_M2_X2**2)
R2X_concat = []
X_hat = np.zeros_like(X1_M2_X2)
P_concat = pls_concat.x_loadings_
for a in range(LV):
    t = T_concat[:, [a]]
    p = P_concat[:, [a]]
    X_hat += t @ p.T

    R2X_concat.append(np.sum(X_hat ** 2) / np.sum(X1_M2_X2 ** 2))
# DTU
R2X_dtu = list(dtu_model.explained_var_x_)

# pyphi
R2X_pyphi = np.array(mbpls_obj['r2x']).tolist()

print("\n===== R2X COMPARISON =====")
print(f"{'LV':<6} {'my_mbpls':<12} {'pls_concat':<12} {'DTU':<12} {'pyphi':<12}")
cumX_my = cumX_concat = cumX_dtu = cumX_pyphi = 0
for i in range(LV):
    cumX_my     += R2X_my[i]
    cumX_concat += R2X_concat[i]
    cumX_dtu    += R2X_dtu[i]
    cumX_pyphi  += R2X_pyphi[i]
    print(f"LV{i+1:<4} {R2X_my[i]:<12.4f} {R2X_concat[i]:<12.4f} {R2X_dtu[i]:<12.4f} {R2X_pyphi[i]:<12.4f}")
print(f"{'sum':<6} {cumX_my:<12.4f} {cumX_concat:<12.4f} {cumX_dtu:<12.4f} {cumX_pyphi:<12.4f}")

# =========================================================
# PREDICTIONS vs PYPHI
# =========================================================
Yhat_my = np.sum(my_model['Y_predict'], axis=0)
y_mean  = mbpls_obj['my'][0]
y_std   = mbpls_obj['sy'][0]
Yhat_pyphi_scaled = (Yhat_pyphi - y_mean) / y_std

print("\n===== PREDICTIONS vs PYPHI =====")
print(f"{'Obs':<6} {'my_mbpls':<12} {'pyphi':<12} {'diff':<12}")
for i in range(min(10, len(Yhat_my))):
    diff = abs(Yhat_my[i].item() - Yhat_pyphi_scaled[i].item())
    print(f"{i+1:<6} {Yhat_my[i].item():<12.4f} {Yhat_pyphi_scaled[i].item():<12.4f} {diff:<12.6f}")
print(f"max diff: {np.max(np.abs(Yhat_my - Yhat_pyphi_scaled)):.6f}")
print(f"mean diff: {np.mean(np.abs(Yhat_my - Yhat_pyphi_scaled)):.6f}")

# =========================================================
# SUPER SCORES vs PYPHI
# =========================================================
print("\n===== SUPER SCORES vs PYPHI =====")
print(f"{'LV':<6} {'ratio constant':<18} {'ratio value':<12} {'sign flip':<10}")
for a in range(LV):
    t_my    = my_model['super_scores'][a].ravel()
    t_pyphi = np.array(mbpls_obj['T'])[:, a]
    ratio   = t_pyphi / t_my
    constant = np.allclose(ratio, ratio[0], atol=1e-4)
    sign_flip = ratio[0] < 0
    print(f"LV{a+1:<4} {str(constant):<18} {ratio[0]:<12.4f} {str(sign_flip):<10}")

lvs = np.arange(1, max_lv + 1)

plt.figure(figsize=(4, 3))

plt.plot(
    lvs,
    rmsecv,
    marker="o",
    color="black",
    linewidth=1.5
)

# highlight best LV
plt.scatter(
    chosen_lv,
    rmsecv[chosen_lv - 1],
    color="red",
    s=60,
    label=f"Best LV = {chosen_lv}"
)

plt.xlabel("Latent Variables (LV)", fontsize = 20)
plt.ylabel("RMSECV", fontsize = 20)
plt.xticks(lvs)
plt.legend(frameon=False)

plt.tight_layout()
plt.show()

# your computed BIP
bip = np.array(my_model["BIP"])

# reference values
ref_names = ["X1", "M2", "X2"]
ref_vals = np.array([0.69, 1.13, 1.12])

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