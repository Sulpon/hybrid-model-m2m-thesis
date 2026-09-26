import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# LOAD DATA
# =========================
X1_train = pd.read_excel("X1.xlsx").values
X2_train = pd.read_excel("X2.xlsx").values
M2_train = pd.read_excel("M2.xlsx").values
Y_train  = pd.read_excel("Y.xlsx").values


# =========================
# AUTOSCALE
# =========================
def scaling(X):
    X = X.copy()
    for i in range(X.shape[1]):
        mean = np.mean(X[:, i])
        std = np.std(X[:, i], ddof=1)
        X[:, i] = (X[:, i] - mean) / std
    return X


X1_train_scaled = scaling(X1_train)
M2_train_scaled = scaling(M2_train)
X2_train_scaled = scaling(X2_train)
Y_train_scaled  = scaling(Y_train)


# =========================
# BLOCK NORMALIZATION
# =========================
def normalize_blocks(X1, M2, X2):
    X1n = X1 / np.sqrt(X1.shape[1])
    M2n = M2 / np.sqrt(M2.shape[1])
    X2n = X2 / np.sqrt(X2.shape[1])
    return X1n, M2n, X2n

X1n, M2n, X2n = normalize_blocks(X1_train_scaled, M2_train_scaled, X2_train_scaled)

# Paper order: X1 -> M2 -> X2
X_blocks = [X1n, M2n, X2n]

def fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1
    return mean, std

def apply_scaler(X, mean, std):
    return (X - mean) / std


def preprocess_train_test(X1_train, M2_train, X2_train, Y_train,
                          X1_test, M2_test, X2_test, Y_test):

    # fit scalers only on training fold
    x1_mean, x1_std = fit_scaler(X1_train)
    m2_mean, m2_std = fit_scaler(M2_train)
    x2_mean, x2_std = fit_scaler(X2_train)
    y_mean, y_std   = fit_scaler(Y_train)

    # apply to train
    X1_train_s = apply_scaler(X1_train, x1_mean, x1_std)
    M2_train_s = apply_scaler(M2_train, m2_mean, m2_std)
    X2_train_s = apply_scaler(X2_train, x2_mean, x2_std)
    Y_train_s  = apply_scaler(Y_train, y_mean, y_std)

    # apply same scalers to test
    X1_test_s = apply_scaler(X1_test, x1_mean, x1_std)
    M2_test_s = apply_scaler(M2_test, m2_mean, m2_std)
    X2_test_s = apply_scaler(X2_test, x2_mean, x2_std)
    Y_test_s  = apply_scaler(Y_test, y_mean, y_std)

    # block normalization
    X1_train_n = X1_train_s / np.sqrt(X1_train_s.shape[1])
    M2_train_n = M2_train_s / np.sqrt(M2_train_s.shape[1])
    X2_train_n = X2_train_s / np.sqrt(X2_train_s.shape[1])

    X1_test_n = X1_test_s / np.sqrt(X1_train_s.shape[1])
    M2_test_n = M2_test_s / np.sqrt(M2_train_s.shape[1])
    X2_test_n = X2_test_s / np.sqrt(X2_train_s.shape[1])

    X_train_blocks = [X1_train_n, M2_train_n, X2_train_n]
    X_test_blocks  = [X1_test_n,  M2_test_n,  X2_test_n]

    return X_train_blocks, X_test_blocks, Y_train_s, Y_test_s, y_mean, y_std

# =========================
# SO-PLS FIT
# =========================
def my_sopls(X_blocks, Y, LV):
    S = len(X_blocks)

    block_models = []
    block_scores = []
    block_loadings_x = []
    block_loadings_y = []
    H_matrices = []
    Yhat_blocks = []

    # stores how each fitted block orthogonalizes later blocks
    ortho_coef = []
    block_r2_residual = []

    X_blocks_current = [X.copy() for X in X_blocks]
    Y_current = Y.copy()

    Y_mean = Y.mean(axis=0)
    Y_total_var = np.sum((Y - Y_mean) ** 2)

    for s in range(S):

        if LV[s] == 0:
            block_scores.append(None)
            block_loadings_x.append(None)
            block_loadings_y.append(None)
            block_models.append(None)
            H_matrices.append(None)
            Yhat_blocks.append(np.zeros_like(Y))
            ortho_coef.append({})
            continue

        Xs = X_blocks_current[s]

        pls = PLSRegression(n_components=LV[s],scale=False)
        pls.fit(Xs, Y_current)

        Yhat = pls.predict(Xs)
        Ts = pls.transform(Xs)

        P = pls.x_loadings_
        Q = pls.y_loadings_

        H = Ts @ np.linalg.pinv(Ts.T @ Ts) @ Ts.T
        I = np.eye(Ts.shape[0])

        block_scores.append(Ts)
        block_loadings_x.append(P)
        block_loadings_y.append(Q)
        block_models.append(pls)
        H_matrices.append(H)
        Yhat_blocks.append(Yhat)

        # training projection coefficients:
        # X_future_hat = Ts @ G
        # X_future_orth = X_future - Ts @ G
        G_dict = {}

        Tpinv = np.linalg.pinv(Ts.T @ Ts) @ Ts.T

        for n in range(s + 1, S):
            G = Tpinv @ X_blocks_current[n]
            G_dict[n] = G
            X_blocks_current[n] = X_blocks_current[n] - Ts @ G
        ortho_coef.append(G_dict)
        # orthogonalize Y for the next block during training
        ss_before = np.sum(Y_current ** 2)
        GY = Tpinv @ Y_current
        Y_next = Y_current - Ts @ GY

        ss_after = np.sum(Y_next ** 2)

        block_r2_residual.append(100 * (ss_before - ss_after) / Y_total_var)

        Y_current = Y_next

    Yhat_total = np.sum(Yhat_blocks, axis=0)

    SS_blocks = [np.sum((Y-Yb) ** 2) for Yb in Yhat_blocks]
    contrib_pct = [100 * (1- ss / Y_total_var) for ss in SS_blocks]

    return {
        "models": block_models,
        "T": block_scores,
        "P": block_loadings_x,
        "Q": block_loadings_y,
        "H": H_matrices,
        "Yhat_blocks": Yhat_blocks,
        "Yhat_total": Yhat_total,
        "SS": contrib_pct,
        "Y_total_var": Y_total_var,
        "ortho_coef": ortho_coef,
        "R2_residual": block_r2_residual,
        "LV": LV
    }


# =========================
# SO-PLS PREDICTION
# =========================
def predict_sopls(model, X_blocks):

    X_current = [X.copy() for X in X_blocks]
    n = X_blocks[0].shape[0]

    Yhat_total = np.zeros((n, 1))

    for s, pls in enumerate(model["models"]):

        if pls is None:
            continue

        Xs = X_current[s]

        # predict residual Y contribution from current block
        Yhat_s = pls.predict(Xs)
        Yhat_total += Yhat_s

        # test scores in the training PLS score space
        Ts_test = pls.transform(Xs)

        # use TRAINING orthogonalization coefficients
        for j, G in model["ortho_coef"][s].items():
            X_current[j] = X_current[j] - Ts_test @ G

    return Yhat_total


# =========================
# RMSECV
# =========================
def sopls_rmsecv_raw(X1_raw, M2_raw, X2_raw, Y_raw, LV, return_predictions=False, return_block_contrib=False, random_state=42):

    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    Y_cv_scaled = np.zeros_like(Y_raw, dtype=float)
    Y_true_scaled = np.zeros_like(Y_raw, dtype=float)

    for train_idx, test_idx in cv.split(Y_raw):

        X_train_blocks, X_test_blocks, Y_train_s, Y_test_s, y_mean, y_std = preprocess_train_test(
            X1_raw[train_idx], M2_raw[train_idx], X2_raw[train_idx], Y_raw[train_idx],
            X1_raw[test_idx],  M2_raw[test_idx],  X2_raw[test_idx],  Y_raw[test_idx])

        model = my_sopls(X_train_blocks, Y_train_s, LV)

        Y_pred_s = predict_sopls(model, X_test_blocks)

        Y_cv_scaled[test_idx] = Y_pred_s
        Y_true_scaled[test_idx] = Y_test_s

    rmse = np.sqrt(mean_squared_error(Y_true_scaled, Y_cv_scaled))
    mae = np.mean(np.abs(Y_true_scaled - Y_cv_scaled))
    bias = np.mean(Y_cv_scaled - Y_true_scaled)

    press = np.sum((Y_true_scaled - Y_cv_scaled) ** 2)
    tss = np.sum((Y_true_scaled - Y_true_scaled.mean(axis=0)) ** 2)
    q2 = 1 - press / tss
    r2_cv = r2_score(Y_true_scaled, Y_cv_scaled)

    output = [rmse, q2, r2_cv, mae, bias]

    if return_block_contrib:
        lv_x1, lv_m2, lv_x2 = LV

        # ---------- CV / Q2 block contributions ----------
        _, q2_x1, r2cv_x1, _, _ = sopls_rmsecv_raw(
            X1_raw, M2_raw, X2_raw, Y_raw,
            [lv_x1, 0, 0],random_state=random_state)

        _, q2_x1m2, r2cv_x1m2, _, _ = sopls_rmsecv_raw(
            X1_raw, M2_raw, X2_raw, Y_raw,
            [lv_x1, lv_m2, 0],random_state=random_state)

        q2_block_contrib = [100 * q2_x1, 100 * (q2_x1m2 - q2_x1), 100 * (q2 - q2_x1m2)]

        r2cv_block_contrib = [100 * r2cv_x1, 100 * (r2cv_x1m2 - r2cv_x1), 100 * (r2_cv - r2cv_x1m2)]

        output.append(q2_block_contrib)
        output.append(r2cv_block_contrib)
    if return_predictions:
        output.extend([Y_cv_scaled, Y_true_scaled])

    return tuple(output)

def sopls_cv_contrib_bootstrap(X1_raw, M2_raw, X2_raw, Y_raw, LV, n_reps=50):

    contribs = []
    totals = []

    for seed in range(n_reps):

        rmse, q2, r2cv, mae, bias, q2_blocks, r2cv_blocks = \
            sopls_rmsecv_raw(X1_raw, M2_raw, X2_raw, Y_raw, LV, return_block_contrib=True, random_state=seed)

        contribs.append(q2_blocks)
        totals.append(100 * q2)

    contribs = np.array(contribs)

    mean_contrib = contribs.mean(axis=0)
    sd_contrib = contribs.std(axis=0, ddof=1)

    return {
        "mean": mean_contrib,
        "sd": sd_contrib,
        "total_mean": np.mean(totals),
        "total_sd": np.std(totals, ddof=1)}

def sopls_grid_search(X_blocks, Y_scaled,
                      X1_raw, M2_raw, X2_raw, Y_raw,
                      max_lv=6):

    results = []

    for lv_x1 in range(0, max_lv + 1):
        for lv_m2 in range(0, max_lv + 1):
            for lv_x2 in range(0, max_lv + 1):

                LV = [lv_x1, lv_m2, lv_x2]

                model = my_sopls(X_blocks, Y_scaled, LV)

                explained = sum(model["SS"])

                rmse, q2, r2cv, mae, bias = sopls_rmsecv_raw(X1_raw, M2_raw, X2_raw, Y_raw, LV)

                results.append([lv_x1, lv_m2, lv_x2, explained, rmse, q2, r2cv, mae, bias, model["SS"][0], model["SS"][1], model["SS"][2]])

    results_df = pd.DataFrame(results,
        columns=[
            "LV_X1",
            "LV_M2",
            "LV_X2",
            "Explained",
            "RMSECV",
            "Q2",
            "R2_CV",
            "MAE_CV",
            "Bias_CV",
            "Block_X1",
            "Block_M2",
            "Block_X2"])

    results_df["TotalLV"] = (results_df["LV_X1"] + results_df["LV_M2"] + results_df["LV_X2"])
    return results_df

def plot_sopls_mage(results_df, paper_LV, case_name):

    lv_x1_p, lv_m2_p, lv_x2_p = paper_LV

    best_row = results_df.loc[
        results_df["RMSECV"].idxmin()]

    best_x1 = int(best_row["LV_X1"])
    best_m2 = int(best_row["LV_M2"])
    best_x2 = int(best_row["LV_X2"])

    # =========================
    # BEST PATH
    # =========================
    best_path_x = []
    best_path_y = []
    best_labels = []

    for x1 in range(0, best_x1 + 1):

        row = results_df[
            (results_df["LV_X1"] == x1) &
            (results_df["LV_M2"] == 0) &
            (results_df["LV_X2"] == 0)]

        if len(row):
            best_path_x.append(x1)
            best_path_y.append(row["RMSECV"].iloc[0])
            best_labels.append((x1, 0, 0))

    for m2 in range(1, best_m2 + 1):

        row = results_df[
            (results_df["LV_X1"] == best_x1) &
            (results_df["LV_M2"] == m2) &
            (results_df["LV_X2"] == 0)]

        if len(row):
            best_path_x.append(best_x1 + m2)
            best_path_y.append(row["RMSECV"].iloc[0])
            best_labels.append((best_x1, m2, 0))

    for x2 in range(1, best_x2 + 1):

        row = results_df[
            (results_df["LV_X1"] == best_x1) &
            (results_df["LV_M2"] == best_m2) &
            (results_df["LV_X2"] == x2)]

        if len(row):
            best_path_x.append(best_x1 + best_m2 + x2)
            best_path_y.append(row["RMSECV"].iloc[0])
            best_labels.append((best_x1, best_m2, x2))

    # =========================
    # PAPER PATH
    # =========================
    paper_path_x = []
    paper_path_y = []
    paper_labels = []

    row = results_df[
        (results_df["LV_X1"] == lv_x1_p) &
        (results_df["LV_M2"] == 0) &
        (results_df["LV_X2"] == 0)]

    if len(row):
        paper_path_x.append(lv_x1_p)
        paper_path_y.append(row["RMSECV"].iloc[0])
        paper_labels.append((lv_x1_p, 0, 0))

    for m2 in range(1, lv_m2_p + 1):

        row = results_df[
            (results_df["LV_X1"] == lv_x1_p) &
            (results_df["LV_M2"] == m2) &
            (results_df["LV_X2"] == 0)]

        if len(row):
            paper_path_x.append(lv_x1_p + m2)
            paper_path_y.append(row["RMSECV"].iloc[0])
            paper_labels.append((lv_x1_p, m2, 0))

    for x2 in range(1, lv_x2_p + 1):

        row = results_df[
            (results_df["LV_X1"] == lv_x1_p) &
            (results_df["LV_M2"] == lv_m2_p) &
            (results_df["LV_X2"] == x2)]

        if len(row):
            paper_path_x.append(lv_x1_p + lv_m2_p + x2)
            paper_path_y.append(row["RMSECV"].iloc[0])
            paper_labels.append((lv_x1_p, lv_m2_p, x2))

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(10, 6))

    plt.scatter(results_df["TotalLV"], results_df["RMSECV"], alpha=0.25, label="All configurations")

    plt.plot(best_path_x, best_path_y, marker="o", linewidth=3, color="red", label=f"Best path ({best_x1},{best_m2},{best_x2})")

    plt.plot(paper_path_x, paper_path_y, "-s", linewidth=3, color="blue", label=f"Paper path ({lv_x1_p},{lv_m2_p},{lv_x2_p})")

    for x, y, label in zip(best_path_x, best_path_y, best_labels):
        plt.text(x, y + 0.02, f"{label[0]},{label[1]},{label[2]}", fontsize=10, fontweight="bold", color="red", ha="center", zorder=20)

    for x, y, label in zip(paper_path_x, paper_path_y, paper_labels):
        plt.text(x, y + 0.02, f"{label[0]},{label[1]},{label[2]}", fontsize=10, fontweight="bold", color="blue", ha="center", zorder=20)

    plt.scatter(best_x1 + best_m2 + best_x2, best_row["RMSECV"], s=150, color="black", zorder=10, label="Best model")

    plt.xlabel("Total Number of Components", fontsize = 18)
    plt.ylabel("RMSECV", fontsize = 18)
    plt.title(f"{case_name}: SO-PLS Måge Plot", fontsize = 20)
    plt.grid(alpha=0.3)
    plt.legend(fontsize = 12)
    plt.xticks(range(0, 19), fontsize = 14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


def run_sopls_case(case_name, x1_file, m2_file, x2_file, y_file, paper_LV, make_plots = True):
    print(f"\n==============================")
    print(f"Running SO-PLS case: {case_name}")
    print(f"Paper LV: {paper_LV}")
    print(f"==============================")

    X1_train = pd.read_excel(x1_file).values
    M2_train = pd.read_excel(m2_file).values
    X2_train = pd.read_excel(x2_file).values
    Y_train  = pd.read_excel(y_file).values

    X1_train_scaled = scaling(X1_train)
    M2_train_scaled = scaling(M2_train)
    X2_train_scaled = scaling(X2_train)
    Y_train_scaled  = scaling(Y_train)

    X1n, M2n, X2n = normalize_blocks(
        X1_train_scaled,
        M2_train_scaled,
        X2_train_scaled
    )

    X_blocks = [X1n, M2n, X2n]

    paper_model = my_sopls(
        X_blocks,
        Y_train_scaled,
        paper_LV
    )

    rmse_fit = np.sqrt(
        mean_squared_error(
            Y_train_scaled,
            paper_model["Yhat_total"]
        )
    )

    r2_fit = r2_score(
        Y_train_scaled,
        paper_model["Yhat_total"]
    )

    mae_fit = np.mean(
        np.abs(Y_train_scaled - paper_model["Yhat_total"])
    )

    bias_fit = np.mean(
        paper_model["Yhat_total"] - Y_train_scaled
    )

    rmsecv, q2, r2cv, mae_cv, bias_cv, q2_blocks, r2cv_blocks, Y_cv, Y_true_cv = sopls_rmsecv_raw(
        X1_train,
        M2_train,
        X2_train,
        Y_train,
        paper_LV,
        return_predictions=True,
        return_block_contrib=True
    )

    boot = sopls_cv_contrib_bootstrap(
        X1_train,
        M2_train,
        X2_train,
        Y_train,
        paper_LV,
        n_reps=50
    )

    print("\n===== FIT METRICS =====")
    print("RMSE fit:", rmse_fit)
    print("MAE fit:", mae_fit)
    print("Bias fit:", bias_fit)
    print("R2 fit:", r2_fit)
    print("SS blocks:", paper_model["SS"])
    print("Total explained:", sum(paper_model["SS"]))

    print("\n===== CV METRICS =====")
    print("RMSECV:", rmsecv)
    print("Q2:", q2)
    print("CV R2:", r2cv)

    print("\n===== Q2 BLOCK CONTRIBUTIONS =====")
    print("X1:", q2_blocks[0])
    print("M2:", q2_blocks[1])
    print("X2:", q2_blocks[2])
    print("Total:", sum(q2_blocks))

    print("\n===== BOOTSTRAP CV CONTRIBUTIONS =====")
    print("X1:", boot["mean"][0], "+/-", boot["sd"][0])
    print("M2:", boot["mean"][1], "+/-", boot["sd"][1])
    print("X2:", boot["mean"][2], "+/-", boot["sd"][2])
    print("Total:", boot["total_mean"], "+/-", boot["total_sd"])

    results_df = sopls_grid_search(
        X_blocks,
        Y_train_scaled,
        X1_train,
        M2_train,
        X2_train,
        Y_train,
        max_lv=6)
    if make_plots:
        plot_sopls_mage(results_df, paper_LV, case_name)
        cum_mean = np.array([boot["mean"][0], boot["mean"][0] + boot["mean"][1], boot["mean"][0] + boot["mean"][1] + boot["mean"][2]])

        cum_sd = np.array([boot["sd"][0], np.sqrt(boot["sd"][0]**2 + boot["sd"][1]**2), boot["total_sd"]])

        blocks = ["X1", "M2", "X2"]
        x = np.arange(3)

        plt.figure(figsize=(4, 5))

        plt.errorbar(x, cum_mean, yerr=cum_sd, fmt="o", color="black", capsize=6, markersize=8)

        plt.plot(x, cum_mean, "--", color="black", linewidth=2)

        for i in range(3):
            plt.text(x[i], cum_mean[i] + 3, f"{cum_mean[i]:.1f}%", ha="center", fontsize=12)

        plt.xticks(x, blocks, fontsize = 14)
        plt.yticks(fontsize=14)
        plt.ylabel("Explained variance [%]", fontsize = 18)
        plt.xlabel("Blocks", fontsize = 18)
        plt.ylim(0, 105)
        plt.title(f"{case_name}: explained variance", fontsize = 20)
        plt.tight_layout()
        plt.show()

    return {
        "case_name": case_name,
        "X_blocks": X_blocks,
        "Y_scaled": Y_train_scaled,
        "paper_model": paper_model,
        "rmse_fit": rmse_fit,
        "r2_fit": r2_fit,
        "rmsecv": rmsecv,
        "q2": q2,
        "q2_blocks": q2_blocks,
        "boot": boot}

case_true = run_sopls_case(
    case_name="Case (i)",
    x1_file="X1.xlsx",
    m2_file="M2.xlsx",
    x2_file="X2.xlsx",
    y_file="Y.xlsx",
    paper_LV=[4, 6, 2],
    make_plots=False)

case_mis = run_sopls_case(
    case_name="Case (ii)",
    x1_file="X1_mis.xlsx",
    m2_file="M2_mis.xlsx",
    x2_file="X2_mis.xlsx",
    y_file="Y_mis.xlsx",
    paper_LV=[2, 3, 6],
    make_plots=True)

case_true2 = run_sopls_case(
    case_name="Case (i)",
    x1_file="X1.xlsx",
    m2_file="M2.xlsx",
    x2_file="X2.xlsx",
    y_file="Y.xlsx",
    paper_LV=[1, 3, 5],
    make_plots=False)

case_mis2 = run_sopls_case(
    case_name="Case (ii)",
    x1_file="X1_mis.xlsx",
    m2_file="M2_mis.xlsx",
    x2_file="X2_mis.xlsx",
    y_file="Y_mis.xlsx",
    paper_LV=[1, 3, 5],
    make_plots=False)

case_true3 = run_sopls_case(
    case_name="Case (i)",
    x1_file="X1.xlsx",
    m2_file="M2.xlsx",
    x2_file="X2.xlsx",
    y_file="Y.xlsx",
    paper_LV=[0, 4, 5],
    make_plots=False)

case_mis3 = run_sopls_case(
    case_name="Case (ii)",
    x1_file="X1_mis.xlsx",
    m2_file="M2_mis.xlsx",
    x2_file="X2_mis.xlsx",
    y_file="Y_mis.xlsx",
    paper_LV=[0, 4, 5],
    make_plots=False)

case_true4 = run_sopls_case(
    case_name="Case (i)",
    x1_file="X1.xlsx",
    m2_file="M2.xlsx",
    x2_file="X2.xlsx",
    y_file="Y.xlsx",
    paper_LV=[0, 4, 2],
    make_plots=False)

case_mis4 = run_sopls_case(
    case_name="Case (ii)",
    x1_file="X1_mis.xlsx",
    m2_file="M2_mis.xlsx",
    x2_file="X2_mis.xlsx",
    y_file="Y_mis.xlsx",
    paper_LV=[2, 3, 5],
    make_plots=False)


