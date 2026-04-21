import numpy as np
import pandas as pd
from scipy.optimize import nnls
from itertools import combinations
import os

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_4sources_case3.csv"
OUTPUT_DIR = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/uncertainty_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

K_LIST = [2, 3, 4, 5, 6, 7, 8]
MC_RUNS = 100
OBS_UNCERTAINTY = 0.10   # 10% observation noise

# Metal-wise uncertainty (set None to disable)
SIGNATURE_UNCERTAINTY = [0.06, 0.08, 0.05, 0.04, 0.07]
# SIGNATURE_UNCERTAINTY = None


# ============================================================
# NNLS SOLVER CLASS
# ============================================================

class NNLSUncertainty:

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        metals = ["Metal1", "Metal2", "Metal3", "Metal4", "Metal5"]

        self.A_full = df[metals].values.T   # (5 × n_sources)
        self.source_ids = df["source_ID"].values
        self.x_true_full = df["contribution_x"].values

    # --------------------------------------------------------
    def pick_combos(self, n, k):
        return list(combinations(range(n), k))

    # --------------------------------------------------------
    def condition_number(self, A):
        return np.linalg.cond(A)

    # --------------------------------------------------------
    def metrics(self, x_est, x_true, A, b_clean):
        residual = A @ x_est - b_clean
        rmse = np.sqrt(np.mean(residual**2))
        rel_err = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)
        return rmse, rel_err

    # ========================================================
    # CORE METHOD: OBSERVATION NOISE + OPTIONAL SIGNATURE UNCERTAINTY
    # ========================================================

    def run_analysis(self, k_list, mc_runs, obs_uncertainty, signature_uncertainty):

        results = []

        n_sources = len(self.source_ids)

        for k in k_list:
            combos = self.pick_combos(n_sources, k)

            for combo in combos:

                A_sub = self.A_full[:, combo]

                # -------------------------------------------
                # APPLY SIGNATURE UNCERTAINTY (METAL-WISE)
                # -------------------------------------------
                if signature_uncertainty is not None:
                    delta = np.array(signature_uncertainty).reshape(-1, 1)
                    A_used = A_sub * (1 + delta)
                else:
                    A_used = A_sub.copy()

                x_true = self.x_true_full[list(combo)]

                b_clean = A_used @ x_true

                condA = self.condition_number(A_used)

                x_all = []
                rmse_all = []
                err_all = []

                # -------------------------------------------
                # MONTE CARLO SIMULATION (OBSERVATION NOISE ONLY)
                # -------------------------------------------
                for _ in range(mc_runs):

                    noise = np.random.uniform(
                        -obs_uncertainty,
                        obs_uncertainty,
                        size=b_clean.shape
                    )

                    b_noisy = b_clean * (1 + noise)

                    x_est, _ = nnls(A_used, b_noisy)

                    rmse, rel_err = self.metrics(x_est, x_true, A_used, b_clean)

                    x_all.append(x_est)
                    rmse_all.append(rmse)
                    err_all.append(rel_err)

                x_all = np.array(x_all)

                results.append({
                    "k": k,
                    "combination": ",".join(self.source_ids[list(combo)]),
                    "cond_A": condA,
                    "obs_uncertainty": obs_uncertainty,
                    "signature_uncertainty": signature_uncertainty,
                    "mean_rmse": np.mean(rmse_all),
                    "std_rmse": np.std(rmse_all),
                    "mean_relerr": np.mean(err_all),
                    "std_relerr": np.std(err_all),
                    "mean_x_est": x_all.mean(axis=0),
                    "std_x_est": x_all.std(axis=0),
                    "cv_x_est": x_all.std(axis=0) / (x_all.mean(axis=0) + 1e-12),
                    "true_x": x_true
                })

        return pd.DataFrame(results)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    solver = NNLSUncertainty(CSV_PATH)

    df_results = solver.run_analysis(
        k_list=K_LIST,
        mc_runs=MC_RUNS,
        obs_uncertainty=OBS_UNCERTAINTY,
        signature_uncertainty=SIGNATURE_UNCERTAINTY
    )

    # Rename for consistency
    df_results = df_results.rename(columns={
        "mean_rmse": "rmse",
        "mean_relerr": "relerr"
    })

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "nnls_uncertainty_results.csv")
    df_results.to_csv(output_file, index=False)

    print("========================================")
    print("NNLS Uncertainty Analysis Completed")
    print(f"Results saved to: {output_file}")
    print("========================================")