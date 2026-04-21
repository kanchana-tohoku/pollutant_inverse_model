import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/uncertainty_analysis/nnls_uncertainty_results.csv"
OUTPUT_DIR = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Convert string arrays back to numpy arrays if needed
def parse_array(col):
    return df[col].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ') if isinstance(x, str) else x)

if isinstance(df["mean_x_est"].iloc[0], str):
    df["mean_x_est"] = parse_array("mean_x_est")
    df["std_x_est"] = parse_array("std_x_est")
    df["true_x"] = parse_array("true_x")

# ============================================================
# 1. RMSE vs Number of Sources
# ============================================================

plt.figure()
for label, grp in df.groupby("signature_uncertainty"):
    grp_mean = grp.groupby("k")["rmse"].mean()
    plt.plot(grp_mean.index, grp_mean.values, marker='o', label=f"Signature Unc: {label}")

plt.xlabel("Number of Sources (k)")
plt.ylabel("RMSE")
plt.title("RMSE vs Number of Sources")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "rmse_vs_sources.png"), dpi=300)
plt.close()

# ============================================================
# 2. Relative Error vs Number of Sources
# ============================================================

plt.figure()
for label, grp in df.groupby("signature_uncertainty"):
    grp_mean = grp.groupby("k")["relerr"].mean()
    plt.plot(grp_mean.index, grp_mean.values, marker='o', label=f"Signature Unc: {label}")

plt.xlabel("Number of Sources (k)")
plt.ylabel("Relative Error")
plt.title("Relative Error vs Number of Sources")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "relerr_vs_sources.png"), dpi=300)
plt.close()

# ============================================================
# 3. Boxplot of Relative Error (Uncertainty Spread)
# ============================================================

plt.figure()

data = [df[df["k"] == k]["relerr"] for k in sorted(df["k"].unique())]

plt.boxplot(data)
plt.xticks(range(1, len(data)+1), sorted(df["k"].unique()))

plt.xlabel("Number of Sources (k)")
plt.ylabel("Relative Error")
plt.title("Uncertainty Spread in Source Estimation")

plt.savefig(os.path.join(OUTPUT_DIR, "relerr_boxplot.png"), dpi=300)
plt.close()

# ============================================================
# 4. Condition Number vs RMSE
# ============================================================

plt.figure()

plt.scatter(df["cond_A"], df["rmse"])

plt.xlabel("Condition Number")
plt.ylabel("RMSE")
plt.title("Effect of Ill-Conditioning on Error")

plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "cond_vs_rmse.png"), dpi=300)
plt.close()

# ============================================================
# 5. True vs Estimated Contributions (Example Case)
# ============================================================

example = df.iloc[0]

true_x = example["true_x"]
est_x = example["mean_x_est"]
std_x = example["std_x_est"]

x = np.arange(len(true_x))

plt.figure()

plt.errorbar(x, est_x, yerr=std_x, fmt='o', label="Estimated")
plt.scatter(x, true_x, marker='x', label="True")

plt.xlabel("Source Index")
plt.ylabel("Contribution")
plt.title("True vs Estimated Contributions (with uncertainty)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "truth_vs_estimation.png"), dpi=300)
plt.close()

# ============================================================
# 6. Coefficient of Variation (Stability Indicator)
# ============================================================

cv_list = []

for _, row in df.iterrows():
    cv = np.mean(row["std_x_est"] / (row["mean_x_est"] + 1e-12))
    cv_list.append(cv)

df["mean_cv"] = cv_list

plt.figure()

for label, grp in df.groupby("signature_uncertainty"):
    grp_mean = grp.groupby("k")["mean_cv"].mean()
    plt.plot(grp_mean.index, grp_mean.values, marker='o', label=f"Signature Unc: {label}")

plt.xlabel("Number of Sources (k)")
plt.ylabel("Mean Coefficient of Variation")
plt.title("Stability of Source Apportionment")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "cv_vs_sources.png"), dpi=300)
plt.close()

print("========================================")
print("All publication-quality plots generated.")
print(f"Saved in: {OUTPUT_DIR}")
print("========================================")