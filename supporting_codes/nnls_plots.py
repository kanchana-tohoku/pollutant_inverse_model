import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os



# ============================================================
# 1. PLOT CONDITION NUMBER VS k
# ============================================================
def plot_condition_number(df_ideal):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df_ideal, x="k", y="cond_A", hue="k", s=80, palette="viridis")
    plt.title("Condition Number vs Number of Sources (k)")
    plt.ylabel("Condition Number")
    plt.xlabel("Number of Sources (k)")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# 2. RELATIVE ERROR VS CONDITION NUMBER
# ============================================================
def plot_error_vs_condition(df_ideal):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df_ideal, x="cond_A", y="rel_err", hue="k", palette="turbo")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Condition Number (log)")
    plt.ylabel("Relative Error (log)")
    plt.title("Relative Error vs Condition Number")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# 3. VIOLIN PLOT FOR CV OF CONTRIBUTIONS (MC)
# ============================================================
def plot_contribution_violin(df_mc, k_value):
    df_k = df_mc[df_mc["k"] == k_value]
    if df_k.empty:
        print("No k =", k_value)
        return

    cv_data = []
    for _, row in df_k.iterrows():
        for i, val in enumerate(row["cv_x_est"]):
            cv_data.append({"Source": f"Src{i+1}", "cv": val})

    df_plot = pd.DataFrame(cv_data)

    plt.figure(figsize=(8,5))
    sns.violinplot(data=df_plot, x="Source", y="cv", inner="quartile", palette="Set2")
    plt.title(f"Coefficient of Variation of Contributions (k={k_value})")
    plt.ylabel("CV")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# 4. HEATMAP OF MEAN CONTRIBUTIONS
# ============================================================
def plot_mean_contribution_heatmap(df_mc, k_value):
    df_k = df_mc[df_mc["k"] == k_value]
    if df_k.empty:
        print("No k =", k_value)
        return

    heatmap_matrix = np.array([row["mean_x_est"] for _, row in df_k.iterrows()])
    labels = df_k["combination"].tolist()

    plt.figure(figsize=(10,7))
    sns.heatmap(
        heatmap_matrix, cmap="magma",
        xticklabels=[f"Src{i+1}" for i in range(heatmap_matrix.shape[1])],
        yticklabels=labels
    )
    plt.title(f"Mean Contribution Estimates (k={k_value})")
    plt.xlabel("Source Index")
    plt.ylabel("Combination")
    plt.show()


# ============================================================
# 5. RMSE COMPARISON BARPLOT
# ============================================================
def plot_rmse_comparison(df_mc, k_value):
    df_k = df_mc[df_mc["k"] == k_value]
    if df_k.empty:
        print("No k =", k_value)
        return

    plt.figure(figsize=(8,5))
    sns.barplot(data=df_k, x="combination", y="mean_rmse", palette="viridis")
    plt.xticks(rotation=90)
    plt.ylabel("Mean RMSE")
    plt.title(f"RMSE Comparison for Combinations (k={k_value})")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# 6. TRUE vs ESTIMATED CONTRIBUTION SCATTER
# ============================================================
def plot_truth_vs_estimation(df_mc, k_value, source_index=0):
    df_k = df_mc[df_mc["k"] == k_value]
    if df_k.empty:
        print("No k =", k_value)
        return

    true_vals = []
    est_vals = []

    for _, row in df_k.iterrows():
        true_vals.append(row["true_x"][source_index])
        est_vals.append(row["mean_x_est"][source_index])

    plt.figure(figsize=(6,5))
    plt.scatter(true_vals, est_vals, alpha=0.7)
    plt.xlabel("True Contribution")
    plt.ylabel("Estimated Mean Contribution")
    plt.title(f"Truth vs Estimated Contribution (Source {source_index+1}, k={k_value})")
    plt.grid(alpha=0.3)
    plt.show()

# ============================================================
# Number of Sources vs Mean RMSE
# ============================================================

def plot_sources_vs_rmse(df, rmse_col="mean_rmse"):

    df_group = df.groupby("k")[rmse_col].mean().reset_index()

    plt.figure(figsize=(8,6))
    plt.scatter(df_group["k"], df_group[rmse_col], s=80)
    plt.plot(df_group["k"], df_group[rmse_col], linewidth=1)

    plt.xlabel("Number of Sources (k)", fontsize=12)
    plt.ylabel("Mean RMSE", fontsize=12)
    plt.title("Number of Sources vs Mean RMSE", fontsize=14)
    plt.grid(True, alpha=0.3)

    # ---- FIX ----
    plt.xticks(df_group["k"])  
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

    return df_group


# ============================================================
#relative error vs number of sources
# ============================================================
def plot_sources_vs_relerr(df, relerr_col="mean_relerr"):
    df_group = df.groupby("k")[relerr_col].mean().reset_index()

    plt.figure(figsize=(8,6))
    plt.scatter(df_group["k"], df_group[relerr_col], s=80)
    plt.plot(df_group["k"], df_group[relerr_col], linewidth=1)

    plt.xlabel("Number of Sources (k)", fontsize=12)
    plt.ylabel("Mean Relative Error", fontsize=12)
    plt.title("Number of Sources vs Mean Relative Error", fontsize=14)
    plt.grid(True, alpha=0.3)

    # ---- FIX ----
    plt.xticks(df_group["k"])
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

    return df_group


# ============================================================
# Combined Plot: All Scenarios — RMSE
# ============================================================
def plot_rmse_all_scenarios(df, rmse_col="rmse", title_suffix = ""):
    """
    df must contain:
        - 'k'
        - rmse_col
        - 'scenario'
    """

    plt.figure(figsize=(9, 6))

    for scen, df_s in df.groupby("scenario"):
        df_g = df_s.groupby("k")[rmse_col].mean().reset_index()
        plt.plot(df_g["k"], df_g[rmse_col], marker="o", label=scen)

    plt.xlabel("Number of Sources (k)", fontsize=18)
    plt.ylabel("Mean RMSE", fontsize=18)
    plt.title(f"Number of Sources vs Mean RMSE {title_suffix}", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Scenario", fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

# ============================================================
# Combined Plot: All Scenarios — Relative Error
# ============================================================

def plot_relerr_all_scenarios(df, relerr_col="relerr"):
   
    plt.figure(figsize=(9, 6))

    for scen, df_s in df.groupby("scenario"):
        df_g = df_s.groupby("k")[relerr_col].mean().reset_index()
        plt.plot(df_g["k"], df_g[relerr_col], marker="o", label=scen)

    plt.xlabel("Number of Sources (k)", fontsize=18)
    plt.ylabel("Mean Relative Error", fontsize=18)
    plt.title("Number of Sources vs Mean Relative Error (All Scenarios)", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Scenario", fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

# ============================================================
# Combined Plot: All K Values in One Plot (Overlaid)
# ============================================================

def plot_truth_vs_est_single(results_dict, source_index_list, scenario_name):
    plt.figure(figsize=(7,7))

    for k, df_k in results_dict.items():
        for sidx in source_index_list:
            plt.scatter(
                df_k[f"true_x_{sidx}"],
                df_k[f"mean_x_est_{sidx}"],
                alpha=0.6,
                label=f"K={k} Src={sidx}"
            )

    # 1:1 reference
    plt.plot([0,1], [0,1], "k--", linewidth=1)

    plt.xlabel("True Contribution")
    plt.ylabel("Estimated Contribution")
    plt.title(f"Truth vs Estimated Contributions – {scenario_name}")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.show()
    
# ============================================================
# BOX & WHISKER PLOT — k vs Relative Error
# ============================================================
def plot_box_relerr(df, relerr_col="mean_relerr", title_suffix=""):

    plt.figure(figsize=(6, 6))

    sns.boxplot(
        data=df,
        x="k",
        y=relerr_col,
        palette="Set3",
        showfliers=True
    )


    plt.xlabel("Number of Sources (k)", fontsize=18)
    plt.ylabel("Relative Error", fontsize=18)
    plt.title(f"Relative Error vs Number of Sources {title_suffix}", fontsize=16)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# BOX & WHISKER PLOT — k vs RMSE
# ============================================================
def plot_box_rmse(df, rmse_col="rmse", title_suffix="", output_dir = "", filename = ""):


    plt.figure(figsize=(6, 6))

    sns.boxplot(
        data=df,
        x="k",
        y="rmse",
        palette="Set3",
        showfliers=True
    )

    plt.xlabel("Number of Sources (k)", fontsize=18)
    plt.ylabel("RMSE", fontsize=18)
    plt.title(f"RMSE vs Number of Sources {title_suffix}", fontsize=16)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()





