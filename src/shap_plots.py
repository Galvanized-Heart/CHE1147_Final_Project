import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shap
import numpy as np

# 1. DEFINE MAPPINGS
# ---------------------------------------------------------
model_map = {
    "nn": "Neural Network",
    "linear": "Linear Regression",
    "xgb": "Gradient Boosting",
    "rf": "Random Forest",
    "Linear": "Linear Regression"
}

target_map = {
    "log_kcat_value": "Log $k_{cat}$ Value",
    "log_km_value": "Log $K_M$ Value"
}

feature_map = {
    "log_seq_length": "Log Sequence Length",
    "log_seq_mol_wt": "Log Sequence Molecular Weight",
    "Log p": "Log Partition Coefficient (p)",
    "log1p_tpsa": "Log1p TPSA",
    "log_mot_wt": "Log Molecular Weight",
    "log1p_num_h_donors": "Log1p Number of H Donors",
    "log1p_num_h_acceptors": "Log1p Number of H Acceptors",
    "log1p_num_rot_bonds": "Log1p Number of Rotatable Bonds",
    "linear_temperature_value": "Temperature",
    "linear_instability_value": "Instability Index",
    "linear_pI": "Isoelectric Point (pI)",
    "linear_aromaticity": "Aromaticity",
    "linear_pH_value": "pH",
}
# ---------------------------------------------------------

# 2. LOAD DATA
shap_csv_path = "/Users/joshgoldman/Documents/Courses/CHE1147/CHE1147_Final_Project/reports/data/shap_experiment_data.csv"
df = pd.read_csv(shap_csv_path)

# 3. SETUP GRID - 3 ROWS (MODELS) x 2 COLUMNS (TARGETS)
# --------------------------------------------------------------------
unique_targets = sorted(df['target'].unique())
unique_models = sorted(df['model'].unique())

# --- SWAPPED: n_rows is now based on models, n_cols on targets ---
n_rows = len(unique_models)
n_cols = len(unique_targets)

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(9 * n_cols, 7 * n_rows),
    squeeze=False,
    # --- Adjusted hspace for the new 3-row layout ---
    gridspec_kw={'wspace': 0, 'hspace': 0.35}
)

# --- MONKEY PATCH TO PREVENT SHAP FROM BREAKING SUBPLOTS ---
original_tight_layout = plt.tight_layout
plt.tight_layout = lambda *args, **kwargs: None
# -----------------------------------------------------------

try:
    # --- SWAPPED LOOPS: Outer loop is now models (rows), inner loop is targets (columns) ---
    for i, model in enumerate(unique_models):
        for j, target in enumerate(unique_targets):
            ax = axes[i, j]

            subset = df[(df['model'] == model) & (df['target'] == target)]

            if subset.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_axis_off()
            else:
                shap_vals = subset.pivot(index='sample_id', columns='feature', values='shap_value')
                feature_vals = subset.pivot(index='sample_id', columns='feature', values='feature_value')
                feature_vals = feature_vals[shap_vals.columns]
                feature_vals_renamed = feature_vals.rename(columns=feature_map)

                plt.sca(ax)
                shap.summary_plot(
                    shap_vals.values,
                    feature_vals_renamed,
                    show=False,
                    color_bar=False,
                    plot_size=None
                )

                display_model = model_map.get(model, model)
                display_target = target_map.get(target, target)
                ax.set_title(f"{display_model}\n{display_target}", fontsize=16, pad=15)

                # --- HIDE LABELS TO CREATE A CLEAN, COMPACT GRID (Logic remains the same) ---

                # 1. X-axis: Only show label and ticks on the bottom row
                if i < n_rows - 1:
                    ax.set_xlabel("")
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel("SHAP value")

                # 2. Y-axis: Only show label and ticks on the first column
                if j > 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelleft=False)

finally:
    # Restore the original tight_layout function
    plt.tight_layout = original_tight_layout

# --- Final saving: `bbox_inches='tight'` handles the layout automatically ---
output_path = "/Users/joshgoldman/Documents/Courses/CHE1147/CHE1147_Final_Project/reports/figures/shap_summary_plots_compact_3x2.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print("Done. Generated compact 3x2 plot with side-by-side subplots.")