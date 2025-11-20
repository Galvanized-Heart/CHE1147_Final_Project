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
    "rf": "Random Forest",  # Added based on your previous data
    "Linear": "Linear Regression" # Handling potential capitalization differences
}

target_map = {
    "log_kcat_value": "Log $k_{cat}$ Value", # Using LaTeX for subscript
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

# 3. SETUP GRID
unique_targets = sorted(df['target'].unique())
unique_models = sorted(df['model'].unique())
n_rows = len(unique_targets)
n_cols = len(unique_models)

fig, axes = plt.subplots(
    nrows=n_rows, 
    ncols=n_cols, 
    figsize=(10 * n_cols, 6 * n_rows), 
    squeeze=False,
    gridspec_kw={'wspace': 0.6, 'hspace': 0.4}
)

# --- MONKEY PATCH TO PREVENT SHAP FROM BREAKING SUBPLOTS ---
original_tight_layout = plt.tight_layout
plt.tight_layout = lambda *args, **kwargs: None
# -----------------------------------------------------------

try:
    for i, target in enumerate(unique_targets):
        for j, model in enumerate(unique_models):
            ax = axes[i, j]
            
            subset = df[(df['model'] == model) & (df['target'] == target)]
            
            if subset.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_axis_off()
            else:
                # Pivot Data
                shap_vals = subset.pivot(index='sample_id', columns='feature', values='shap_value')
                feature_vals = subset.pivot(index='sample_id', columns='feature', values='feature_value')
                
                # Ensure columns align
                feature_vals = feature_vals[shap_vals.columns]
                
                # --- APPLY FEATURE MAPPING HERE ---
                # We rename the columns of the DataFrame passed to SHAP.
                # .rename() will use the new name if found in dict, or keep original if not.
                feature_vals_renamed = feature_vals.rename(columns=feature_map)
                # ----------------------------------

                # Set Active Axis
                plt.sca(ax)
                
                # Generate SHAP Plot using the RENAMED dataframe
                shap.summary_plot(
                    shap_vals.values,       # Numerical SHAP values
                    feature_vals_renamed,   # Feature values with Mapped Column Names
                    show=False, 
                    color_bar=False,
                    plot_size=None 
                )
                
                # --- APPLY MODEL & TARGET MAPPING HERE ---
                # Use .get(key, key) to return the original string if key is missing
                display_model = model_map.get(model, model)
                display_target = target_map.get(target, target)
                
                ax.set_title(f"{display_model}\n{display_target}", fontsize=14, pad=15)
                ax.set_xlabel("SHAP value" if i == n_rows - 1 else "")

            # Draw the Outline Frame
            pad_left = 0.6   
            pad_right = 0.05 
            pad_top = 0.15    
            pad_bottom = 0.15 

            rect = patches.Rectangle(
                (0 - pad_left, 0 - pad_bottom), 
                1 + pad_left + pad_right, 
                1 + pad_bottom + pad_top, 
                linewidth=2, 
                edgecolor='black', 
                facecolor='none', 
                transform=ax.transAxes, 
                clip_on=False,          
                zorder=10               
            )
            ax.add_patch(rect)

finally:
    plt.tight_layout = original_tight_layout

# Finalize and Save
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
output_path = "/Users/joshgoldman/Documents/Courses/CHE1147/CHE1147_Final_Project/reports/figures/shap_summary_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print("Done.")

