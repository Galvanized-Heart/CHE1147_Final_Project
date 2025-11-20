import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 1. CONFIGURATION & MAPPINGS
# ---------------------------------------------------------
input_path = "/Users/joshgoldman/Documents/Courses/CHE1147/CHE1147_Final_Project/reports/data/parity_experiment_data.csv"
output_dir = "/Users/joshgoldman/Documents/Courses/CHE1147/CHE1147_Final_Project/reports/figures/"

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

# 2. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(input_path)

# Get unique experiments to create separate figures per experiment
unique_experiments = df['experiment'].unique()

# 3. GENERATE PLOTS
# ---------------------------------------------------------
for exp_name in unique_experiments:
    print(f"Processing Experiment: {exp_name}...")
    
    # Filter data for this experiment
    exp_df = df[df['experiment'] == exp_name]
    
    unique_targets = sorted(exp_df['target'].unique())
    unique_models = sorted(exp_df['model'].unique())
    
    n_rows = len(unique_targets)
    n_cols = len(unique_models)
    
    # Setup Figure
    fig, axes = plt.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        figsize=(6 * n_cols, 6 * n_rows), # Square-ish aspect ratio for parity plots
        squeeze=False,
        gridspec_kw={'wspace': 0.4, 'hspace': 0.4}
    )
    
    for i, target in enumerate(unique_targets):
        for j, model in enumerate(unique_models):
            ax = axes[i, j]
            
            # Filter subset
            subset = exp_df[(exp_df['model'] == model) & (exp_df['target'] == target)]
            
            # Handle empty data
            if subset.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_axis_off()
                continue
            
            y_true = subset['y_true']
            y_pred = subset['y_pred']
            
            # --- PLOT DATA ---
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=40, color='#1f77b4')
            
            # --- ADD PARITY LINE (y=x) ---
            # Determine limits to ensure the line covers the data
            all_vals = pd.concat([y_true, y_pred])
            min_val = all_vals.min()
            max_val = all_vals.max()
            
            # Add a little buffer
            buffer = (max_val - min_val) * 0.1
            lims = [min_val - buffer, max_val + buffer]
            
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0) # Dashed diagonal line
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            # --- CALCULATE METRICS ---
            if len(subset) > 1:
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                # Add text box with stats
                textstr = f'$R^2 = {r2:.2f}$\n$RMSE = {rmse:.2f}$\n$N = {len(subset)}$'
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

            # --- LABELS & TITLES ---
            display_model = model_map.get(model, model)
            display_target = target_map.get(target, target)
            
            ax.set_title(f"{display_model}\n{display_target}", fontsize=12, pad=10)
            
            # Labels: Only bottom row gets X label, only left col gets Y label (optional, 
            # but usually cleaner. Remove 'if' conditions if you want labels on every plot)
            if i == n_rows - 1:
                ax.set_xlabel("True Value")
            if j == 0:
                ax.set_ylabel("Predicted Value")
            
            # --- DRAW OUTLINE BOX (Same style as SHAP plots) ---
            pad_left = 0.2   
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

    # Save
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Save filename based on experiment name
    save_name = f"parity_plot_{exp_name}.png"
    save_path = output_dir + save_name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

print("All plots generated.")