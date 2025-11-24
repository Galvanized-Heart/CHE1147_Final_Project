import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load the provided data
df = pd.read_csv("combined_metrics.csv", index_col=False)

# 2. Preprocessing
exp_map = {
    'no_temp_ph_no_advanced': 'Baseline',
    'yes_temp_ph_no_advanced': '+Temp/pH',
    'yes_temp_ph_yes_advanced': '+Temp/pH\n+Advanced'
}
df['Experiment'] = df['experiment'].map(exp_map)

target_map = {
    'log_kcat_value': 'Log kcat',
    'log_km_value': 'Log KM'
}
df['Target'] = df['target'].map(target_map)
df['Model'] = df['model'].str.upper()

# 3. Shared Plotting Configurations
sns.set_theme(style="whitegrid", font_scale=1.1)
model_order = ['LINEAR', 'NN', 'XGB']

def add_bar_labels(ax):
    for container in ax.containers:
        # Adjust format to check for negative numbers or size
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

# ============================================================================
# PLOT SET 1: ERROR & ACCURACY (R2, MSE, MAE)
# ============================================================================

# We need 3 Rows (R2, MSE, MAE) and 2 Columns (kcat, KM)
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

metrics_error = [
    ('val_mse', 'Validation MSE (Lower is Better)'),
    ('val_mae', 'Validation MAE (Lower is Better)')
]

for row_idx, (metric_key, metric_title) in enumerate(metrics_error):
    
    # --- Column 0: kcat (Warm Colors) ---
    subset_kcat = df[(df['metric_type'] == metric_key) & (df['Target'] == 'Log kcat')]
    ax = axes1[row_idx, 0]
    
    sns.barplot(
        data=subset_kcat, x='Experiment', y='metric_value', hue='Model',
        palette="OrRd", hue_order=model_order, ax=ax
    )
    ax.set_ylabel(metric_title, fontsize=11)
    ax.set_xlabel('')
    add_bar_labels(ax)
    
    # Add zero line for R2
    if metric_key == 'val_r2':
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)

    # --- Column 1: KM (Cold Colors) ---
    subset_km = df[(df['metric_type'] == metric_key) & (df['Target'] == 'Log KM')]
    ax = axes1[row_idx, 1]
    
    sns.barplot(
        data=subset_km, x='Experiment', y='metric_value', hue='Model',
        palette="GnBu", hue_order=model_order, ax=ax
    )
    ax.set_ylabel('') # Hide ylabel for second column
    ax.set_xlabel('')
    add_bar_labels(ax)

    if metric_key == 'val_r2':
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)

# Set Main Titles on Top Row
axes1[0, 0].set_title('Log kcat Prediction Performance', fontsize=16, color='#B22222', weight='bold')
axes1[0, 1].set_title('Log KM Prediction Performance', fontsize=16, color='#00008B', weight='bold')

plt.tight_layout()
plt.savefig('error_accuracy_analysis.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved error_accuracy_analysis.svg")


# ============================================================================
# PLOT SET 2: CORRELATIONS (Pearson, Spearman)
# ============================================================================

# We need 2 Rows (Pearson, Spearman) and 2 Columns (kcat, KM)
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 14), sharey=True, sharex=True)

metrics_corr = [
    ('val_r2', 'Validation R² (Higher is Better)'),
    ('val_pearson', 'Pearson Correlation (Higher is Better)'),
    ('val_spearman', 'Spearman Rank Correlation (Higher is Better)')
]

for row_idx, (metric_key, metric_title) in enumerate(metrics_corr):
    
    # --- Column 0: kcat (Warm Colors) ---
    subset_kcat = df[(df['metric_type'] == metric_key) & (df['Target'] == 'Log kcat')]
    ax = axes2[row_idx, 0]
    
    sns.barplot(
        data=subset_kcat, x='Experiment', y='metric_value', hue='Model',
        palette="OrRd", hue_order=model_order, ax=ax
    )
    ax.set_ylabel(metric_title, fontsize=12)
    ax.set_xlabel('')
    add_bar_labels(ax)

    # --- Column 1: KM (Cold Colors) ---
    subset_km = df[(df['metric_type'] == metric_key) & (df['Target'] == 'Log KM')]
    ax = axes2[row_idx, 1]
    
    sns.barplot(
        data=subset_km, x='Experiment', y='metric_value', hue='Model',
        palette="GnBu", hue_order=model_order, ax=ax
    )
    ax.set_ylabel('')
    ax.set_xlabel('')
    add_bar_labels(ax)

# Set Main Titles on Top Row
axes2[0, 0].set_title('Log kcat Correlations', fontsize=16, color='#B22222', weight='bold')
axes2[0, 1].set_title('Log KM Correlations', fontsize=16, color='#00008B', weight='bold')

# Global Y Limit (Correlations are 0 to 1)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('correlation_analysis.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved correlation_analysis.svg")

# ============================================================================
# PLOT SET 1: ERROR & ACCURACY (R2, MSE, MAE)
# ============================================================================

# We need 3 Rows (R2, MSE, MAE) and 2 Columns (kcat, KM)
fig1, axes1 = plt.subplots(5, 2, figsize=(16, 20), sharex=True)

metrics_error = [
    ('val_r2', 'Validation R² (Higher is Better)'),
    ('val_pearson', 'Pearson Correlation (Higher is Better)'),
    ('val_spearman', 'Spearman Rank Correlation (Higher is Better)'),
    ('val_mse', 'Validation MSE (Lower is Better)'),
    ('val_mae', 'Validation MAE (Lower is Better)')
]

for row_idx, (metric_key, metric_title) in enumerate(metrics_error):
    
    # --- Column 0: kcat (Warm Colors) ---
    subset_kcat = df[(df['metric_type'] == metric_key) & (df['Target'] == 'Log kcat')]
    ax = axes1[row_idx, 0]
    
    sns.barplot(
        data=subset_kcat, x='Experiment', y='metric_value', hue='Model',
        palette="OrRd", hue_order=model_order, ax=ax
    )
    ax.set_ylabel(metric_title, fontsize=11)
    ax.set_xlabel('')
    add_bar_labels(ax)
    
    # Add zero line for R2
    if metric_key == 'val_r2':
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)

    # --- Column 1: KM (Cold Colors) ---
    subset_km = df[(df['metric_type'] == metric_key) & (df['Target'] == 'Log KM')]
    ax = axes1[row_idx, 1]
    
    sns.barplot(
        data=subset_km, x='Experiment', y='metric_value', hue='Model',
        palette="GnBu", hue_order=model_order, ax=ax
    )
    ax.set_ylabel('') # Hide ylabel for second column
    ax.set_xlabel('')
    add_bar_labels(ax)

    if metric_key == 'val_r2':
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)

# Set Main Titles on Top Row
axes1[0, 0].set_title('Log kcat Prediction Performance', fontsize=16, color='#B22222', weight='bold')
axes1[0, 1].set_title('Log KM Prediction Performance', fontsize=16, color='#00008B', weight='bold')

plt.tight_layout()
plt.savefig('all_accuracy_analysis.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved all_accuracy_analysis.svg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# --- (Assuming 'df' and 'model_order' are already loaded from previous steps) ---
# If not, re-run the data loading block from the previous response first.
# -----------------------------------------------------------------------------

# Setup layout: 2 Rows (Targets) x 5 Columns (Metrics)
fig, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True)

# Define the Metrics for the Columns
metrics_config = [
    ('val_r2', 'R²'),
    ('val_pearson', 'Pearson (r)'),
    ('val_spearman', 'Spearman (ρ)'),
    ('val_mse', 'MSE'),
    ('val_mae', 'MAE')
]

# Define Targets for the Rows
targets = ['Log kcat', 'Log KM']

# Helper for bar labels
def add_bar_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=0)

# --- LOOP ---
for row_idx, target_name in enumerate(targets):
    
    # Determine Color Scheme based on Row (Target)
    if target_name == 'Log kcat':
        palette = "OrRd"
        row_color = '#B22222' # Dark Red
        row_title = "Log kcat"
    else:
        palette = "GnBu"
        row_color = '#00008B' # Dark Blue
        row_title = "Log KM"

    for col_idx, (metric_key, metric_display) in enumerate(metrics_config):
        
        ax = axes[row_idx, col_idx]
        
        # Filter Data
        subset = df[(df['metric_type'] == metric_key) & (df['Target'] == target_name)]
        
        # Plot
        sns.barplot(
            data=subset, 
            x='Experiment', 
            y='metric_value', 
            hue='Model',
            palette=palette, 
            hue_order=['LINEAR', 'NN', 'XGB'], 
            ax=ax
        )
        
        # Add Values on bars
        add_bar_labels(ax)
        
        # --- STYLING ---
        
        # 1. Titles: Only on the top row to identify the column metric
        if row_idx == 0:
            ax.set_title(metric_display, fontsize=16, weight='bold', pad=15)
        else:
            ax.set_title('')

        # 2. Y-Axis Labels: 
        # Use the First Column to clearly label the Row Target (kcat vs KM)
        if col_idx == 0:
            ax.set_ylabel(f"{row_title}\nValue", fontsize=14, weight='bold', color=row_color)
        else:
            ax.set_ylabel('') # Hide Y label for inner columns
            
        # 3. X-Axis Labels: Rotate them to fit
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # 4. Remove individual legends (we will add one global legend)
        ax.get_legend().remove()
        
        # 5. Add Zero line for R2 (Column 0)
        if metric_key == 'val_r2':
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)

# --- GLOBAL LEGEND & LAYOUT ---

# Create a single shared legend at the bottom
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), 
           ncol=3, fontsize=14, title="Model Architecture", title_fontsize=14, frameon=False)

# Adjust layout to prevent overlapping labels
plt.tight_layout()
plt.subplots_adjust(bottom=0.15) # Make room for the legend at the bottom

# Save
plt.savefig('landscape_metrics_analysis.svg', format='svg', bbox_inches='tight')
plt.show()
print("Saved landscape_metrics_analysis.svg")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load Data
csv_data = """experiment,model,target,metric_type,metric_display_name,metric_value
no_temp_ph_no_advanced,linear,log_kcat_value,val_pearson,Validation Pearson,0.192422141882514
no_temp_ph_no_advanced,linear,log_kcat_value,val_spearman,Validation Spearman,0.1900998124725462
no_temp_ph_no_advanced,linear,log_km_value,val_pearson,Validation Pearson,0.36040127849910986
no_temp_ph_no_advanced,linear,log_km_value,val_spearman,Validation Spearman,0.3885214165127642
no_temp_ph_no_advanced,nn,log_kcat_value,val_pearson,Validation Pearson,0.5793511954493681
no_temp_ph_no_advanced,nn,log_kcat_value,val_spearman,Validation Spearman,0.5611918058740142
no_temp_ph_no_advanced,nn,log_km_value,val_pearson,Validation Pearson,0.6377166988846361
no_temp_ph_no_advanced,nn,log_km_value,val_spearman,Validation Spearman,0.6309057798850113
no_temp_ph_no_advanced,xgb,log_kcat_value,val_pearson,Validation Pearson,0.7340108889610774
no_temp_ph_no_advanced,xgb,log_kcat_value,val_spearman,Validation Spearman,0.7188276384036655
no_temp_ph_no_advanced,xgb,log_km_value,val_pearson,Validation Pearson,0.7780104374408131
no_temp_ph_no_advanced,xgb,log_km_value,val_spearman,Validation Spearman,0.7696868425487579
yes_temp_ph_no_advanced,linear,log_kcat_value,val_pearson,Validation Pearson,0.20993994403248656
yes_temp_ph_no_advanced,linear,log_kcat_value,val_spearman,Validation Spearman,0.20910745094401417
yes_temp_ph_no_advanced,linear,log_km_value,val_pearson,Validation Pearson,0.3765379457105969
yes_temp_ph_no_advanced,linear,log_km_value,val_spearman,Validation Spearman,0.4002135345612165
yes_temp_ph_no_advanced,nn,log_kcat_value,val_pearson,Validation Pearson,0.5824351389080056
yes_temp_ph_no_advanced,nn,log_kcat_value,val_spearman,Validation Spearman,0.5651555333450679
yes_temp_ph_no_advanced,nn,log_km_value,val_pearson,Validation Pearson,0.6413944861958312
yes_temp_ph_no_advanced,nn,log_km_value,val_spearman,Validation Spearman,0.6344737641722897
yes_temp_ph_no_advanced,xgb,log_kcat_value,val_pearson,Validation Pearson,0.7451977377602239
yes_temp_ph_no_advanced,xgb,log_kcat_value,val_spearman,Validation Spearman,0.7297463635733068
yes_temp_ph_no_advanced,xgb,log_km_value,val_pearson,Validation Pearson,0.7825259001848286
yes_temp_ph_no_advanced,xgb,log_km_value,val_spearman,Validation Spearman,0.7749182162036826
yes_temp_ph_yes_advanced,linear,log_kcat_value,val_pearson,Validation Pearson,0.43343945497774516
yes_temp_ph_yes_advanced,linear,log_kcat_value,val_spearman,Validation Spearman,0.5421800276360021
yes_temp_ph_yes_advanced,linear,log_km_value,val_pearson,Validation Pearson,0.5113235315358301
yes_temp_ph_yes_advanced,linear,log_km_value,val_spearman,Validation Spearman,0.6404217174905001
yes_temp_ph_yes_advanced,nn,log_kcat_value,val_pearson,Validation Pearson,0.7001223552129415
yes_temp_ph_yes_advanced,nn,log_kcat_value,val_spearman,Validation Spearman,0.6877016961743286
yes_temp_ph_yes_advanced,nn,log_km_value,val_pearson,Validation Pearson,0.7523884169507561
yes_temp_ph_yes_advanced,nn,log_km_value,val_spearman,Validation Spearman,0.7476077709930343
yes_temp_ph_yes_advanced,xgb,log_kcat_value,val_pearson,Validation Pearson,0.7857010168519809
yes_temp_ph_yes_advanced,xgb,log_kcat_value,val_spearman,Validation Spearman,0.7733680328994067
yes_temp_ph_yes_advanced,xgb,log_km_value,val_pearson,Validation Pearson,0.8057515956381123
yes_temp_ph_yes_advanced,xgb,log_km_value,val_spearman,Validation Spearman,0.7975930544963518
no_temp_ph_no_advanced,linear,log_kcat_value,val_mse,Validation MSE,10.559345722198486
no_temp_ph_no_advanced,linear,log_km_value,val_mse,Validation MSE,8.002262592315674
no_temp_ph_no_advanced,linear,log_kcat_value,val_mae,Validation MAE,2.5756524801254272
no_temp_ph_no_advanced,linear,log_km_value,val_mae,Validation MAE,2.2113925218582158
no_temp_ph_no_advanced,linear,log_kcat_value,val_r2,Validation R²,0.0370259582996368
no_temp_ph_no_advanced,linear,log_km_value,val_r2,Validation R²,0.1298518478870391
no_temp_ph_no_advanced,xgb,log_kcat_value,val_mse,Validation MSE,5.165728569030762
no_temp_ph_no_advanced,xgb,log_km_value,val_mse,Validation MSE,3.687246322631836
no_temp_ph_no_advanced,xgb,log_kcat_value,val_mae,Validation MAE,1.7019997239112854
no_temp_ph_no_advanced,xgb,log_km_value,val_mae,Validation MAE,1.4191709160804749
no_temp_ph_no_advanced,xgb,log_kcat_value,val_r2,Validation R²,0.5289044678211212
no_temp_ph_no_advanced,xgb,log_km_value,val_r2,Validation R²,0.59908327460289
no_temp_ph_no_advanced,nn,log_kcat_value,val_mse,Validation MSE,7.407520532608032
no_temp_ph_no_advanced,nn,log_km_value,val_mse,Validation MSE,5.509087085723877
no_temp_ph_no_advanced,nn,log_kcat_value,val_mae,Validation MAE,2.0673751831054688
no_temp_ph_no_advanced,nn,log_km_value,val_mae,Validation MAE,1.776647925376892
no_temp_ph_no_advanced,nn,log_kcat_value,val_r2,Validation R²,0.3244597315788269
no_temp_ph_no_advanced,nn,log_km_value,val_r2,Validation R²,0.4009603559970855
yes_temp_ph_no_advanced,linear,log_kcat_value,val_mse,Validation MSE,10.48207488884674
yes_temp_ph_no_advanced,linear,log_km_value,val_mse,Validation MSE,7.892929407663725
yes_temp_ph_no_advanced,linear,log_kcat_value,val_mae,Validation MAE,2.564571571703212
yes_temp_ph_no_advanced,linear,log_km_value,val_mae,Validation MAE,2.1899617343465048
yes_temp_ph_no_advanced,linear,log_kcat_value,val_r2,Validation R²,0.0440726398560658
yes_temp_ph_no_advanced,linear,log_km_value,val_r2,Validation R²,0.1417509883485386
yes_temp_ph_no_advanced,xgb,log_kcat_value,val_mse,Validation MSE,4.984367609024048
yes_temp_ph_no_advanced,xgb,log_km_value,val_mse,Validation MSE,3.625297069549561
yes_temp_ph_no_advanced,xgb,log_kcat_value,val_mae,Validation MAE,1.6671074032783508
yes_temp_ph_no_advanced,xgb,log_km_value,val_mae,Validation MAE,1.4092910289764404
yes_temp_ph_no_advanced,xgb,log_kcat_value,val_r2,Validation R²,0.5454440116882324
yes_temp_ph_no_advanced,xgb,log_km_value,val_r2,Validation R²,0.6058073341846466
yes_temp_ph_no_advanced,nn,log_kcat_value,val_mse,Validation MSE,7.410960945569018
yes_temp_ph_no_advanced,nn,log_km_value,val_mse,Validation MSE,5.479482465689049
yes_temp_ph_no_advanced,nn,log_kcat_value,val_mae,Validation MAE,2.0740851736937045
yes_temp_ph_no_advanced,nn,log_km_value,val_mae,Validation MAE,1.7719201272982597
yes_temp_ph_no_advanced,nn,log_kcat_value,val_r2,Validation R²,0.3241449193508052
yes_temp_ph_no_advanced,nn,log_km_value,val_r2,Validation R²,0.4041663025778414
yes_temp_ph_yes_advanced,linear,log_kcat_value,val_mse,Validation MSE,11.182035051354251
yes_temp_ph_yes_advanced,linear,log_km_value,val_mse,Validation MSE,8.867498994095403
yes_temp_ph_yes_advanced,linear,log_kcat_value,val_mae,Validation MAE,2.2848746689146133
yes_temp_ph_yes_advanced,linear,log_km_value,val_mae,Validation MAE,1.9107196259087709
yes_temp_ph_yes_advanced,linear,log_kcat_value,val_r2,Validation R²,-0.0197528087463041
yes_temp_ph_yes_advanced,linear,log_km_value,val_r2,Validation R²,0.0361582874739373
yes_temp_ph_yes_advanced,xgb,log_kcat_value,val_mse,Validation MSE,4.267570018768311
yes_temp_ph_yes_advanced,xgb,log_km_value,val_mse,Validation MSE,3.289770603179932
yes_temp_ph_yes_advanced,xgb,log_kcat_value,val_mae,Validation MAE,1.5241127610206604
yes_temp_ph_yes_advanced,xgb,log_km_value,val_mae,Validation MAE,1.3363130688667295
yes_temp_ph_yes_advanced,xgb,log_kcat_value,val_r2,Validation R²,0.6108137369155884
yes_temp_ph_yes_advanced,xgb,log_km_value,val_r2,Validation R²,0.6422860026359558
yes_temp_ph_yes_advanced,nn,log_kcat_value,val_mse,Validation MSE,5.849361192787823
yes_temp_ph_yes_advanced,nn,log_km_value,val_mse,Validation MSE,4.13831895818713
yes_temp_ph_yes_advanced,nn,log_kcat_value,val_mae,Validation MAE,1.7533018372197244
yes_temp_ph_yes_advanced,nn,log_km_value,val_mae,Validation MAE,1.474792412415871
yes_temp_ph_yes_advanced,nn,log_kcat_value,val_r2,Validation R²,0.4665608126580384
yes_temp_ph_yes_advanced,nn,log_km_value,val_r2,Validation R²,0.5500264600698306
"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Preprocessing
exp_map = {
    'no_temp_ph_no_advanced': 'Baseline',
    'yes_temp_ph_no_advanced': '+Temp/pH',
    'yes_temp_ph_yes_advanced': '+Temp/pH\n+Advanced'
}
df['Experiment'] = df['experiment'].map(exp_map)

target_map = {
    'log_kcat_value': 'Log kcat',
    'log_km_value': 'Log KM'
}
df['Target'] = df['target'].map(target_map)
df['Model'] = df['model'].str.upper()

# 3. Setup Plot
sns.set_theme(style="whitegrid", font_scale=1.1)

# Setup 2 Rows x 5 Columns
fig, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True)

# Define Metrics Config (Columns)
metrics_config = [
    ('val_r2', 'R²'),
    ('val_pearson', 'Pearson (r)'),
    ('val_spearman', 'Spearman (ρ)'),
    ('val_mse', 'MSE'),
    ('val_mae', 'MAE')
]

# Define Targets (Rows)
targets = ['Log kcat', 'Log KM']

# Helper
def add_bar_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=0)

# --- PLOTTING LOOP ---
for row_idx, target_name in enumerate(targets):
    
    # Determine Colors
    if target_name == 'Log kcat':
        palette = "OrRd"
        row_color = '#B22222'
        row_title = "Log kcat"
    else:
        palette = "GnBu"
        row_color = '#00008B'
        row_title = "Log KM"

    for col_idx, (metric_key, metric_display) in enumerate(metrics_config):
        
        ax = axes[row_idx, col_idx]
        
        # Filter Data
        subset = df[(df['metric_type'] == metric_key) & (df['Target'] == target_name)]
        
        # Plot
        sns.barplot(
            data=subset, 
            x='Experiment', 
            y='metric_value', 
            hue='Model',
            palette=palette, 
            hue_order=['LINEAR', 'NN', 'XGB'], 
            ax=ax
        )
        
        # Labels & Styling
        add_bar_labels(ax)
        
        # Titles on top row only
        if row_idx == 0:
            ax.set_title(metric_display, fontsize=16, weight='bold', pad=15)
        else:
            ax.set_title('')

        # Y Label on first col only (shows Target Name)
        if col_idx == 0:
            ax.set_ylabel(f"{row_title}\nValue", fontsize=14, weight='bold', color=row_color)
        else:
            ax.set_ylabel('') 
            
        # X Axis
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Remove default legend inside the plot
        ax.get_legend().remove()
        
        # Zero line for R2
        if metric_key == 'val_r2':
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)

# --- CREATE DUAL LEGENDS ---

# 1. Get handles/labels from a Red plot (Row 0, Col 0)
handles_red, labels_red = axes[0, 0].get_legend_handles_labels()

# 2. Get handles/labels from a Blue plot (Row 1, Col 0)
handles_blue, labels_blue = axes[1, 0].get_legend_handles_labels()

# 3. Place Legend 1 (Red/kcat) at bottom left-center
fig.legend(handles_red, labels_red, 
           loc='upper center', 
           bbox_to_anchor=(0.35, 0.0), 
           ncol=3, 
           title="Log kcat Models", 
           title_fontsize=12, 
           frameon=False)

# 4. Place Legend 2 (Blue/KM) at bottom right-center
fig.legend(handles_blue, labels_blue, 
           loc='upper center', 
           bbox_to_anchor=(0.65, 0.0), 
           ncol=3, 
           title="Log KM Models", 
           title_fontsize=12, 
           frameon=False)

# Layout adjustments
plt.tight_layout()
plt.subplots_adjust(bottom=0.18) # Increase bottom margin to fit the two legends

# Save
plt.savefig('landscape_metrics_analysis_dual_legend.svg', format='svg', bbox_inches='tight')
print("Saved landscape_metrics_analysis_dual_legend.svg")