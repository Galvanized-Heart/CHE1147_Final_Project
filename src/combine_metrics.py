import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 1. Load your parity data
df = pd.read_csv('parity_experiment_data.csv')

# 2. Calculate Metrics
new_metrics = []

# Group by Experiment, Model, and Target
grouped = df.groupby(['experiment', 'model', 'target'])

for (exp, model, target), group in grouped:
    y_true = group['y_true']
    y_pred = group['y_pred']
    
    # Pearson Correlation
    r_val, _ = pearsonr(y_true, y_pred)
    new_metrics.append({
        'experiment': exp,
        'model': model,
        'target': target,
        'metric_type': 'val_pearson',
        'metric_display_name': 'Validation Pearson',
        'metric_value': r_val
    })
    
    # Spearman Rank Correlation
    rho_val, _ = spearmanr(y_true, y_pred)
    new_metrics.append({
        'experiment': exp,
        'model': model,
        'target': target,
        'metric_type': 'val_spearman',
        'metric_display_name': 'Validation Spearman',
        'metric_value': rho_val
    })

# 3. Create Dataframe and Display
df_correlations = pd.DataFrame(new_metrics)

df2 = pd.read_csv("metrics_experiment_data.csv", index_col=False)
result_df = pd.concat([df_correlations, df2], ignore_index=True)
result_df.to_csv("combined_metrics.csv", index=False)