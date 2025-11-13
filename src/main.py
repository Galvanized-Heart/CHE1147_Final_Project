from config import PROCESSED_DATA_PATH
import pandas as pd
from dataset import download_and_clean_data
from features import process_data
from modeling.train import single_experiment

def main() -> None:
    download_and_clean_data()
    process_data()

    """
    FEATURE_TRANSFORMS = {
        "kcat_value": ("log", log),
        "km_value": ("log", log),
        "pH_value": ("linear", linear),
        "temperature_value": ("linear", linear),
        "mol_wt": ("log", log),
        "log_p": ("linear", linear),
        "tpsa": ("log1p", log1p),
        "num_h_donors": ("log1p", log1p),
        "num_h_acceptors": ("log1p", log1p),
        "num_rot_bonds": ("log1p", log1p),
        "seq_length": ("log", log),
        "seq_mol_wt": ("log", log),
        "pI": ("linear", linear),
        "aromaticity": ("linear", linear),
        "instability_index": ("linear", linear)
    }
    """

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    feature_columns = ["linear_pH_value", "linear_temperature_value",
                       "log_mol_wt", "linear_log_p", "log1p_tpsa",
                       "log1p_num_h_donors", "log1p_num_h_acceptors", "log1p_num_rot_bonds",
                       "log_seq_length", "log_seq_mol_wt", "linear_pI",
                       "linear_aromaticity", "linear_instability_index"]
    target_columns = ["log_kcat_value", "log_km_value"]
    X = df[feature_columns]
    y = df[target_columns]

    results_df = single_experiment(X, y)
    print(results_df)

if __name__ == "__main__":
    main()