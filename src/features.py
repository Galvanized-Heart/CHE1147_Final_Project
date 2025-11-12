from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Import config and all your featurizer classes/functions
from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, EsmModel



def get_molecular_descriptors(smiles):
    # ... (paste your code) ...
    pass

def get_sequence_descriptors(sequence):
    # ... (paste your code) ...
    pass

class MorganFeaturizer:
    # ... (paste your MorganFeaturizer, but make it return a numpy array) ...
    def featurize(self, smiles_list: list) -> dict:
        # small change to return numpy array for easier df creation
        results = {}
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp_bv = self.generator.GetFingerprint(mol)
                fp_array = np.array(fp_bv, dtype=np.float32)
                results[smiles] = fp_array
        return results

class ESMFeaturizer:
    # ... (paste your ESMFeaturizer with the mean-pooling logic from my last answer) ...
    pass


app = typer.Typer()

@app.command()
def main(
    input_path: Path = typer.Option(INTERIM_DATA_DIR / "cleaned_data.parquet", help="Path to the cleaned interim data."),
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory to save the processed, split datasets."),
):
    """
    Loads cleaned data, engineers basic and advanced (Morgan, ESM) features,
    applies target transformations, splits the data into train/val/test sets,
    and saves them to the processed directory.
    """
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_parquet(input_path)

    logger.info("Engineering basic molecular and sequence descriptors...")
    desc_names = ['mol_wt', 'log_p', 'tpsa', 'num_h_donors', 'num_h_acceptors', 'num_rot_bonds']
    df[desc_names] = df['smiles'].apply(lambda s: pd.Series(get_molecular_descriptors(s), index=desc_names))
    seq_desc_names = ['seq_length', 'seq_mol_wt', 'pI', 'aromaticity', 'instability_index']
    df[seq_desc_names] = df['sequence'].apply(lambda s: pd.Series(get_sequence_descriptors(s), index=seq_desc_names))
    
    logger.info("Engineering Morgan fingerprints... (may be slow)")
    morgan_featurizer = MorganFeaturizer(radius=2, fp_size=2048)
    morgan_features = morgan_featurizer.featurize(df['smiles'].unique().tolist())
    morgan_df = pd.DataFrame.from_dict(morgan_features, orient='index').add_prefix('morgan_')
    df = df.merge(morgan_df, left_on='smiles', right_index=True, how='left')

    logger.info("Engineering ESM2 embeddings... (may be very slow, requires GPU for speed)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    esm_featurizer = ESMFeaturizer(model_name="facebook/esm2_t6_8M_UR50D", device=device)
    esm_features = esm_featurizer.featurize(df['sequence'].unique().tolist())
    esm_df = pd.DataFrame.from_dict(esm_features, orient='index').add_prefix('esm_')
    df = df.merge(esm_df, left_on='sequence', right_index=True, how='left')

    logger.info("Finalizing data: dropping NaNs and transforming targets...")
    df.dropna(inplace=True)
    df['log_kcat'] = np.log1p(df['kcat (s^{-1})'])
    df['log_km'] = np.log1p(df['km (M)'])

    # Define features and targets
    feature_cols = [c for c in df.columns if c.startswith(('morgan_', 'esm_', 'mol_', 'log_p', 'tpsa', 'num_', 'seq_', 'pI', 'aromaticity', 'instability_index', 'pH', 'temperature'))]
    X = df[feature_cols]
    y = df[['log_kcat', 'log_km']]
    
    logger.info(f"Final feature matrix shape: {X.shape}")

    logger.info("Splitting data into train, validation, and test sets (64/16/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    logger.info(f"Train set size: {len(X_train)}, Val set size: {len(X_val)}, Test set size: {len(X_test)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(output_dir / 'X_train.parquet')
    y_train.to_parquet(output_dir / 'y_train.parquet')
    X_val.to_parquet(output_dir / 'X_val.parquet')
    y_val.to_parquet(output_dir / 'y_val.parquet')
    X_test.to_parquet(output_dir / 'X_test.parquet')
    y_test.to_parquet(output_dir / 'y_test.parquet')

    logger.success(f"Feature engineering complete. Processed data saved to {output_dir}")

if __name__ == "__main__":
    app()