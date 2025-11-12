from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

# Import config and all your featurizer classes/functions
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_PATH, PROCESSED_DATA_DIR
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, EsmModel



def batch_generator(data: list, batch_size: int):
    """Yields successive batches from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def get_molecular_descriptors(smiles: str) -> list:
    """Computes basic molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 6
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
    ]

def get_sequence_descriptors(sequence: str) -> list:
    """Computes basic protein sequence descriptors."""
    if not isinstance(sequence, str) or any(c in sequence for c in 'XUZB'):
        return [np.nan] * 5
    try:
        analysed_seq = ProteinAnalysis(sequence)
        return [
            len(sequence),
            analysed_seq.molecular_weight(),
            analysed_seq.isoelectric_point(),
            analysed_seq.aromaticity(),
            analysed_seq.instability_index(),
        ]
    except Exception:
        return [np.nan] * 5

def get_morgan_fingerprints(smiles):
    # Initialize ECFP generator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate embeddings
    results = {}
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp_bv = generator.GetFingerprint(mol)
            fp_array = np.array(fp_bv, dtype=np.float32)
            results[smile] = fp_array
    return results

@torch.no_grad()
def get_esm_embeddings(sequences, model, tokenizer, device):
    # Get tokens
    inputs = tokenizer(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=1022).to(device)
        
    # Get per-token embeddings from the last hidden state
    token_embeddings = model(**inputs).last_hidden_state
    attention_mask = inputs['attention_mask']
    
    # Aggregate features into a single vector
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * expanded_mask, 1)    
    sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
    pooled_embeddings = sum_embeddings / sum_mask

    return {seq: emb.cpu().numpy() for seq, emb in zip(sequences, pooled_embeddings)}

app = typer.Typer()

@app.command()
def main(
    input_path: Path = typer.Option(INTERIM_DATA_PATH, help="Path to cleaned data."),
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory to save processed datasets."),
):
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_parquet(input_path)

    # RDKit features
    logger.info("Engineering basic molecular descriptors...")
    desc_names = ['mol_wt', 'log_p', 'tpsa', 'num_h_donors', 'num_h_acceptors', 'num_rot_bonds']
    df[desc_names] = df['smiles'].apply(lambda s: pd.Series(get_molecular_descriptors(s), index=desc_names))

    # BioPython features
    logger.info("Engineering basic sequence descriptors...")
    seq_desc_names = ['seq_length', 'seq_mol_wt', 'pI', 'aromaticity', 'instability_index']
    df[seq_desc_names] = df['sequence'].apply(lambda s: pd.Series(get_sequence_descriptors(s), index=seq_desc_names))
    
    # ECFP embeddings
    logger.info("Engineering Morgan fingerprints for unique SMILES...")
    unique_smiles = df['smiles'].unique().tolist()
    morgan_features_dict = {}
    batch_size = 1024
    for batch in tqdm(batch_generator(unique_smiles, batch_size), total=len(unique_smiles)//batch_size + 1, desc="Processing Morgan Batches"):
        batch_results = get_morgan_fingerprints(batch)
        morgan_features_dict.update(batch_results)
    morgan_df = pd.DataFrame.from_dict(morgan_features_dict, orient='index').add_prefix('morgan_')
    df = df.merge(morgan_df, left_on='smiles', right_index=True, how='left')

    # ESM embeddings
    logger.info("Engineering ESM2 embeddings for unique sequences... (This can be slow)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ESM2 model to device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()

    unique_sequences = df['sequence'].unique().tolist()
    esm_features_dict = {}
    batch_size = 128
    for batch in tqdm(batch_generator(unique_sequences, batch_size), total=len(unique_sequences)//batch_size + 1, desc="Processing ESM Batches"):
        batch_results = get_esm_embeddings(batch, model, tokenizer, device)
        esm_features_dict.update(batch_results)
    esm_df = pd.DataFrame.from_dict(esm_features_dict, orient='index').add_prefix('esm_')
    df = df.merge(esm_df, left_on='sequence', right_index=True, how='left')

    # Drop any last minute NaNs
    logger.info("Finalizing data: dropping NaNs and transforming targets...")
    df.dropna(inplace=True)
    df['log_kcat'] = np.log1p(df['kcat (s^{-1})'])
    df['log_km'] = np.log1p(df['km (M)'])

    # Define features and targets
    feature_cols = [c for c in df.columns if c.startswith(('morgan_', 'esm_', 'mol_', 'log_p', 'tpsa', 'num_', 'seq_', 'pI', 'aromaticity', 'instability_index', 'pH', 'temperature'))]
    X = df[feature_cols]
    y = df[['log_kcat', 'log_km']]
    
    logger.info(f"Final feature matrix shape: {X.shape}")

    # Split data
    logger.info("Splitting data into train, validation, and test sets (64/16/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    logger.info(f"Train set size: {len(X_train)}, Val set size: {len(X_val)}, Test set size: {len(X_test)}")

    # Save data
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