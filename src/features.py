from typing import Dict, List
from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Import config and all your featurizer classes/functions
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, EsmModel

from config import INTERIM_DATA_PATH, PROCESSED_DATA_PATH, COLUMN_TRANSFORMS


class FeatureDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def get_molecular_descriptors(smiles: str) -> np.ndarray:
    """Computes basic molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    res = [np.nan] * 6
    if mol is not None:
        res = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
        ]

    return np.array(res, dtype=np.float32)


def get_sequence_descriptors(sequence: str) -> np.ndarray:
    """Computes basic protein sequence descriptors."""
    try:
        analysed_seq = ProteinAnalysis(sequence)
        res = [
            len(sequence),
            analysed_seq.molecular_weight(),
            analysed_seq.isoelectric_point(),
            analysed_seq.aromaticity(),
            analysed_seq.instability_index(),
        ]
    except Exception:
        res = [np.nan] * 5
    
    return np.array(res, dtype=np.float32)


def get_morgan_fingerprint(smiler: str, generator=None) -> np.ndarray:
    """Computes Morgan fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiler)
    res = [np.nan] * 2048
    if mol is not None:
        if generator is None:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        res = generator.GetFingerprint(mol)
    return np.array(res, dtype=np.float32)


def get_morgan_fingerprint_dict(smiles, generator=None) -> Dict[str, np.ndarray]:
    if generator is None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    res = {}
    for smile in smiles:
        res[smile] = get_morgan_fingerprint(smile, generator)
    
    return res 


@torch.no_grad()
def get_esm_embedding_dict(sequences: List[str], model: EsmModel, tokenizer: AutoTokenizer, device: str) -> Dict[str, np.ndarray]:
    # Get tokens
    inputs: Dict[str, torch.Tensor] = tokenizer(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=1022).to(device)
        
    # Get per-token embeddings from the last hidden state
    token_embeddings: torch.Tensor = model(**inputs).last_hidden_state
    attention_mask: torch.Tensor = inputs['attention_mask']
    
    # Aggregate features into a single vector
    expanded_mask: torch.Tensor = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings: torch.Tensor = torch.sum(token_embeddings * expanded_mask, 1)    
    sum_mask: torch.Tensor = torch.clamp(expanded_mask.sum(1), min=1e-9)
    pooled_embeddings: torch.Tensor = sum_embeddings / sum_mask

    return {seq: emb.cpu().numpy() for (seq, emb) in zip(sequences, pooled_embeddings)}


def process_data():
    if not INTERIM_DATA_PATH.exists():
        logger.error(f"Interim data file {INTERIM_DATA_PATH} does not exist. Please run data cleaning first.")
        raise FileNotFoundError(f"Interim data file {INTERIM_DATA_PATH} does not exist.")
    
    if PROCESSED_DATA_PATH.exists():
        logger.info(f"Processed data file {PROCESSED_DATA_PATH} already exists. Skipping processing.")
        return

    logger.info(f"Loading cleaned data from {INTERIM_DATA_PATH}")
    df = pd.read_parquet(INTERIM_DATA_PATH)

    # RDKit features
    logger.info("Engineering basic molecular descriptors...")
    desc_names = ['mol_wt', 'log_p', 'tpsa', 'num_h_donors', 'num_h_acceptors', 'num_rot_bonds']
    df[desc_names] = df['smiles'].apply(lambda s: pd.Series(get_molecular_descriptors(s), index=desc_names)) # .apply(get_molecular_descriptors).apply(pd.Series)

    # BioPython features
    logger.info("Engineering basic sequence descriptors...")
    seq_desc_names = ['seq_length', 'seq_mol_wt', 'pI', 'aromaticity', 'instability_index']
    df[seq_desc_names] = df['sequence'].apply(lambda s: pd.Series(get_sequence_descriptors(s), index=seq_desc_names))
    
    # ECFP embeddings
    logger.info("Engineering Morgan fingerprints for unique SMILES...")
    unique_smiles = df['smiles'].unique().tolist()
    morgan_features_dict = get_morgan_fingerprint_dict(unique_smiles)
    morgan_df = pd.DataFrame.from_dict(morgan_features_dict, orient='index').add_prefix('morgan_')
    df = df.merge(morgan_df, left_on='smiles', right_index=True, how='left')

    # ESM embeddings
    logger.info("Engineering ESM2 embeddings for unique sequences... (This can be slow)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ESM2 model to device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()

    unique_sequences = df['sequence'].unique().tolist()
    dataset = FeatureDataset(unique_sequences)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    esm_features_dict = {}
    for batch in tqdm(loader, desc="Processing ESM Batches"):
        batch_results = get_esm_embedding_dict(batch, model, tokenizer, device)
        esm_features_dict.update(batch_results)
    
    esm_df = pd.DataFrame.from_dict(esm_features_dict, orient='index').add_prefix('esm_')
    df = df.merge(esm_df, left_on='sequence', right_index=True, how='left')

    # Drop any last minute NaNs
    logger.info("Finalizing data: dropping NaNs and transforming targets...")
    df.dropna(inplace=True)

    # Reset indices
    df.reset_index(drop=True, inplace=True)

    # Transform values
    logger.info("Applying feature transformations to target variables...")
    for (col, (transform_name, transform)) in COLUMN_TRANSFORMS.items():
        df[f'{transform_name}_{col}'] = transform(df[col])

    logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    df.to_parquet(PROCESSED_DATA_PATH)
    logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    process_data()