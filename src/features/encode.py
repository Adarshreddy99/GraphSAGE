import os
import yaml
import torch
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def encode_text():
    """
    Loads MiniLM, encodes concatenated title + abstract for each paper, 
    and saves the embeddings tensor.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    model_name = params["features"]["model"]
    batch_size = params["features"]["encoding_batch_size"]
    
    logger.info(f"Loading metadata to extract text...")
    metadata_path = "data/processed/papers_metadata.csv"
    if not os.path.exists(metadata_path):
        logger.error("papers_metadata.csv missing. Run preprocess.py --step metadata first.")
        raise FileNotFoundError
        
    df = pd.read_csv(metadata_path)
    # Handle NaN gracefully (e.g. absent abstracts)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    
    # Concatenate title and abstract with a separator suitable for MiniLM
    text_list = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    
    logger.info(f"Loading SentenceTransformer model: {model_name}...")
    # Device setup: Use CPU as per requirements unless GPU forced, but project specified CPU.
    # We will let sentence_transformers auto-detect or force CPU to stick to spec.
    device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    logger.info(f"Beginning encoding of {len(text_list)} papers in batches of {batch_size}...")
    
    embeddings = model.encode(
        text_list, 
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    
    out_path = "data/processed/minilm_embeddings.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # We detach to make sure it's a raw CPU tensor without gradients tracked just in case
    embeddings = embeddings.cpu().detach()
    torch.save(embeddings, out_path)
    logger.info(f"Successfully saved {embeddings.shape} embeddings to {out_path}.")
    
    # Optional baseline evaluation logic could go here or in a separate script.
    # The README says "Compute MiniLM baseline immediately after encoding".
    # I'll create a separate baseline function here.
    return embeddings

if __name__ == "__main__":
    encode_text()
