import os
import yaml
import gzip
import pandas as pd
import numpy as np
import torch
from loguru import logger
import argparse
from tqdm import tqdm
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data

def build_metadata(params):
    """
    Parses the raw title/abstract file and OGB metadata to create a unified papers_metadata.csv.
    """
    dataset_name = params["data"]["dataset"]
    num_papers = params["data"]["num_papers"]
    
    if dataset_name.lower() != "ogbn-arxiv":
        logger.info("Metadata generation is only configured for ogbn-arxiv.")
        return

    logger.info("Loading OGBN-Arxiv from local cache...")
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='data/raw/')
    graph, label = dataset[0]
    
    # graph['node_year'] shape is (N, 1), label shape is (N, 1)
    years = graph["node_year"].squeeze()
    subjects = label.squeeze()

    # Load titles and abstracts
    raw_titles_path = params["data"]["raw_titles_file"]
    logger.info(f"Loading raw titles from {raw_titles_path}...")
    
    # Load without headers since titleabs is known to omit them and have varying sizes
    # We must use quoting=3 (QUOTE_NONE) because abstracts contain random quotes that break pandas
    df_text = pd.read_csv(raw_titles_path, sep="\t", compression="gzip", header=None, names=["paper_id", "title", "abstract"], quoting=3)
    
    # Clean up any bad rows (NaN paper_id) and convert strictly to Int64 to avoid Windows 32-bit overflow for large MAG IDs (e.g. 3 billion)
    df_text['paper_id'] = pd.to_numeric(df_text['paper_id'], errors='coerce')
    df_text = df_text.dropna(subset=['paper_id'])
    df_text["paper_id"] = df_text["paper_id"].astype(np.int64)
    
    # Load mapping from OGB nodes to MAG paper_ids
    mapping_path = "data/raw/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"
    logger.info(f"Loading mapping from {mapping_path}...")
    mapping = pd.read_csv(mapping_path) # typically has headers "node idx", "paper id"
    # Ensure correct column names
    mapping.columns = ["node_idx", "paper_id"]
    
    # Merge titles with the mapping to perfectly align with OGB node indices
    df_text = pd.merge(mapping, df_text, on="paper_id", how="left")
    
    # Sort by node_idx to exactly match node indices 0 to N-1
    df_text = df_text.sort_values(by="node_idx").reset_index(drop=True)
    
    # Optional check:
    if len(df_text) != num_papers:
        logger.warning(f"Raw titles length ({len(df_text)}) does not match num_papers ({num_papers}).")

    # Attach Year and Subject from the graph dict to ensure perfect alignment
    df_text["year"] = years
    df_text["subject"] = subjects
    
    # We only keep what's needed for the final system
    # Abstract can be discarded from the final CSV to save space, but let's keep it for encoding if needed.
    # We'll save all fields, just in case.
    out_path = "data/processed/papers_metadata.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_text.to_csv(out_path, index=False)
    logger.info(f"Saved metadata to {out_path} with {len(df_text)} papers.")

def build_graph(params):
    """
    Builds the PyTorch Geometric citation_graph.pt object.
    Combines the OGB graph structure with the MiniLM embeddings.
    """
    dataset_name = params["data"]["dataset"]
    if dataset_name.lower() != "ogbn-arxiv":
        logger.info("Graph generation is only configured for ogbn-arxiv.")
        return

    logger.info("Loading OGBN-Arxiv graph structure...")
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='data/raw/')
    graph, label = dataset[0]
    
    # Edge index
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    
    # Labels
    y = torch.tensor(label.squeeze(), dtype=torch.long)
    
    # Split indices (train/val/test masks)
    split_idx = dataset.get_idx_split()
    
    num_nodes = graph["num_nodes"]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True

    # Load MiniLM Embeddings
    embeddings_path = "data/processed/minilm_embeddings.pt"
    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file {embeddings_path} missing! Run encode.py first.")
        raise FileNotFoundError(f"Missing {embeddings_path}")
        
    logger.info("Loading MiniLM embeddings...")
    x = torch.load(embeddings_path)
    
    # Construct PyG Data
    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    
    out_path = "data/processed/citation_graph.pt"
    torch.save(pyg_data, out_path)
    logger.info(f"Successfully compiled and saved PyG graph to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument("--step", type=str, required=True, choices=["metadata", "graph", "all"], 
                        help="Which preprocessing step to run.")
    args = parser.parse_args()
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    if args.step in ["metadata", "all"]:
        build_metadata(params)
        
    if args.step in ["graph", "all"]:
        build_graph(params)
