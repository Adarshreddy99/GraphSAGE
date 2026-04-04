import yaml
import os
from loguru import logger

def ingest_dataset():
    """
    Downloads the required dataset (Cora or OGBN-Arxiv) based on params.yaml.
    """
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    dataset_name = params["data"]["dataset"]
    save_dir = "data/raw/"
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Starting ingestion for dataset: {dataset_name}")
    
    if dataset_name.lower() == "cora":
        from torch_geometric.datasets import Planetoid
        logger.info("Downloading Cora from PyG...")
        dataset = Planetoid(root=save_dir, name='Cora')
        logger.info(f"Successfully downloaded Cora. Data: {dataset[0]}")
        
    elif dataset_name.lower() == "ogbn-arxiv":
        from ogb.nodeproppred import NodePropPredDataset
        logger.info("Downloading OGBN-Arxiv from OGB...")
        dataset = NodePropPredDataset(name='ogbn-arxiv', root=save_dir)
        logger.info(f"Successfully downloaded OGBN-Arxiv. Node count: {dataset[0][0]['num_nodes']}")
        
    else:
        logger.error(f"Unsupported dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset {dataset_name} in params.yaml")

if __name__ == "__main__":
    ingest_dataset()
