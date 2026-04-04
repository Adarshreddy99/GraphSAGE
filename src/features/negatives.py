import os
import yaml
import torch
import faiss
import numpy as np
import pickle
from loguru import logger
from tqdm import tqdm

def mine_hard_negatives():
    """
    Finds textually similar papers that are NOT actual citations (hard negatives).
    
    1. Builds FAISS Flat IP index on MiniLM embeddings.
    2. Searches top-50 nearest neighbours for all training papers.
    3. Removes actual citations (from citation_graph.pt).
    4. Samples 15 remaining as hard negatives and 5 as random negatives.
    5. Saves a dictionary: { paper_id: {'hard': [15 ids], 'random': [5 ids]} }
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    num_hard = params["training"]["num_hard_negatives"]
    num_random = params["training"]["num_random_negatives"]
    total_samples = params["data"]["num_papers"]

    logger.info("Loading graph and embeddings...")
    graph_path = "data/processed/citation_graph.pt"
    if not os.path.exists(graph_path):
        logger.error("Graph missing. Run preprocess.py --step graph")
        raise FileNotFoundError
        
    pyg_data = torch.load(graph_path)
    # Get edge list as an array of row, col
    edges = pyg_data.edge_index.numpy()
    
    logger.info("Building citation adjacency lookup for fast access...")
    # OGB edges are directed usually (source -> target citation). Let's be safe and make it Undirected for negatives.
    citation_adj = {i: set() for i in range(total_samples)}
    for u, v in zip(edges[0], edges[1]):
        citation_adj[u].add(v)
        citation_adj[v].add(u) # Count symmetric just in case

    logger.info("Building FAISS Flat index...")
    # L2 normalize embeddings for cosine similarity using Inner Product
    x = pyg_data.x.numpy()
    faiss.normalize_L2(x)
    
    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)

    logger.info("Finding nearest textual neighbours...")
    # Query all nodes for top 50
    k_search = 50
    # Process in batches to avoid huge memory spike for distances
    batch_size = 10000
    hard_negatives_dict = {}
    
    # Identify training nodes to mine negatives for (or we can just mine for all to be safe)
    # Let's just mine for train nodes to save time, or all if preferred.
    # The README says: "For each training paper: find top-50... sample 15..."
    train_indices = torch.where(pyg_data.train_mask)[0].numpy()
    
    logger.info(f"Mining negatives for {len(train_indices)} training papers...")
    
    for start_idx in tqdm(range(0, len(train_indices), batch_size)):
        end_idx = min(start_idx + batch_size, len(train_indices))
        batch_nodes = train_indices[start_idx:end_idx]
        
        batch_embeddings = x[batch_nodes]
        distances, indices = index.search(batch_embeddings, k=k_search)
        
        for i, node_id in enumerate(batch_nodes):
            candidate_neighbours = indices[i]
            # Exclude self
            candidates = [c for c in candidate_neighbours if c != node_id]
            
            # Exclude actual citations
            actual_cites = citation_adj[node_id]
            valid_hards = [c for c in candidates if c not in actual_cites]
            
            # Sample exactly num_hard (or fewer if we ran out, though unlikely with 50)
            if len(valid_hards) >= num_hard:
                sampled_hards = np.random.choice(valid_hards, num_hard, replace=False).tolist()
            else:
                sampled_hards = valid_hards
                
            # Random negatives
            # We randomly select from the entire corpus, just ensuring they aren't the node itself
            rands = np.random.choice(total_samples, num_random + 5, replace=False)
            valid_rands = [r for r in rands if r != node_id and r not in actual_cites][:num_random]
            
            hard_negatives_dict[node_id] = {
                'hard': sampled_hards,
                'random': valid_rands
            }

    out_path = "data/processed/hard_negatives.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(hard_negatives_dict, f)
        
    logger.info(f"Successfully saved negative samples to {out_path}")

if __name__ == "__main__":
    mine_hard_negatives()
