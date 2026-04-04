import torch
import numpy as np
import faiss
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_distances

def compute_recall_at_k(actual_citations, retrieved_indices, k=10):
    if not actual_citations: return 0.0
    top_k = set(retrieved_indices[:k])
    actual_set = set(actual_citations)
    hits = len(top_k.intersection(actual_set))
    return hits / len(actual_set)

def compute_ndcg_at_k(actual_citations, retrieved_indices, k=10):
    if not actual_citations: return 0.0
    y_true = np.zeros(k)
    y_score = np.linspace(1, 0.1, k)
    actual_set = set(actual_citations)
    for i, idx in enumerate(retrieved_indices[:k]):
        if idx in actual_set:
            y_true[i] = 1.0
    return ndcg_score([y_true], [y_score], k=k)

def compute_mrr(actual_citations, retrieved_indices, k=10):
    actual_set = set(actual_citations)
    for i, idx in enumerate(retrieved_indices[:k]):
        if idx in actual_set:
            return 1.0 / (i + 1)
    return 0.0

def compute_hit_rate(actual_citations, retrieved_indices, k=10):
    actual_set = set(actual_citations)
    for idx in retrieved_indices[:k]:
        if idx in actual_set:
            return 1.0
    return 0.0

def compute_ild(retrieved_embeddings):
    if len(retrieved_embeddings) < 2: return 0.0
    dists = cosine_distances(retrieved_embeddings)
    mask = np.triu(np.ones(dists.shape, dtype=bool), k=1)
    pairwise_dists = dists[mask]
    return np.mean(pairwise_dists)

def run_evaluation(model, pyg_data, params, split='test'):
    """
    Performs full batch evaluation using FAISS for fast retrieval.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Generating full graph embeddings for {split} evaluation...")
    with torch.no_grad():
        # ogbn-arxiv embeddings fit in RAM/VRAM
        x, edge_index = pyg_data.x.to(device), pyg_data.edge_index.to(device)
        embeddings = model(x, edge_index).cpu().numpy()
    
    # Build FAISS index for fast retrieval
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings) # Cosine similarity
    index.add(embeddings)
    
    # Identify target nodes for evaluation
    if split == 'test':
        target_idx = torch.where(pyg_data.test_mask)[0].numpy()
    elif split == 'val':
        target_idx = torch.where(pyg_data.val_mask)[0].numpy()
    else:
        target_idx = np.arange(pyg_data.num_nodes)
        
    # Citation adjacency for ground truth
    edges = pyg_data.edge_index.numpy()
    adj = {i: set() for i in range(pyg_data.num_nodes)}
    for u, v in zip(edges[0], edges[1]):
        adj[u].add(v)

    k = params["evaluation"]["top_k"][1] # K=10
    
    metrics = {"recall": [], "ndcg": [], "mrr": [], "hit_rate": [], "ild": []}
    
    logger.info(f"Evaluating {len(target_idx)} nodes...")
    # Search for top K+5 (to exclude self and ensure we have at least K)
    distances, indices = index.search(embeddings[target_idx], k=k+5)
    
    for i, node_id in enumerate(tqdm(target_idx, desc=f"Scoring {split}")):
        actual = adj[node_id]
        if not actual: continue
        
        # Exclude self-hit if present
        preds = [idx for idx in indices[i] if idx != node_id][:k]
        
        metrics["recall"].append(compute_recall_at_k(actual, preds, k))
        metrics["ndcg"].append(compute_ndcg_at_k(actual, preds, k))
        metrics["mrr"].append(compute_mrr(actual, preds, k))
        metrics["hit_rate"].append(compute_hit_rate(actual, preds, k))
        
        # Diversity (ILD) - sampled for speed
        if i % 500 == 0:
            metrics["ild"].append(compute_ild(embeddings[preds]))
            
    final_scores = {f"{split}_{m}": np.mean(v) for m, v in metrics.items() if v}
    return final_scores
