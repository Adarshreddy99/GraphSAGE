import os
import yaml
import torch
import torch.nn.functional as F
import pickle
import mlflow
import numpy as np
from loguru import logger
from torch_geometric.loader import NeighborLoader
from graphsage import GraphSAGE
from evaluate import run_evaluation
from tqdm import tqdm

def infonce_loss(anchor_emb, pos_emb, neg_embs, temperature=0.07):
    """
    Standard InfoNCE: -log( exp(sim(a,p)/t) / sum(exp(sim(a,n)/t)) )
    """
    # Normalize for cosine similarity
    anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)
    pos_emb = F.normalize(pos_emb, p=2, dim=-1)
    neg_embs = F.normalize(neg_embs, p=2, dim=-1)
    
    # Positive similarity (N, 1)
    pos_sim = torch.sum(anchor_emb * pos_emb, dim=-1) / temperature
    
    # Negative similarities (N, num_neg)
    neg_sims = torch.matmul(anchor_emb.unsqueeze(1), neg_embs.transpose(1, 2)).squeeze(1) / temperature
    
    # Logits: [pos_sim, neg0, neg1, ...] (N, 1 + num_neg)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor_emb.device)
    
    return F.cross_entropy(logits, labels)

import argparse
import dagshub
from dotenv import load_dotenv

def train_model(eval_only=False):
    load_dotenv() # Load from .env
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    # Connect to DagsHub if URI is provided in .env
    remote_uri = os.getenv("MLFLOW_TRACKING_URI")
    if remote_uri:
        logger.info(f"Connecting to DagsHub Remote: {remote_uri}")
        mlflow.set_tracking_uri(remote_uri)
        # dagshub.init automatically handles auth if user/pass are in env
        repo_owner = remote_uri.split('/')[-2]
        repo_name = remote_uri.split('/')[-1].replace('.mlflow', '')
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
    else:
        logger.info("No remote URI found in .env. Using local sqlite:///mlflow.db")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    logger.info("Loading graph, embeddings, and negatives...")
    pyg_data = torch.load("data/processed/citation_graph.pt")
    
    with open("data/processed/hard_negatives.pkl", "rb") as f:
        hard_negatives_dict = pickle.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model Setup
    model_params = params["model"]
    model = GraphSAGE(
        input_dim=params["features"]["embedding_dim"],
        hidden_dim=model_params["hidden_dim"],
        output_dim=model_params["output_dim"],
        num_layers=model_params["num_layers"],
        aggregator=model_params["aggregator"],
        dropout=model_params["dropout"]
    ).to(device)
    
    if not eval_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=params["training"]["learning_rate"])
        
        # Setup Data Pipeline
        train_idx = torch.where(pyg_data.train_mask)[0]
        loader = NeighborLoader(
            pyg_data,
            num_neighbors=model_params["neighbor_samples"],
            batch_size=params["training"]["batch_size"],
            input_nodes=train_idx,
            shuffle=True
        )
        
        # Efficient single-pass Adjacency Index for selecting local positives
        edges = pyg_data.edge_index.numpy()
        adj = {i: [] for i in range(pyg_data.num_nodes)}
        for u, v in zip(edges[0], edges[1]):
            adj[u].append(int(v))
        
        mlflow.set_experiment("GraphSAGE_Recommender")
        
        best_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting training loop...")
        for epoch in range(params["training"]["epochs"]):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{params['training']['epochs']}")
            for batch in progress_bar:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                embeddings = model(batch.x, batch.edge_index)
                proj = model.project(embeddings)
                num_anchors = batch.batch_size
                anchors_emb = proj[:num_anchors]
                anchors_list = batch.n_id[:num_anchors].cpu().numpy()
                
                batch_pos_embs = []
                batch_neg_embs = []
                valid_indices = []
                
                for i, anchor_id in enumerate(anchors_list):
                    possible_pos = adj.get(anchor_id, [])
                    if not possible_pos: continue
                    pos_id = np.random.choice(possible_pos)
                    
                    neg_info = hard_negatives_dict.get(anchor_id, {'hard': [], 'random': []})
                    neg_ids = list(neg_info['hard']) + list(neg_info['random'])
                    
                    total_target = params["training"]["num_hard_negatives"] + params["training"]["num_random_negatives"]
                    if len(neg_ids) < total_target:
                        needed = total_target - len(neg_ids)
                        extra_rands = np.random.choice(pyg_data.num_nodes, needed, replace=False).tolist()
                        neg_ids += extra_rands
                    
                    pos_emb = model.feat_map(pyg_data.x[pos_id].to(device).unsqueeze(0))
                    pos_emb = model.project(pos_emb).squeeze(0)
                    neg_embs = model.feat_map(pyg_data.x[neg_ids].to(device))
                    neg_embs = model.project(neg_embs)
                    
                    batch_pos_embs.append(pos_emb)
                    batch_neg_embs.append(neg_embs)
                    valid_indices.append(i)
                    
                if not valid_indices: continue
                
                anchor_final = anchors_emb[valid_indices]
                pos_final = torch.stack(batch_pos_embs)
                neg_final = torch.stack(batch_neg_embs)
                
                loss = infonce_loss(anchor_final, pos_final, neg_final, params["training"]["temperature"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(loader)
            logger.info(f"Epoch {epoch+1}/{params['training']['epochs']} | Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "models/best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= params["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered.")
                break
    
    # Final Evaluation & Logging (Always runs, especially in eval_only mode)
    logger.info("Starting final evaluation for DagsHub...")
    mlflow.set_experiment("GraphSAGE_Recommender")
    with mlflow.start_run(run_name="GraphSAGE_DagsHub_Push"):
        mlflow.log_params(params["training"])
        mlflow.log_params(params["model"])
        
        # Load best weights
        model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
        
        val_metrics = run_evaluation(model, pyg_data, params, split='val')
        test_metrics = run_evaluation(model, pyg_data, params, split='test')
        
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.pytorch.log_model(model, "model")
        
        if remote_uri:
            logger.info("Success! Results are now live on your DagsHub Dashboard.")
        else:
            logger.info("Success! Results are saved to your local mlflow.db (create a .env to push to DagsHub).")
        for k, v in test_metrics.items():
            logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation for logging")
    args = parser.parse_args()
    
    train_model(eval_only=args.eval_only)
