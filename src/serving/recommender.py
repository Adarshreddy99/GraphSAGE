import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from loguru import logger
from sentence_transformers import SentenceTransformer

# Add src to path to import GraphSAGE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
try:
    from graphsage import GraphSAGE
except ImportError:
    from model.graphsage import GraphSAGE

class Recommender:
    def __init__(self, config_path="params.yaml"):
        # 1. Load Parameters
        with open(config_path, "r") as f:
            self.params = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Recommender initializing on {self.device}...")

        # 2. Load Metadata
        self.df = pd.read_csv("data/processed/papers_metadata.csv")
        self.pyg_data = torch.load("data/processed/citation_graph.pt")
        
        # 3. Load Models
        self.text_model = SentenceTransformer(self.params["features"]["model"])
        
        model_params = self.params["model"]
        self.sage_model = GraphSAGE(
            input_dim=self.params["features"]["embedding_dim"],
            hidden_dim=model_params["hidden_dim"],
            output_dim=model_params["output_dim"],
            num_layers=model_params["num_layers"],
            aggregator=model_params["aggregator"],
            dropout=model_params["dropout"]
        )
        
        # Loading best weights
        model_path = "models/best_model.pt"
        if os.path.exists(model_path):
            self.sage_model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Loaded best GraphSAGE weights.")
        else:
            logger.warning("No best_model.pt found. Using untrained weights!")
            
        self.sage_model.to(self.device)
        self.sage_model.eval()

        # 4. Build/Load FAISS Index
        self.index_path = "models/faiss_hnsw.bin"
        self._initialize_index()

    def _initialize_index(self):
        """
        Builds the HNSW index from all papers in the graph.
        """
        logger.info("Generating final GraphSAGE embeddings for all 169k papers...")
        with torch.no_grad():
            x = self.pyg_data.x.to(self.device)
            edge_index = self.pyg_data.edge_index.to(self.device)
            self.all_embeddings = self.sage_model(x, edge_index).cpu().numpy()
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.all_embeddings)
        
        d = self.params["model"]["output_dim"]
        m = self.params["inference"]["hnsw_m"]
        
        logger.info(f"Building HNSW index (M={m})...")
        self.index = faiss.IndexHNSWFlat(d, m)
        self.index.hnsw.efConstruction = self.params["inference"]["hnsw_ef_construction"]
        self.index.add(self.all_embeddings)
        
        # Set search parameters
        self.index.hnsw.efSearch = self.params["inference"]["hnsw_ef_search"]
        logger.info("FAISS HNSW Index is ready.")

    def mmr_diversity(self, query_emb, candidate_indices, k=10, lambda_param=0.7):
        """
        Maximal Marginal Relevance (MMR) for result diversity.
        """
        if len(candidate_indices) <= k:
            return candidate_indices

        selected = [candidate_indices[0]]
        remaining = list(candidate_indices[1:])
        
        # Pre-fetch embeddings for all candidates
        candidate_embs = self.all_embeddings[candidate_indices]
        
        while len(selected) < k:
            scores = []
            for idx in remaining:
                # Find index in the original candidate list to get its embedding
                orig_pos = list(candidate_indices).index(idx)
                target_emb = candidate_embs[orig_pos]
                
                # Relevance: similarity to query
                relevance = np.dot(query_emb, target_emb)
                
                # Diversity: max similarity to any already selected item
                selected_embs = self.all_embeddings[selected]
                redundancy = np.max(np.dot(selected_embs, target_emb))
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                scores.append(mmr_score)
            
            # Select best score
            best_idx = remaining[np.argmax(scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)
            
        return selected

    def recommend(self, query_text, k=10, lambda_param=0.7):
        """
        Inductive Recommendation pipeline.
        """
        # 1. Encode query with MiniLM
        raw_emb = self.text_model.encode([query_text], convert_to_tensor=True).to(self.device)
        
        # 2. Find 5 Pseudo-neighbors using raw features (inductive step)
        raw_features = self.pyg_data.x.numpy().astype('float32')
        faiss.normalize_L2(raw_features)
        
        query_raw = raw_emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_raw)
        
        # Search top-5 pseudo-neighbors using a temporary flat index
        d_raw = raw_features.shape[1]
        temp_index = faiss.IndexFlatL2(d_raw)
        temp_index.add(raw_features)
        _, neighbor_indices = temp_index.search(query_raw, k=self.params["inference"]["pseudo_neighbors"])
        neighbor_indices = neighbor_indices[0]
        
        # 3. Create a pseudo-neighborhood GraphSAGE forward pass
        with torch.no_grad():
            neighbor_embs = torch.from_numpy(self.all_embeddings[neighbor_indices]).to(self.device).float()
            query_sage_emb = neighbor_embs.mean(dim=0)
            final_query_emb = F.normalize(query_sage_emb, p=2, dim=0).cpu().numpy().astype('float32')
        
        # 4. Search HNSW Index for top candidates
        top_k_candidates = self.params["inference"]["top_k_candidates"]
        # Reshape to (1, D) for FAISS
        distances, indices = self.index.search(final_query_emb.reshape(1, -1), k=top_k_candidates)
        indices = indices[0]
        
        # 5. Apply MMR Diversity
        selected_indices = self.mmr_diversity(final_query_emb, indices, k=k, lambda_param=lambda_param)
        
        # 6. Join with Metadata
        results = []
        for idx in selected_indices:
            row = self.df.iloc[idx]
            results.append({
                "paper_id": int(row["paper_id"]),
                "title": str(row["title"]),
                "year": int(row["year"]),
                "subject": str(row["subject"]),
                "score": float(np.dot(final_query_emb, self.all_embeddings[idx]))
            })
            
        return results

if __name__ == "__main__":
    rec = Recommender()
    test_query = "Deep learning for graph neural networks and node classification"
    res = rec.recommend(test_query, k=5)
    print("\n--- Test Recommendations ---")
    for r in res:
        print(f"[{r['score']:.4f}] {r['title']} ({r['year']})")
