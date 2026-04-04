import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, aggregator, dropout):
        super().__init__()
        # PyG SAGEConv does the heavy lifting
        self.conv = SAGEConv(in_channels, out_channels, aggr=aggregator)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, aggregator='mean', dropout=0.3):
        """
        3-Layer GraphSAGE Architecture.
        Transforms node features through graph neighborhoods.
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Layer 1: Input to Hidden
        self.layers.append(GraphSAGEBlock(input_dim, hidden_dim, aggregator, dropout))
        
        # Intermediate layers (if any)
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGEBlock(hidden_dim, hidden_dim, aggregator, dropout))
            
        # Final layer: Hidden to Output
        self.layers.append(GraphSAGEBlock(hidden_dim, output_dim, aggregator, dropout))
        
        # Mapper for raw features (for contrastive learning on nodes outside current neighborhood)
        self.feat_map = nn.Linear(input_dim, output_dim)
        
        # Projection Head for InfoNCE training
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, edge_index):
        """
        Standard forward pass for inference. Returns the standard embedding.
        """
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
    
    def project(self, x):
        """
        Projection step used specifically during training loss computation.
        """
        return self.projection_head(x)
