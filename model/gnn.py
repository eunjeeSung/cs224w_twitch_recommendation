import math

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg

from model import embedding


class GNNEncoder(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, cfgs):
        super(GNNEncoder, self).__init__()
        self.cfgs = cfgs
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gated = pyg.nn.GatedGraphConv(self.hidden_size, num_layers=1)
        self.e2s = embedding.Embedding2HybridEmbedding(self.hidden_size)

    def forward(self, data):
        x, edge_index, batch = data.x - 1, data.edge_index, data.batch

        embedding = self.embedding(x).squeeze()
        hidden = self.gated(embedding, edge_index)
        session_embedding = F.relu(hidden)
        hybrid_embedding = self.e2s(session_embedding, self.embedding, batch)
        
        hybrid_embedding = hybrid_embedding.unsqueeze(1).expand(
            hybrid_embedding.shape[0],
            self.cfgs['seq_len'],
            hybrid_embedding.shape[1])  # [10, 64] -> [10, 16, 64]
        return hybrid_embedding
