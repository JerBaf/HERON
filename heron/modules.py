""" Building blocks of HERON: attention pooling and one stochastic-state inference path (paper Eqs. 2-6). """
import torch
import torch.nn as nn
from typing import Optional
# Custom Library
from .config import ModelConfig


# Helper Functions

def create_mlp(layer_dims: list, activation_cls: type = nn.ReLU) -> nn.Sequential:
    """ Create an MLP with the given layer sizes and an activation between linear layers. """
    # Activations between linear layers
    layers = []
    for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(in_size, out_size))
        if i < len(layer_dims) - 2:
            layers.append(activation_cls())
    return nn.Sequential(*layers)


def bound_logvar(logvar: torch.Tensor, low: float = -6.0, high: float = 4.0) -> torch.Tensor:
    """ Smoothly bound a log-variance to [low, high] for numerical stability. """
    return low + (high - low) * torch.sigmoid(logvar)


# Graph Pooling

class AttentionPooler(nn.Module):
    """ Gated attention readout over the existing nodes of each graph. """
    def __init__(self, emb_dim: int):
        """ Create the tanh-gated scoring network. """
        super().__init__()
        self.gate = create_mlp([emb_dim, emb_dim, 1], activation_cls=nn.Tanh)

    def forward(self, node_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # [B, N, D], [B, N]
        """ Pool the embeddings of the existing nodes into one graph embedding. """
        scores = self.gate(node_embs).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, -1) * mask  # empty graphs pool to zero
        return (weights.unsqueeze(-1) * node_embs).sum(1)


# Inference Path

class InferencePath(nn.Module):
    """ Inference path of the posterior phi or the prior psi, with K regime experts (Eqs. 2-6). """
    def __init__(self, encoder: nn.Module, config: ModelConfig):
        """ Create the attention pooler, the regime head and the K regime experts around the encoder. """
        super().__init__()
        # Key sizes
        emb_dim, K = config.entity_embedding_dim, config.num_regimes
        # Relational encoder of heron/backbone.py, output size emb_dim
        self.encoder = encoder
        # Regime head on (h^G_t, emb^G_t)
        self.pool = AttentionPooler(emb_dim)
        self.regime = create_mlp([config.system_hidden_dim + emb_dim, config.regime_hidden_dim,
                                  config.regime_hidden_dim, K])
        # Regime experts on (h^i_t, emb^i_t)
        self.experts = nn.ModuleList(
            create_mlp([config.entity_hidden_dim + emb_dim, config.expert_hidden_dim, config.expert_hidden_dim,
                        2 * config.latent_dim])
            for _ in range(K))

    def forward(self, h: torch.Tensor, system_state: torch.Tensor, encoder_input: torch.Tensor, mask: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> tuple:
        """ Compute the regime logits and the Gaussian latent of every expert for one time step. """
        # Relational node embeddings
        node_embs = self.encoder(encoder_input, mask, edge_index, edge_attr)
        # Regime logits from (h^G_t, emb^G_t)
        regime_logits = self.regime(torch.cat([system_state, self.pool(node_embs, mask)], -1))
        # Expert latents from (h^i_t, emb^i_t)
        expert_input = torch.cat([h, node_embs], -1)
        mu_experts, logvar_experts = torch.stack([expert(expert_input) for expert in self.experts], -2).chunk(2, -1)
        return regime_logits, mu_experts, bound_logvar(logvar_experts)  # [B, K], [B, N, K, dz], [B, N, K, dz]
