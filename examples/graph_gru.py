""" A graph-recurrent backbone for HERON, and the template for plugging an existing architecture (heron/backbone.py).

    GraphGRU mixes each entity's state with its neighbours' before the GRU update, in the spirit of
    graph-convolutional recurrent networks: h_t = GRU([z_{t-1}, A h_{t-1}, ..., A^k h_{t-1}], h_{t-1}), with A the
    mean over neighbours. Select it in a config (with the repository root on PYTHONPATH):

        model:
          backbone: {class: examples.graph_gru:GraphGRU, hops: 1}

"""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
# Custom Library
from heron.backbone import Backbone


class GraphGRU(Backbone):
    """ Entity-level GRU whose update also reads the k-hop neighbourhood means of the previous states. """
    def __init__(self, input_dim: int, state_dim: int, hops: int = 1):
        """ Create the GRU cell over the latent and the `hops` neighbourhood means. """
        super().__init__(input_dim, state_dim)
        self.hops = hops
        self.gru = nn.GRUCell(input_dim + hops * state_dim, state_dim)

    def init_states(self, B: int, N: int, like: torch.Tensor) -> torch.Tensor:
        """ Initialize the entity states at zero. """
        return like.new_zeros(B, N, self.state_dim)

    def forward_one_time_step(self, h: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor,
                              alive: torch.Tensor) -> tuple:
        """ Perform one GRU transition on the latent and the neighbourhood means of the previous states. """
        # Reset newborn entities and flatten
        B, N, _ = h.shape
        keep = alive.unsqueeze(-1).to(h.dtype)          # newborn entities start from the zero state
        h, z = (h * keep).flatten(0, 1), (z * keep).flatten(0, 1)
        # Neighbourhood aggregation
        src, dst = edge_index
        all_hop_states, hop_states = [], h
        for _ in range(self.hops):                      # k-hop neighbourhood means of the previous state
            hop_states = scatter(hop_states[src], dst, 0, dim_size=B * N, reduce="mean")
            all_hop_states.append(hop_states)
        # GRU update
        h = self.gru(torch.cat([z, *all_hop_states], -1), h).view(B, N, -1)
        return h, h
