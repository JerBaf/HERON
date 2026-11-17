""" Plug-in interface: HERON around an existing backbone (paper Sec. 3.2, "Integration to existing frameworks";
    App. 10.5.7).

    HERON adds a system-level regime pathway (graph-level GRU, regime inference, regime-gated experts, decoder,
    objective, autonomous rollout) around two swappable parts:

    - a `Backbone`, the entity-level transition h^i_t = f(h^i_{t-1}, z^i_{t-1}, E_t). Any latent graph-temporal model
      with a per-entity recurrent state fits, e.g. GraphSSM run in recurrent mode. It is driven by the entity latent
      z, never by observations: these enter only through the posterior encoder, so the rollout stays autonomous
      (Algorithm 1).
    - an `Encoder`, the relational encoder of the posterior and prior paths. The paper uses a GNN, but any encoder of
      a set of interacting entities works (footnote 2).

    The defaults, `GRUBackbone` and `GINEEncoder`, are standalone HERON. A config selects either part by short name
    or by import path, with keyword arguments:

        model:
          backbone: {class: examples.graph_gru:GraphGRU, hops: 1}
          encoder: {class: gine}

"""
import importlib
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv
from typing import Optional
# Custom Library
from .modules import create_mlp


# Backbones

class Backbone(nn.Module):
    """ Abstract class for entity-level transitions.

        Built as `cls(input_dim, state_dim, **kwargs)` with input_dim = latent_dim and state_dim = entity_hidden_dim.
        Subclasses implement:

        init_states(B, N, like): states of B graphs with N entity slots before the first step, as a tensor or a
            tuple of tensors (e.g. an SSM state); `like` is a tensor carrying the dtype and device.
        forward_one_time_step(states, z, edge_index, alive) -> (states, h): one transition. z [B, N, input_dim] is
            the previous entity latent, edge_index [2, E] the current graph over the B * N flattened slots (graph b
            owns slots b*N..b*N+N-1), alive [B, N] the entities that existed at the previous step; all others must
            restart from the initial state. h [B, N, state_dim] is the entity state HERON reads (encoders, experts,
            decoder).

    """
    def __init__(self, input_dim: int, state_dim: int):
        """ Store the latent input size and the entity state size. """
        super().__init__()
        self.input_dim, self.state_dim = input_dim, state_dim

    def init_states(self, B: int, N: int, like: torch.Tensor):
        """ Initialize the states of B graphs with N entity slots. """
        raise NotImplementedError

    def forward_one_time_step(self, states, z: torch.Tensor, edge_index: torch.Tensor, alive: torch.Tensor) -> tuple:
        """ Perform one entity-level transition and return the new states and h. """
        raise NotImplementedError


class GRUBackbone(Backbone):
    """ Entity-level GRU of standalone HERON, h^i_t = GRU^V(h^i_{t-1}, z^i_{t-1}); the encoders carry relations. """
    def __init__(self, input_dim: int, state_dim: int):
        """ Create the entity-level GRU cell. """
        super().__init__(input_dim, state_dim)
        self.gru = nn.GRUCell(input_dim, state_dim)

    def init_states(self, B: int, N: int, like: torch.Tensor) -> torch.Tensor:
        """ Initialize the entity states at zero. """
        return like.new_zeros(B, N, self.state_dim)

    def forward_one_time_step(self, h: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor,
                              alive: torch.Tensor) -> tuple:
        """ Perform one GRU transition, newborn entities restarting from the zero state. """
        B, N, _ = h.shape
        keep = alive.unsqueeze(-1).to(h.dtype)
        h = self.gru((z * keep).flatten(0, 1), (h * keep).flatten(0, 1)).view(B, N, -1)
        return h, h


# Relational Encoders

class Encoder(nn.Module):
    """ Abstract class for relational encoders.

        Built as `cls(input_dim, hidden_dim, output_dim, num_layers, edge_dim, **kwargs)`. Subclasses implement:

        forward(x, mask, edge_index, edge_attr) -> [B, N, output_dim]: x [B, N, input_dim], mask [B, N],
            edge_index [2, E], edge_attr [E, edge_dim] or None. The posterior path passes edge features
            (edge_dim > 0 when the data has them); the prior path never does.

    """


class GINEEncoder(Encoder):
    """ Stack of GINE layers (GIN when edges carry no features) producing node embeddings. """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, edge_dim: int = 0):
        """ Create the GINE (or GIN) layers. """
        super().__init__()
        # Store key parameters
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.edge_dim = edge_dim
        # Create GINE layers
        self.layers = nn.ModuleList(
            GINEConv(create_mlp([in_size, out_size, out_size]), edge_dim=edge_dim) if edge_dim
            else GINConv(create_mlp([in_size, out_size, out_size]))
            for in_size, out_size in zip(layer_dims[:-1], layer_dims[1:]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Compute the node embeddings over the flattened entity slots. """
        # Flatten the B graphs
        B, N, _ = x.shape
        x = x.reshape(B * N, -1)
        # Message passing, ReLU between layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr) if self.edge_dim else layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x.view(B, N, -1)


# Plug-in Creation

BUILTIN_PLUGINS = {"gru": GRUBackbone, "gine": GINEEncoder}


def create_plugin(plugin_dict: dict, *args) -> nn.Module:
    """ Create the plug-in {"class": short name or "package.module:Class", **kwargs} with the positional `args`. """
    # Copy to keep config intact
    plugin_dict = dict(plugin_dict)
    plugin_name = plugin_dict.pop("class")
    # Short name or import path
    if plugin_name in BUILTIN_PLUGINS:
        plugin_cls = BUILTIN_PLUGINS[plugin_name]
    else:
        module_name, _, class_name = plugin_name.partition(":")
        plugin_cls = getattr(importlib.import_module(module_name), class_name)
    return plugin_cls(*args, **plugin_dict)
