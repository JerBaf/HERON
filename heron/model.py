""" HERON: hierarchical regime-conditioned dynamics on graph sequences (paper Sec. 3, Algorithm 1).

    Steps are 0-indexed: frames 0..C-1 are the context, frames C..T-1 the horizon (T = C + H). At step t the model
    predicts the velocity Y_t = X_{t+1} - X_t of the dynamic node features. The entity transition and the
    relational encoders are plug-ins (heron/backbone.py); everything else here is the backbone-agnostic regime
    pathway.

"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
# Custom Library
from .backbone import create_plugin
from .config import ModelConfig
from .modules import InferencePath, bound_logvar, create_mlp


# Model Outputs

@dataclass
class Outputs:
    """ Per-step quantities used by the training objective (context steps unless noted). """
    posterior_mean: torch.Tensor           # [B, C, N, D] posterior velocity prediction
    posterior_logvar: torch.Tensor
    prior_mean: torch.Tensor               # [B, T-1, N, D] prior velocity prediction, context and horizon
    prior_logvar: torch.Tensor
    target: torch.Tensor                   # [B, T, N, D] normalized velocity targets
    target_mask: torch.Tensor              # [B, T, N]
    node_mask: torch.Tensor                # [B, C, N]
    posterior_regime_logits: torch.Tensor  # [B, C, K] posterior regime logits
    prior_regime_logits: torch.Tensor      # [B, C, K] prior regime logits
    posterior_latent_mu: torch.Tensor      # [B, C, N, dz] routed posterior entity latent
    posterior_latent_logvar: torch.Tensor
    prior_latent_mu: torch.Tensor          # routed prior entity latent
    prior_latent_logvar: torch.Tensor
    posterior_mu_experts: torch.Tensor     # [B, C, N, K, dz] every posterior expert mean


@dataclass
class Forecast:
    """ Forecast of the H horizon frames with the context and horizon regimes. """
    positions: torch.Tensor        # [B, H, N, D] absolute dynamic features of frames C..T-1
    velocity_mean: torch.Tensor    # [B, H, N, D] normalized velocity predictions
    velocity_logvar: torch.Tensor
    context_regimes: torch.Tensor  # [B, C, K] posterior regime probabilities q(r_t)
    horizon_regimes: torch.Tensor  # [B, H, K] regime that drove each forecast step


# HERON Model

class HERON(nn.Module):
    """ Hierarchical regime-conditioned dynamics model around a plug-in entity backbone. """
    def __init__(self, config: ModelConfig, feat_dim: int, dynamic_dims: list, context: int, edge_dim: int = 0):
        """ Create the inference paths, the transitions, the decoder and the normalization buffers. """
        super().__init__()
        # Store key parameters
        self.config, self.context = config, context
        self.meta_dict = dict(feat_dim=feat_dim, dynamic_dims=list(dynamic_dims), context=context, edge_dim=edge_dim)
        K, D = config.num_regimes, len(dynamic_dims)
        entity_hidden_dim, system_hidden_dim, latent_dim = (config.entity_hidden_dim, config.system_hidden_dim,
                                                            config.latent_dim)
        self.register_buffer("dyn_index", torch.tensor(list(dynamic_dims)), persistent=False)

        def create_encoder(encoder_input_dim: int, encoder_edge_dim: int) -> nn.Module:
            """ Create the relational encoder of one inference path. """
            return create_plugin(config.encoder, encoder_input_dim, config.encoder_hidden_dim,
                                 config.entity_embedding_dim, config.encoder_layers, encoder_edge_dim)

        # Posterior q_phi reads X_t, Delta X_t and edges
        self.posterior = InferencePath(create_encoder(entity_hidden_dim + feat_dim + D, edge_dim), config)
        # Prior p_psi reads entity state and topology only
        self.prior = InferencePath(create_encoder(entity_hidden_dim, 0), config)
        # Entity transition h^i_t = f(h^i_{t-1}, z^i_{t-1}, E_t)
        self.backbone = create_plugin(config.backbone, latent_dim, entity_hidden_dim)
        # System transition h^G_t = GRU^G(h^G_{t-1}, r_{t-1})
        self.system_gru = nn.GRUCell(K, system_hidden_dim)
        # Decoder
        self.decoder = create_mlp([system_hidden_dim + entity_hidden_dim + K + latent_dim, config.decoder_hidden_dim,
                                   config.decoder_hidden_dim, 2 * D])
        # Normalization buffers
        for name, size in (("x", feat_dim), ("v", D), ("e", edge_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_std", torch.ones(size))

    # Inputs

    def set_normalizer(self, stats_dict: dict):
        """ Copy the training statistics into the normalization buffers of the same name. """
        for key, value in stats_dict.items():
            getattr(self, key).copy_(value)

    def get_inputs_and_targets(self, batch: dict) -> tuple:
        """ Retrieve the normalized node inputs [X_t, Delta X_t], velocity targets Y_t and their validity mask. """
        # Normalized velocities
        x, mask = batch["x"], batch["mask"]
        dynamic_features = x[..., self.dyn_index]
        pair_mask = mask[:, 1:] & mask[:, :-1]
        velocity = ((dynamic_features[:, 1:] - dynamic_features[:, :-1] - self.v_mean) / self.v_std
                    * pair_mask.unsqueeze(-1))
        zero = velocity.new_zeros(velocity[:, :1].shape)
        # Node inputs
        node_input = torch.cat([(x - self.x_mean) / self.x_std * mask.unsqueeze(-1),
                                torch.cat([zero, velocity], 1)], -1)
        # Target mask
        target_mask = torch.cat([pair_mask, torch.zeros_like(mask[:, :1])], 1)
        target_mask[:, self.context:] &= mask[:, self.context - 1:self.context]  # horizon node set V* = V_C
        # Normalized edge features
        edge_attr = batch.get("edge_attr")
        if edge_attr is not None:
            edge_attr = [(edge_attr_t - self.e_mean) / self.e_std for edge_attr_t in edge_attr]
        return node_input, torch.cat([velocity, zero], 1), target_mask, edge_attr

    # Step Components

    def _compute_regime_weights(self, logits: torch.Tensor, tau: float, stochastic: bool) -> torch.Tensor:
        """ Compute the regime weights: softmax, straight-through Gumbel one-hot or argmax one-hot. """
        if self.config.routing == "soft":
            return logits.softmax(-1)
        if stochastic:
            return F.gumbel_softmax(logits, tau=tau, hard=True)  # one-hot forward, relaxed gradient (Eqs. 8-10)
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)

    @staticmethod
    def _gate_experts(r: torch.Tensor, mu_experts: torch.Tensor, logvar_experts: torch.Tensor, mask: torch.Tensor,
                      stochastic: bool) -> tuple:
        """ Combine the expert Gaussians with the regime weights r into the masked entity latent z. """
        weights = r[:, None, :, None]  # [B, 1, K, 1]
        mu, logvar = (weights * mu_experts).sum(-2), (weights * logvar_experts).sum(-2)  # [B, N, dz]
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if stochastic else mu
        return mu, logvar, z * mask.unsqueeze(-1)

    def _decode_velocity(self, system_state: torch.Tensor, h: torch.Tensor, r: torch.Tensor, z: torch.Tensor) -> tuple:
        """ Decode the velocity mean and bounded log-variance from (h^G_t, h^i_t, r_t, z^i_t). """
        N = h.shape[1]
        decoder_input = torch.cat([system_state.unsqueeze(1).expand(-1, N, -1), h,
                                   r.unsqueeze(1).expand(-1, N, -1), z], -1)
        mean, logvar = self.decoder(decoder_input).chunk(2, -1)
        return mean, bound_logvar(logvar)

    # Unrolling

    def _forward_window(self, batch: dict, tau: float, stochastic: bool, return_loss_terms: bool,
                        clamp_regime: Optional[int] = None) -> tuple:
        """ Compute the per-step outputs for one window: posterior on the context, prior rollout on the horizon. """
        # Inputs and targets
        node_input, target, target_mask, edge_attr = self.get_inputs_and_targets(batch)
        node_mask, edges = batch["mask"], batch["edge_index"]
        B, T, N = node_mask.shape
        C, K = self.context, self.config.num_regimes
        # State Initialization
        backbone_states = self.backbone.init_states(B, N, node_input)
        z = node_input.new_zeros(B, N, self.config.latent_dim)
        system_state = node_input.new_zeros(B, self.config.system_hidden_dim)
        r = node_input.new_full((B, K), 1.0 / K)  # uniform r_0
        alive = torch.zeros_like(node_mask[:, 0])
        step_outputs = {}

        def append_step_outputs(**values):
            """ Append the quantities of the current step to their running lists. """
            for key, value in values.items():
                step_outputs.setdefault(key, []).append(value)

        for t in range(T - 1):  # the last frame has no next-step target
            # Current topology
            topology_idx = min(t, C - 1)  # horizon keeps (V*, E*) = (V_C, E_C)
            node_mask_t = node_mask[:, topology_idx]
            # Transitions, new entities restart from init
            backbone_states, h = self.backbone.forward_one_time_step(backbone_states, z, edges[topology_idx], alive)
            system_state = self.system_gru(r, system_state)
            alive = node_mask_t
            # Prior path
            if return_loss_terms or t >= C:
                prior_regime_logits, prior_mu_experts, prior_logvar_experts = self.prior(
                    h, system_state, h, node_mask_t, edges[topology_idx])
                prior_regime = self._compute_regime_weights(prior_regime_logits, tau, stochastic)
                if clamp_regime is not None and t >= C:  # regime intervention on the rollout
                    prior_regime = F.one_hot(torch.full((B,), clamp_regime, device=h.device), K).to(h.dtype)
                prior_latent_mu, prior_latent_logvar, prior_latent = self._gate_experts(
                    prior_regime, prior_mu_experts, prior_logvar_experts, node_mask_t, stochastic)
                prior_mean, prior_logvar = self._decode_velocity(system_state, h, prior_regime, prior_latent)
            # Posterior path on the context
            if t < C:
                edge_attr_t = None if edge_attr is None else edge_attr[t]
                posterior_regime_logits, posterior_mu_experts, posterior_logvar_experts = self.posterior(
                    h, system_state, torch.cat([h, node_input[:, t]], -1), node_mask_t, edges[t], edge_attr_t)
                r = self._compute_regime_weights(posterior_regime_logits, tau, stochastic)
                posterior_latent_mu, posterior_latent_logvar, z = self._gate_experts(
                    r, posterior_mu_experts, posterior_logvar_experts, node_mask_t, stochastic)
                posterior_mean, posterior_logvar = self._decode_velocity(system_state, h, r, z)
                # Record step outputs
                append_step_outputs(posterior_mean=posterior_mean, posterior_logvar=posterior_logvar,
                                    posterior_regime_logits=posterior_regime_logits, regime=r)
                if return_loss_terms:
                    append_step_outputs(prior_mean=prior_mean, prior_logvar=prior_logvar,
                                        prior_regime_logits=prior_regime_logits,
                                        posterior_latent_mu=posterior_latent_mu,
                                        posterior_latent_logvar=posterior_latent_logvar,
                                        prior_latent_mu=prior_latent_mu, prior_latent_logvar=prior_latent_logvar,
                                        posterior_mu_experts=posterior_mu_experts)
            else:  # autonomous prior rollout
                r, z = prior_regime, prior_latent
                append_step_outputs(prior_mean=prior_mean, prior_logvar=prior_logvar, regime=r)
        return {k: torch.stack(v, 1) for k, v in step_outputs.items()}, target, target_mask

    def forward(self, batch: dict, tau: float = 1.0) -> Outputs:
        """ Compute the training quantities for one window with the posterior and prior paths. """
        step_outputs, target, target_mask = self._forward_window(batch, tau, stochastic=True, return_loss_terms=True)
        return Outputs(
            posterior_mean=step_outputs["posterior_mean"], posterior_logvar=step_outputs["posterior_logvar"],
            prior_mean=step_outputs["prior_mean"], prior_logvar=step_outputs["prior_logvar"],
            target=target, target_mask=target_mask, node_mask=batch["mask"][:, :self.context],
            posterior_regime_logits=step_outputs["posterior_regime_logits"],
            prior_regime_logits=step_outputs["prior_regime_logits"],
            posterior_latent_mu=step_outputs["posterior_latent_mu"],
            posterior_latent_logvar=step_outputs["posterior_latent_logvar"],
            prior_latent_mu=step_outputs["prior_latent_mu"], prior_latent_logvar=step_outputs["prior_latent_logvar"],
            posterior_mu_experts=step_outputs["posterior_mu_experts"])

    @torch.no_grad()
    def forecast(self, batch: dict, mode: str = "mean", tau: float = 1.0,
                 clamp_regime: Optional[int] = None) -> Forecast:
        """ Forecast the H horizon frames from the C context frames only (Algorithm 1). """
        # Unroll: argmax regimes and latent means, or draws if mode="sample"; clamp_regime=k forces regime k
        step_outputs, _, _ = self._forward_window(batch, tau, mode == "sample", return_loss_terms=False,
                                                  clamp_regime=clamp_regime)
        C = self.context
        # First step from the posterior at the last context step, the rest from the prior rollout
        empty = step_outputs["posterior_mean"][:, :0]  # H = 1: no prior step
        velocity_mean = torch.cat([step_outputs["posterior_mean"][:, -1:],
                                   step_outputs.get("prior_mean", empty)], 1)
        velocity_logvar = torch.cat([step_outputs["posterior_logvar"][:, -1:],
                                     step_outputs.get("prior_logvar", empty)], 1)
        # Integrate the velocities
        last_positions = batch["x"][:, C - 1][..., self.dyn_index]
        positions = last_positions.unsqueeze(1) + (velocity_mean * self.v_std + self.v_mean).cumsum(1)
        return Forecast(positions, velocity_mean, velocity_logvar, step_outputs["posterior_regime_logits"].softmax(-1),
                        step_outputs["regime"][:, C - 1:])
