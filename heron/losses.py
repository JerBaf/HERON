""" Self-supervised HERON objective (paper Sec. 3.3, App. 10.1.4).

    Each term is a mean over its own valid support (nodes, steps, graphs) so that the loss weights do not depend on
    the window length or the graph size.

"""
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# Custom Library
from .config import LossConfig
from .model import Outputs


# Constants
LOG_2PI = math.log(2 * math.pi)


# Loss Primitives

def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """ Compute the mean of the values over the entries selected by the mask. """
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def gaussian_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """ Compute the negative log-likelihood of a diagonal Gaussian, summed over the last dimension. """
    return 0.5 * (logvar + (target - mean).pow(2) * torch.exp(-logvar) + LOG_2PI).sum(-1)


def gaussian_kl(posterior_mu: Tensor, posterior_logvar: Tensor, prior_mu: Tensor, prior_logvar: Tensor) -> Tensor:
    """ Compute KL(q || p) between diagonal Gaussians, summed over the last dimension (Eq. 18). """
    return 0.5 * (prior_logvar - posterior_logvar
                  + (posterior_logvar.exp() + (posterior_mu - prior_mu).pow(2)) * torch.exp(-prior_logvar) - 1).sum(-1)


def categorical_kl(posterior_logits: Tensor, prior_logits: Tensor) -> Tensor:
    """ Compute KL(q || p) between categorical distributions given by logits (Eq. 17). """
    posterior_log_probs, prior_log_probs = F.log_softmax(posterior_logits, -1), F.log_softmax(prior_logits, -1)
    return (posterior_log_probs.exp() * (posterior_log_probs - prior_log_probs)).sum(-1)


# Structural Losses

def balance_kl(probs: Tensor, graph_mask: Tensor, target_usage: Tensor) -> Tensor:
    """ Compute the KL between the batch-averaged regime usage of each step and a target usage (Eq. 23). """
    weights = graph_mask.unsqueeze(-1).to(probs.dtype)
    usage = (probs * weights).sum(0) / weights.sum(0).clamp_min(1.0)  # [C, K]
    kl = (usage * (usage.clamp_min(1e-8).log() - target_usage.log())).sum(-1)
    return masked_mean(kl, graph_mask.any(0))


def specialization_loss(mu_experts: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """ Compute the expert decorrelation (App. Eq. 25), averaged over ordered pairs j != k and existing nodes. """
    # Squared projections between expert means
    gram = mu_experts @ mu_experts.transpose(-1, -2)                    # [..., K, K]
    squared_norms = gram.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)       # ||mu_k||^2, indexed by column k
    term = gram.pow(2) * squared_norms / (squared_norms + eps).pow(2)   # ||projection of mu_j on mu_k||^2
    # Average over pairs j != k
    K = gram.shape[-1]
    off_diagonal = ~torch.eye(K, dtype=torch.bool, device=gram.device)
    return masked_mean(term[..., off_diagonal].mean(-1), mask)


# Objective

class HERONLoss(nn.Module):
    """ Self-supervised HERON objective with prediction, KL and structural terms. """
    def __init__(self, config: LossConfig, num_regimes: int):
        """ Define the loss weights and the regime usage prior of the balance term. """
        super().__init__()
        # Store key parameters
        self.config = config
        # Balance target: uniform, or a slow EMA of the observed usage ("learned" prior)
        self.register_buffer("usage_prior", torch.full((num_regimes,), 1.0 / num_regimes))

    def forward(self, outputs: Outputs) -> tuple:
        """ Compute the total loss and the logged terms, updating the learned usage prior in training. """
        # Context length and masks
        config, C = self.config, outputs.posterior_regime_logits.shape[1]
        node_mask, graph_mask = outputs.node_mask, outputs.node_mask.any(-1)
        # Prediction terms
        posterior_nll = masked_mean(gaussian_nll(outputs.posterior_mean, outputs.posterior_logvar,
                                                 outputs.target[:, :C]),
                                    outputs.target_mask[:, :C])
        prior_nll = gaussian_nll(outputs.prior_mean, outputs.prior_logvar, outputs.target[:, :-1])
        prior_context_nll = masked_mean(prior_nll[:, :C], outputs.target_mask[:, :C])
        prior_horizon_nll = masked_mean(prior_nll[:, C:], outputs.target_mask[:, C:-1])  # only the prior exists here
        prediction = posterior_nll + config.prior_prediction_weight * prior_context_nll + prior_horizon_nll
        # Dynamics and representation KL
        posterior_logits, prior_logits = outputs.posterior_regime_logits, outputs.prior_regime_logits
        dynamics = (masked_mean(categorical_kl(posterior_logits.detach(), prior_logits), graph_mask)
                    + masked_mean(gaussian_kl(outputs.posterior_latent_mu.detach(),
                                              outputs.posterior_latent_logvar.detach(),
                                              outputs.prior_latent_mu, outputs.prior_latent_logvar), node_mask))
        representation = (masked_mean(categorical_kl(posterior_logits, prior_logits.detach()), graph_mask)
                          + masked_mean(gaussian_kl(outputs.posterior_latent_mu, outputs.posterior_latent_logvar,
                                                    outputs.prior_latent_mu.detach(),
                                                    outputs.prior_latent_logvar.detach()), node_mask))
        # Structural terms
        probs = posterior_logits.softmax(-1)  # no Gumbel noise (Eq. 21)
        balance = balance_kl(probs, graph_mask, self.usage_prior)
        entropy = masked_mean(-(probs * F.log_softmax(posterior_logits, -1)).sum(-1), graph_mask)
        specialization = specialization_loss(outputs.posterior_mu_experts, node_mask)
        structural = balance + config.entropy_weight * entropy + config.orthogonality_weight * specialization
        # Total loss
        total = (config.prediction_weight * prediction + config.dynamics_kl_weight * dynamics
                 + config.representation_kl_weight * representation + config.structural_weight * structural)
        # Logged values
        with torch.no_grad():
            usage = probs[graph_mask].mean(0)
            # Update the learned usage prior
            if self.training and config.balance_target == "learned":
                self.usage_prior.mul_(config.prior_momentum).add_((1 - config.prior_momentum) * usage)
            # Regime switches
            hard_regimes = posterior_logits.argmax(-1)
            pair_mask = graph_mask[:, 1:] & graph_mask[:, :-1]
            # Loss values
            log_dict = {
                "total": total.item(), "post_nll": posterior_nll.item(), "prior_context_nll": prior_context_nll.item(),
                "prior_horizon_nll": prior_horizon_nll.item(),
                "kl_dynamics": dynamics.item(), "kl_representation": representation.item(),
                "balance": balance.item(), "entropy": entropy.item(), "specialization": specialization.item(),
                "usage_min": usage.min().item(),
                "switch_rate": masked_mean((hard_regimes[:, 1:] != hard_regimes[:, :-1]).float(), pair_mask).item(),
            }
        return total, log_dict
