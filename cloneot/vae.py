"""Variational autoencoder for the cell-state representation.

CloneOT transports cells in the latent space of a VAE trained on the log-normalized expression of the source and
target cells together, and decodes its predictions with the VAE decoder. The architecture is that of W1OT: an encoder
with ReLU hidden layers, a Gaussian latent space, and a mirrored decoder with a non-negative (ReLU) output; the loss is
the mean squared reconstruction error plus `alpha` times the KL divergence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass
class VAEConfig:
    hidden: list[int] = field(default_factory=lambda: [256, 128])
    latent_dim: int = 50
    alpha: float = 1e-4              # weight of the KL divergence
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256            # half source and half target cells
    steps: int = 25000
    eval_every: int = 500
    validation_fraction: float = 0.15
    seed: int = 0


class VAE(nn.Module):
    def __init__(self, n_genes: int, config: Optional[VAEConfig] = None):
        super().__init__()
        self.config = config or VAEConfig()
        layers, width = [], n_genes
        for h in self.config.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(width, self.config.latent_dim)
        self.fc_logvar = nn.Linear(width, self.config.latent_dim)
        layers, width = [], self.config.latent_dim
        for h in reversed(self.config.hidden):
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers += [nn.Linear(width, n_genes), nn.ReLU()]
        self.decoder = nn.Sequential(*layers)
        self.n_genes = n_genes

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        mse = nn.functional.mse_loss(self.decode(z), x)
        kl = torch.mean(-.5*torch.sum(1+logvar-mu**2-logvar.exp(), 1))
        return mse+self.config.alpha*kl

    @torch.no_grad()
    def latent(self, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
        """Posterior means."""
        self.eval()
        return torch.cat([self.encode(c)[0] for c in x.split(batch)])

    @torch.no_grad()
    def reconstruct(self, z: torch.Tensor, batch: int = 4096) -> torch.Tensor:
        self.eval()
        return torch.cat([self.decode(c) for c in z.split(batch)])

    def fit(self, expression: torch.Tensor, domain: np.ndarray, verbose: bool = False) -> 'VAE':
        """Train on `expression` (cells x genes, log-normalized), with batches balanced between the two domains
        (`domain`: 0 for source and 1 for target cells). The checkpoint with the lowest held-out loss is kept."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = expression.device
        held = rng.random(len(domain)) < cfg.validation_fraction
        train = {d: torch.as_tensor(np.flatnonzero(~held & (domain == d)), device=device) for d in (0, 1)}
        valid = {d: torch.as_tensor(np.flatnonzero(held & (domain == d)), device=device) for d in (0, 1)}
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, state = float('inf'), None
        from .model import _progress
        progress = _progress(cfg.steps, verbose)
        for step in range(1, cfg.steps+1):
            self.train()
            pick = torch.cat([train[d][torch.randint(len(train[d]), (cfg.batch_size//2,), device=device)] for d in (0, 1)])
            loss = self.loss(expression[pick])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 10.)
            optimizer.step()
            if step % cfg.eval_every == 0 or step == cfg.steps:
                self.eval()
                with torch.no_grad():
                    score = float(np.mean([self.loss(expression[valid[d]]).item() for d in (0, 1) if len(valid[d])]))
                if score < best:
                    best, state = score, {k: v.detach().clone() for k, v in self.state_dict().items()}
                progress.set_postfix(validation=f'{score:.4f}')
            progress.update(1)
        progress.close()
        if state is not None:
            self.load_state_dict(state)
        return self.eval()

    def state(self) -> dict:
        return dict(config=asdict(self.config), n_genes=self.n_genes, weights=self.state_dict())

    @classmethod
    def from_state(cls, state: dict) -> 'VAE':
        model = cls(state['n_genes'], VAEConfig(**state['config']))
        model.load_state_dict(state['weights'])
        return model.eval()
