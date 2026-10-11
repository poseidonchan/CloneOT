"""OT-SI baseline (Liu, Balsubramani & Zou, ICLR 2020): transport cost learned from subset correspondence by
differentiating through Sinkhorn.

Follows the paper's algorithm with a quadratic cost family and the clone information given to CloneOT:
cost c_theta(x, y) = |y - B x - delta|^2 / 2 in centred coordinates (OT-SI's degree-2 polynomial cost up to terms
the balanced plan ignores); violating pairs = pairs outside the allowed set A of the lineage problem.
  1. mimic learning (paper, Appendix C): from the Euclidean cost, `mimic_steps` gradient steps on the relative gap
     (OT_A - OT) / (mean C - OT) between the exact constrained and unconstrained (unregularised) OT costs;
  2. OT-SI: balanced entropic plan gamma(theta) by `sinkhorn_iters` log-domain Sinkhorn iterations from zero
     potentials at temperature eps (paper: lambda = 1e3, 200 iterations), differentiated through all iterations;
     loss sum_{(i, j) not in A} gamma_ij^2, gradient steps with the paper's step size (SGD, step 1, 100 epochs).
A new source cell x is coupled through the final Sinkhorn target potential, pi(y_j | x) ~ exp((g_j - c(x, y_j)) / eps),
and mapped to the barycentric projection E_pi[y | x].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import ot
import torch


@dataclass
class OTSIConfig:
    eps: float = 1e-3           # temperature as a fraction of the median cost (paper: 1 / lambda = 1e-3)
    sinkhorn_iters: int = 200
    epochs: int = 100
    lr: float = 1.0             # SGD step of the OT-SI stage (paper: 1)
    mimic_steps: int = 10
    mimic_lr: float = 1.0       # SGD step of the mimic stage (paper: 1)
    seed: int = 0


class OTSI:
    def __init__(self, config: OTSIConfig, device='cpu', dtype=torch.float32):
        self.config, self.device, self.dtype = config, torch.device(device), dtype

    @staticmethod
    def cost(z, y):
        return (.5*(z.square().sum(-1)[:, None]+y.square().sum(-1)[None])-z@y.T).clamp_min(0)

    def _C(self):
        return self.cost(self.X@self.B.T+self.delta, self.Y)

    def _sinkhorn(self, C, iters):
        N, M = C.shape
        g = torch.zeros(M, dtype=C.dtype, device=C.device)
        for _ in range(iters):
            f = -self.eps*(torch.logsumexp((g[None]-C)/self.eps, 1)-math.log(M))
            g = -self.eps*(torch.logsumexp((f[:, None]-C)/self.eps, 0)-math.log(N))
        return f, g

    def fit(self, source, target, source_labels=None, target_labels=None):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        X = torch.as_tensor(source, dtype=self.dtype, device=self.device)
        Y = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        self.mx, self.my = X.mean(0), Y.mean(0)
        self.X, self.Y = X-self.mx, Y-self.my
        N, M, d = len(X), len(Y), X.shape[1]
        self.B = torch.eye(d, dtype=self.dtype, device=self.device, requires_grad=True)
        self.delta = torch.zeros(d, dtype=self.dtype, device=self.device, requires_grad=True)
        self.eps = cfg.eps*float(self._C().detach().median())
        self.history = []
        if source_labels is not None:
            sl, tl = (torch.as_tensor(v).long().to(self.device) for v in (source_labels, target_labels))
            both = torch.tensor(sorted(set(sl[sl >= 0].tolist()) & set(tl[tl >= 0].tolist())), dtype=torch.long,
                                device=self.device)
            sl = torch.where(torch.isin(sl, both), sl, torch.full_like(sl, -1))
            tl = torch.where(torch.isin(tl, both), tl, torch.full_like(tl, -1))
            violating = sl[:, None] != tl[None]
            a, b = np.full(N, 1/N), np.full(M, 1/M)
            opt = torch.optim.SGD([self.B, self.delta], lr=cfg.mimic_lr)
            for step in range(cfg.mimic_steps):
                C = self._C()
                Cn = C.detach().double().cpu().numpy()
                free = torch.as_tensor(ot.emd(a, b, Cn), dtype=self.dtype, device=self.device)
                penal = np.where(violating.cpu().numpy(), Cn+10*Cn.max(), Cn)
                constrained = torch.as_tensor(ot.emd(a, b, penal), dtype=self.dtype, device=self.device)
                uncon = (free*C).sum()
                loss = ((constrained*C).sum()-uncon)/(C.mean()-uncon)
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.history.append(dict(stage='mimic', step=step, loss=float(loss)))
            self.mimic = dict(B=self.B.detach().cpu().tolist(), delta=self.delta.detach().cpu().tolist())
            opt = torch.optim.SGD([self.B, self.delta], lr=cfg.lr)
            for step in range(cfg.epochs):
                C = self._C()
                f, g = self._sinkhorn(C, cfg.sinkhorn_iters)
                plan = torch.exp((f[:, None]+g[None]-C)/self.eps)/(N*M)
                loss = plan[violating].square().sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 25 == 0 or step == cfg.epochs-1:
                    self.history.append(dict(stage='otsi', step=step, loss=float(loss),
                                             violating_mass=float(plan[violating].sum())))
        with torch.no_grad():
            _, self.g = self._sinkhorn(self._C(), cfg.sinkhorn_iters)
        return self

    def _z(self, source):
        x = torch.as_tensor(source, dtype=self.dtype, device=self.device)-self.mx
        return x@self.B.detach().T+self.delta.detach()

    @torch.no_grad()
    def conditional(self, source):
        return torch.softmax((self.g[None]-self.cost(self._z(source), self.Y))/self.eps, 1)

    @torch.no_grad()
    def predict_mean(self, source):
        return (self.conditional(source)@self.Y+self.my).cpu()
