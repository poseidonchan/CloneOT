"""CloneOT: neural inverse optimal transport that learns the transport cost from lineage tracing.

Given unpaired source cells (before a perturbation) and target cells (after it), and the clone identities of a fraction
of the cells, CloneOT learns

* a neural transport cost c(x, y) = h(phi(x) - psi(y)) under which the optimal coupling keeps the cells of every
  labeled clone together, by minimizing the excess cost L = OT^A - OT of the lineage-constrained entropic OT problem
  over the unconstrained one;
* a neural map T(x) = x + F(x) that predicts the response of a new source cell from its state alone.

Both entropic OT problems are solved with neural semi-dual potentials whose c-transforms have a closed form, and the
cost follows the envelope gradient E_{pi^A}[grad c] - E_pi[grad c], so no solver is unrolled.

Example
-------
>>> from cloneot import CloneOT
>>> model = CloneOT().fit(source, target, source_clones, target_clones)   # -1 / None: cell without a clone label
>>> model.fit_map()
>>> prediction = model.transport(new_source)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .networks import ClonePotential, ConvexH, ODEFlow, QuadraticH, mlp, squared_euclidean


@dataclass
class CloneOTConfig:
    """Settings of CloneOT. The defaults are the settings of the paper, shared by all simulations and single-cell
    tasks."""
    eps: float = 0.1                 # entropic temperature, in units of the median m of 1/2 |y - x|^2
    # dual potentials and map
    hidden: int = 128                # width of the potentials and of the map
    layers: int = 3                  # hidden layers of the potentials and of the map
    activation: str = 'silu'
    clone_embed: int = 16            # dimension of the clone embedding of the constrained potential (0: none)
    lr_potential: float = 1e-3
    batch: int = 4096                # target cells per potential step (all target cells if fewer)
    # learned cost c(x, y) = h(phi(x) - psi(y))
    cost: str = 'neural'             # 'neural' (learned) or 'euclidean' (fixed 1/2 |x - y|^2, i.e. entropic W2 OT)
    h: str = 'icnn'                  # 'icnn': g(d) + g(-d) + alpha/2 |d|^2; 'quadratic': alpha/2 |d|^2
    flows: bool = True               # neural ODE flows phi and psi (False: identity maps)
    icnn_hidden: int = 64
    icnn_layers: int = 2
    icnn_init: float = 1.            # scale of the ICNN output weights at initialization
    flow_hidden: int = 64
    flow_layers: int = 2
    flow_steps: int = 4              # Runge-Kutta steps of the flows
    lr_cost: float = 1e-3            # learning rate of h
    lr_flow: float = 1e-4            # learning rate of the flows
    cost_beta2: float = 0.99         # Adam second-moment decay of the cost
    cost_batch: int = 1024           # target cells per cost gradient
    cost_samples: int = 8            # source cells sampled per target cell from each coupling
    # schedule
    warmup: int = 500                # potential steps before the cost moves
    steps: int = 2000                # cost updates
    potential_steps: int = 5         # steps of each potential per cost update
    polish: int = 2000               # steps of the free potential at the final cost
    # neural map
    map_lr: float = 3e-4
    map_steps: int = 3000
    map_batch: int = 256             # source cells per map step
    seed: int = 0


def _as_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=dtype, device=device)


def _unlabeled(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or (
        isinstance(value, (int, np.integer)) and value < 0) or value == ''


def encode_clones(source_clones, target_clones, n_source: int, n_target: int):
    """Integer clone labels (-1: unlabeled) for the source and target cells. Clones observed on one side only are
    treated as unlabeled."""
    if source_clones is None or target_clones is None:
        return np.full(n_source, -1), np.full(n_target, -1)
    source_clones, target_clones = np.asarray(source_clones, dtype=object), np.asarray(target_clones, dtype=object)
    if len(source_clones) != n_source or len(target_clones) != n_target:
        raise ValueError('one clone label per source and per target cell is required')
    if all(isinstance(v, (int, np.integer)) for v in np.concatenate([source_clones, target_clones])):
        return source_clones.astype(int), target_clones.astype(int)
    labels = sorted({str(v) for v in np.concatenate([source_clones, target_clones]) if not _unlabeled(v)})
    index = {v: k for k, v in enumerate(labels)}
    encode = lambda values: np.array([-1 if _unlabeled(v) else index[str(v)] for v in values])
    return encode(source_clones), encode(target_clones)


class CloneOT:
    """Neural inverse optimal transport from lineage tracing.

    Parameters
    ----------
    config : CloneOTConfig, optional
        Settings (defaults: the settings of the paper).
    device : str, optional
        Torch device ('cuda' if available, else 'cpu').
    **overrides
        Fields of CloneOTConfig to override, e.g. ``CloneOT(seed=1)``.
    """

    def __init__(self, config: Optional[CloneOTConfig] = None, device: Optional[str] = None, **overrides):
        config = config or CloneOTConfig()
        self.config = CloneOTConfig(**{**asdict(config), **overrides})
        if self.config.cost not in ('neural', 'euclidean') or self.config.h not in ('icnn', 'quadratic'):
            raise ValueError("cost must be 'neural' or 'euclidean' and h must be 'icnn' or 'quadratic'")
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = torch.float32
        self.h = self.phi = self.psi = None
        self.map = None
        self.C = None          # cost matrix of all training cells at the current cost
        self.history = []

    # ------------------------------------------------------------------------------------------------------- the cost
    @property
    def learned(self) -> bool:
        """Whether the cost is a neural network (otherwise the squared Euclidean cost)."""
        return self.h is not None

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.phi is None else self.phi(x)

    def _psi(self, y: torch.Tensor) -> torch.Tensor:
        return y if self.psi is None else self.psi(y)

    def _pair_cost(self, z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Cost matrix between transformed source cells z = phi(x) and centered target cells Y."""
        if self.h is None:
            return squared_euclidean(z, Y)
        return self.h.pairwise(z, self._psi(Y))

    def _refresh_cost(self):
        """Cost matrix of all training cells and the scale s of the potentials (median cost of random pairs)."""
        if self.h is not None:
            with torch.no_grad():
                self.C = self._pair_cost(self._phi(self.X), self.Y)
                n = min(len(self.sub_x), len(self.sub_y))
                self.unit = float(self.C[self.sub_x[:n], self.sub_y[:n]].median())/self.scale

    # ------------------------------------------------------------------------------------------- semi-dual problems
    def _potential(self, net: nn.Module, z: torch.Tensor) -> torch.Tensor:
        """Potential s * f_xi(z), in units of the current cost (the factors are multiplied in single precision)."""
        unit = torch.ones((), dtype=self.dtype, device=self.device)
        if self.h is not None:
            unit = unit*self.unit
        return unit*self.scale*net(z).squeeze(-1)

    def _semidual(self, fx: torch.Tensor, C: torch.Tensor, target_block: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Semi-dual sum_i omega_i f(x_i) + mean_j g_f(y_j) over the target cells of the columns of C. With
        `target_block`, the c-transform of y runs only over the source cells of its block (lineage constraint)."""
        logits = self.log_w[:, None]+(fx[:, None]-C)/self.eps
        if target_block is not None:
            logits = logits.masked_fill(self.source_block[:, None] != target_block[None], -torch.inf)
        g = -self.eps*torch.logsumexp(logits, 0)
        return (self.w*fx).sum()+g.mean()

    def _targets(self, generator: torch.Generator) -> torch.Tensor:
        M = len(self.Y)
        if self.config.batch >= M:
            return torch.arange(M, device=self.device)
        return torch.randint(M, (self.config.batch,), generator=generator).to(self.device)

    def _potential_step(self, net, optimizer, z, generator, lineage=False) -> float:
        idx = self._targets(generator)
        C = squared_euclidean(z, self.Y[idx]) if self.C is None else self.C[:, idx]
        value = self._semidual(self._potential(net, z), C, self.target_block[idx] if lineage else None)
        optimizer.zero_grad()
        (-value).backward()
        optimizer.step()
        return float(value.detach())

    def _excess_cost(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(D^A(f^A) - D(f) on the target cells idx, surrogate whose gradient is the gradient of the cost at fixed
        potentials). For a neural h, the expectations E_{pi^A}[grad c] - E_pi[grad c] are estimated with
        `cost_samples` source cells per target cell drawn from each conditional coupling; for a quadratic h, they are
        computed exactly."""
        z = self._phi(self.X)
        with torch.no_grad():
            fx, fAx = self._potential(self.f, z), self._potential(self.fA, z)
        if isinstance(self.h, QuadraticH):
            C = self._pair_cost(z, self.Y[idx])
            gap = self._semidual(fAx, C, self.target_block[idx])-self._semidual(fx, C)
            return gap, gap
        k = self.config.cost_samples
        with torch.no_grad():
            C = self.C[:, idx]
            gap = self._semidual(fAx, C, self.target_block[idx])-self._semidual(fx, C)
            free = (self.log_w[:, None]+(fx[:, None]-C)/self.eps).T
            lineage = (self.log_w[:, None]+(fAx[:, None]-C)/self.eps).masked_fill(
                self.source_block[:, None] != self.target_block[idx][None], -torch.inf).T
            i_free = torch.distributions.Categorical(logits=free).sample((k,))
            i_lineage = torch.distributions.Categorical(logits=lineage).sample((k,))
        y = self._psi(self.Y[idx])[None].expand(k, -1, -1).reshape(-1, self.Y.shape[1])
        cost = lambda i: self.h(z[i].reshape(-1, z.shape[1]), y).mean()
        return gap, cost(i_lineage)-cost(i_free)

    # -------------------------------------------------------------------------------------------------------- fitting
    def _prepare(self, source, target, source_clones, target_clones) -> torch.Generator:
        """Centered data, cost scale m, temperature and the source weights omega given by the clone sizes."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        generator = torch.Generator().manual_seed(cfg.seed)
        X, Y = _as_tensor(source, self.device), _as_tensor(target, self.device)
        self.mx, self.my = X.mean(0), Y.mean(0)
        self.X, self.Y = X-self.mx, Y-self.my
        sub_x = torch.randperm(len(X), generator=generator)[:1000]
        sub_y = torch.randperm(len(Y), generator=generator)[:1000]
        self.sub_x, self.sub_y = sub_x.to(self.device), sub_y.to(self.device)
        self.scale = float(squared_euclidean(self.X[self.sub_x], self.Y[self.sub_y]).median())
        self.eps = cfg.eps*self.scale
        self._blocks(*encode_clones(source_clones, target_clones, len(X), len(Y)))
        return generator

    def _blocks(self, source_labels: np.ndarray, target_labels: np.ndarray):
        """Blocks of the lineage problem (labeled clones with cells on both sides, then all other cells) and source
        weights omega that give every block its share of the target cells, split equally within the block."""
        sl, tl = torch.as_tensor(source_labels).long(), torch.as_tensor(target_labels).long()
        both = sorted(set(sl[sl >= 0].tolist()) & set(tl[tl >= 0].tolist()))
        index = {c: k for k, c in enumerate(both)}
        K = len(both)
        block = torch.tensor([index.get(c, K) for c in sl.tolist()], dtype=torch.long)
        self.target_block = torch.tensor([index.get(c, K) for c in tl.tolist()], dtype=torch.long, device=self.device)
        n_source = torch.bincount(block, minlength=K+1).to(self.dtype)
        mass = torch.bincount(self.target_block.cpu(), minlength=K+1).to(self.dtype)/len(self.Y)
        if mass[K] > 0 and n_source[K] == 0:
            raise ValueError('there are unlabeled target cells but no unlabeled source cells')
        share = torch.ones(len(block), dtype=self.dtype)
        total = torch.zeros(K+1, dtype=self.dtype).index_add(0, block, share)
        w = mass[block]*share/total[block]       # 0 for unlabeled sources if every target cell is labeled
        self.source_block = block.to(self.device)
        self.w = w.to(self.device)
        self.log_w = torch.log(self.w)
        self.n_clones = K

    def fit(self, source, target, source_clones: Optional[Sequence] = None, target_clones: Optional[Sequence] = None,
            verbose: bool = False) -> 'CloneOT':
        """Learn the transport cost from the labeled clones and solve the entropic OT problem under it.

        Parameters
        ----------
        source, target : array of shape (n_source, d) and (n_target, d)
            Cell states before and after the perturbation (e.g. VAE latent codes).
        source_clones, target_clones : sequence, optional
            Clone label of every source and target cell; -1, None, NaN or '' for cells without a label. Without
            labels, CloneOT is entropic optimal transport with the squared Euclidean cost.
        verbose : bool
            Show a progress bar.
        """
        cfg = self.config
        generator = self._prepare(source, target, source_clones, target_clones)
        d = self.X.shape[1]
        self.f = mlp(d, 1, cfg.hidden, cfg.layers, cfg.activation).to(self.device, self.dtype)
        self.fA = (ClonePotential(d, self.source_block, cfg.clone_embed, cfg.hidden, cfg.layers, cfg.activation)
                   if cfg.clone_embed else mlp(d, 1, cfg.hidden, cfg.layers, cfg.activation)).to(self.device, self.dtype)
        opt_f = torch.optim.Adam(self.f.parameters(), lr=cfg.lr_potential)
        opt_fA = torch.optim.Adam(self.fA.parameters(), lr=cfg.lr_potential)
        learn_cost = cfg.cost == 'neural' and self.n_clones > 0 and cfg.steps > 0
        if learn_cost:
            self.h = (ConvexH(d, cfg.icnn_hidden, cfg.icnn_layers, cfg.icnn_init) if cfg.h == 'icnn'
                      else QuadraticH()).to(self.device, self.dtype)
            if cfg.flows:
                self.phi = ODEFlow(d, cfg.flow_hidden, cfg.flow_layers, cfg.activation, cfg.flow_steps).to(self.device, self.dtype)
                self.psi = ODEFlow(d, cfg.flow_hidden, cfg.flow_layers, cfg.activation, cfg.flow_steps).to(self.device, self.dtype)
            flows = [q for net in (self.phi, self.psi) if net is not None for q in net.parameters()]
            groups = [dict(params=list(self.h.parameters()), lr=cfg.lr_cost), dict(params=flows, lr=cfg.lr_flow)]
            opt_cost = torch.optim.Adam([g for g in groups if g['params']], betas=(0.9, cfg.cost_beta2))
            self.cost_parameters = [q for g in groups for q in g['params']]

        total = cfg.warmup+(cfg.steps*(1+2*cfg.potential_steps) if learn_cost else 0)+cfg.polish
        progress = _progress(total, verbose)
        z = self._phi(self.X).detach()
        self._refresh_cost()
        for _ in range(cfg.warmup):
            self._potential_step(self.f, opt_f, z, generator)
            if learn_cost:
                self._potential_step(self.fA, opt_fA, z, generator, lineage=True)
            progress.update(1)
        for step in range(cfg.steps if learn_cost else 0):
            z = self._phi(self.X).detach()
            self._refresh_cost()
            for _ in range(cfg.potential_steps):
                free = self._potential_step(self.f, opt_f, z, generator)
                lineage = self._potential_step(self.fA, opt_fA, z, generator, lineage=True)
            if isinstance(self.h, QuadraticH):
                idx = self._targets(generator)
            else:
                idx = torch.randint(len(self.Y), (min(cfg.cost_batch, len(self.Y)),), generator=generator).to(self.device)
            gap, surrogate = self._excess_cost(idx)
            opt_cost.zero_grad()
            (surrogate/self.eps).backward()
            if all(q.grad is None or torch.isfinite(q.grad).all() for q in self.cost_parameters):
                opt_cost.step()
            if step % 50 == 0 or step == cfg.steps-1:
                self.history.append(dict(step=step, excess_cost=float(gap)/self.eps, semidual=free, semidual_lineage=lineage))
                progress.set_postfix(excess=f'{float(gap)/self.eps:.3f}')
            progress.update(1+2*cfg.potential_steps)
        z = self._phi(self.X).detach()
        self._refresh_cost()
        for _ in range(cfg.polish):
            self._potential_step(self.f, opt_f, z, generator)
            progress.update(1)
        progress.close()
        self._finish()
        self.C = None
        return self

    @torch.no_grad()
    def _finish(self):
        """Target potential g_f on the training target cells (the c-transform of the free potential)."""
        z = self._phi(self.X)
        fx = self._potential(self.f, z)
        self.g = torch.cat([-self.eps*torch.logsumexp(self.log_w[:, None]+(fx[:, None]-self._pair_cost(z, Yc))/self.eps, 0)
                            for Yc in self.Y.split(4096)])

    def fit_map(self, decoder: Optional[Callable] = None, evaluate: Optional[Callable] = None, eval_every: int = 500,
                verbose: bool = False) -> 'CloneOT':
        """Fit the neural map T(x) = x + F(x) to the barycentric projection E_pi[y | x] of the learned coupling.

        Parameters
        ----------
        decoder : callable, optional
            Maps cell states to expression (e.g. a VAE decoder). If given, F is fitted so that decoder(T(x)) matches
            the expected decoded descendant E_pi[decoder(y) | x].
        evaluate : callable, optional
            evaluate(transport) -> validation score (lower is better), called every `eval_every` steps with the current
            map; the best checkpoint is kept.
        """
        cfg = self.config
        d = self.X.shape[1]
        self.map = F = mlp(d, d, cfg.hidden, cfg.layers, cfg.activation).to(self.device, self.dtype)
        optimizer = torch.optim.Adam(F.parameters(), lr=cfg.map_lr)
        generator = torch.Generator().manual_seed(cfg.seed+1)
        with torch.no_grad():
            z_all = self._phi(self.X)
            values = self.Y+self.my if decoder is None else torch.cat([decoder(y) for y in (self.Y+self.my).split(2048)])
        self.map_history, best = [], (float('inf'), None)
        progress = _progress(cfg.map_steps, verbose)
        for step in range(1, cfg.map_steps+1):
            pick = torch.randint(len(z_all), (cfg.map_batch,), generator=generator).to(self.device)
            with torch.no_grad():
                target = self._conditional(z_all[pick])@values
            u = self.X[pick]
            out = u+F(u)+self.my
            loss = ((out if decoder is None else decoder(out))-target).square().sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if evaluate is not None and (step % eval_every == 0 or step == cfg.map_steps):
                F.eval()
                score = float(evaluate(self.transport))
                F.train()
                self.map_history.append(dict(step=step, loss=float(loss.detach()), score=score))
                if score < best[0]:
                    best = (score, {k: v.detach().clone() for k, v in F.state_dict().items()})
            progress.update(1)
        progress.close()
        if best[1] is not None:
            F.load_state_dict(best[1])
        self.map = F.eval()
        return self

    # -------------------------------------------------------------------------------------------------------- outputs
    def _conditional(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax((self.g[None]-self._pair_cost(z, self.Y))/self.eps, 1)

    def _source(self, source) -> torch.Tensor:
        return _as_tensor(source, self.device)-self.mx

    @torch.no_grad()
    def transport(self, source) -> np.ndarray:
        """Predicted response T(x) of every source cell (rows of `source`), from its state alone."""
        if self.map is None:
            raise RuntimeError('call fit_map() first')
        u = self._source(source)
        return torch.cat([(c+self.map(c)+self.my) for c in u.split(8192)]).cpu().numpy()

    @torch.no_grad()
    def conditional(self, source) -> np.ndarray:
        """Matrix pi(y_j | x_i) of the learned coupling over the training target cells."""
        return self._conditional(self._phi(self._source(source))).cpu().numpy()

    @torch.no_grad()
    def barycentric_projection(self, source, values=None) -> np.ndarray:
        """E_pi[v(y) | x] over the training target cells (v: the target states by default)."""
        values = self.Y+self.my if values is None else _as_tensor(values, self.device)
        z = self._phi(self._source(source))
        return torch.cat([self._conditional(c)@values for c in z.split(2048)]).cpu().numpy()

    @torch.no_grad()
    def cost(self, source, target) -> np.ndarray:
        """Learned cost matrix c(x_i, y_j) (the squared Euclidean cost if no cost was learned)."""
        z = self._phi(self._source(source))
        return self._pair_cost(z, _as_tensor(target, self.device)-self.my).cpu().numpy()

    # ------------------------------------------------------------------------------------------------- save and load
    def save(self, path) -> None:
        """Save the configuration, the networks and the training data needed by conditional() and cost()."""
        state = dict(config=asdict(self.config), mx=self.mx, my=self.my, X=self.X, Y=self.Y, w=self.w,
                     source_block=self.source_block, eps=self.eps, scale=self.scale, unit=getattr(self, 'unit', 1.),
                     g=self.g, f=self.f.state_dict(), map=None if self.map is None else self.map.state_dict(),
                     h=None if self.h is None else self.h.state_dict(),
                     phi=None if self.phi is None else self.phi.state_dict(),
                     psi=None if self.psi is None else self.psi.state_dict())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state.items()}, path)

    @classmethod
    def load(cls, path, device: Optional[str] = None) -> 'CloneOT':
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(CloneOTConfig(**state['config']), device=device)
        cfg, dev = model.config, model.device
        for key in ('mx', 'my', 'X', 'Y', 'w', 'source_block', 'g'):
            setattr(model, key, state[key].to(dev))
        model.log_w = torch.log(model.w)
        model.eps, model.scale, model.unit = state['eps'], state['scale'], state['unit']
        d = model.X.shape[1]
        model.f = mlp(d, 1, cfg.hidden, cfg.layers, cfg.activation).to(dev)
        model.f.load_state_dict(state['f'])
        if state['h'] is not None:
            model.h = (ConvexH(d, cfg.icnn_hidden, cfg.icnn_layers, cfg.icnn_init) if 'icnn.u' in state['h'] else QuadraticH()).to(dev)
            model.h.load_state_dict(state['h'])
        for name in ('phi', 'psi'):
            if state[name] is not None:
                net = ODEFlow(d, cfg.flow_hidden, cfg.flow_layers, cfg.activation, cfg.flow_steps).to(dev)
                net.load_state_dict(state[name])
                setattr(model, name, net)
        if state['map'] is not None:
            model.map = mlp(d, d, cfg.hidden, cfg.layers, cfg.activation).to(dev)
            model.map.load_state_dict(state['map'])
            model.map.eval()
        return model


class _NoProgress:
    def update(self, n=1):
        pass

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass


def _progress(total: int, verbose: bool):
    if not verbose:
        return _NoProgress()
    from tqdm.auto import tqdm
    return tqdm(total=total, desc='CloneOT')
