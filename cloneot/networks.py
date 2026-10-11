"""Neural networks of CloneOT.

The transport cost is c(x, y) = h(phi(x) - psi(y)) with

* phi, psi: neural ODE flows (time-1 maps of dz/dt = v(z, t), integrated with RK4), the identity at initialization;
* h(d) = g(d) + g(-d) + alpha / 2 |d|^2: a symmetric, strictly convex function built from an input convex neural
  network g (Amos et al., 2017).

The dual potentials and the neural map are multilayer perceptrons with zero-initialized output layers.
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {'silu': nn.SiLU, 'relu': nn.ReLU, 'gelu': nn.GELU, 'softplus': nn.Softplus, 'tanh': nn.Tanh}


def mlp(inputs: int, outputs: int, hidden: int, layers: int, activation: str = 'silu') -> nn.Sequential:
    """Multilayer perceptron with `layers` hidden layers of width `hidden` and a zero-initialized output layer."""
    blocks, width = [], inputs
    for _ in range(layers):
        blocks += [nn.Linear(width, hidden), ACTIVATIONS[activation]()]
        width = hidden
    blocks.append(nn.Linear(width, outputs))
    with torch.no_grad():
        blocks[-1].weight.zero_()
        blocks[-1].bias.zero_()
    return nn.Sequential(*blocks)


def squared_euclidean(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matrix of 1/2 |z_i - y_j|^2."""
    return (.5*(z.square().sum(-1)[:, None]+y.square().sum(-1)[None])-z@y.T).clamp_min(0)


class ConvexNet(nn.Module):
    """Input convex neural network with biases: z_1 = s(W_0 d + b_0), z_{k+1} = s(U_k z_k + W_k d + b_k) and output
    u . z_K + w . d, where s is the softplus function and U_k, u >= 0 through a softplus reparameterization."""

    def __init__(self, d: int, hidden: int, layers: int, init: float = 1.):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(d, hidden) for _ in range(layers)])

        def free(shape, scale):   # a free parameter whose softplus starts near `scale`
            return nn.Parameter(torch.full(shape, float(torch.log(torch.expm1(torch.tensor(scale)))))+.1*torch.randn(shape))
        self.U = nn.ParameterList([free((hidden, hidden), 1./hidden) for _ in range(layers-1)])
        self.u = free((hidden,), init/hidden)
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.from_affine([W(d) for W in self.W], self.w(d).squeeze(-1))

    def from_affine(self, a: list[torch.Tensor], b: torch.Tensor) -> torch.Tensor:
        """Output given the affine terms a_k = W_k d + b_k and b = w . d (used to evaluate all pairs of cells)."""
        z = nn.functional.softplus(a[0])
        for U, ak in zip(self.U, a[1:]):
            z = nn.functional.softplus(z@nn.functional.softplus(U).T+ak)
        return z@nn.functional.softplus(self.u)+b


class ConvexH(nn.Module):
    """h(d) = g(d) + g(-d) + alpha / 2 |d|^2 with an input convex network g and a learned alpha > 0."""

    def __init__(self, d: int, hidden: int = 64, layers: int = 2, init: float = 1.):
        super().__init__()
        self.icnn = ConvexNet(d, hidden, layers, init)
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """h(x_i - y_i) for paired rows."""
        diff = x-y
        return self.icnn(diff)+self.icnn(-diff)+self.log_alpha.exp()*.5*diff.square().sum(-1)

    def pairwise(self, z: torch.Tensor, w: torch.Tensor, chunk: int = 2**24) -> torch.Tensor:
        """Matrix h(z_i - w_j). The affine terms W_k (z_i - w_j) are differences of per-cell terms, so the matrix is
        computed in chunks without forming the inputs of every pair."""
        net = self.icnn
        P, Q = [W(z)-W.bias for W in net.W], [W(w)-W.bias for W in net.W]
        pz, qw = net.w(z).squeeze(-1), net.w(w).squeeze(-1)
        quad = self.log_alpha.exp()*squared_euclidean(z, w)
        step = max(1, chunk//(len(w)*P[0].shape[1]*max(1, len(P))))
        rows = []
        for i in range(0, len(z), step):
            a = [Pk[i:i+step, None]-Qk[None] for Pk, Qk in zip(P, Q)]
            b = pz[i:i+step, None]-qw[None]
            rows.append(net.from_affine([ak+W.bias for ak, W in zip(a, net.W)], b)
                        + net.from_affine([-ak+W.bias for ak, W in zip(a, net.W)], -b))
        return torch.cat(rows)+quad


class QuadraticH(nn.Module):
    """h(d) = alpha / 2 |d|^2 with a learned alpha > 0 (the ablation without the input convex network)."""

    def __init__(self):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_alpha.exp()*.5*(x-y).square().sum(-1)

    def pairwise(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.log_alpha.exp()*squared_euclidean(z, w)


class ODEFlow(nn.Module):
    """Time-1 map of dz/dt = v(z, t), integrated with `steps` steps of the classical fourth-order Runge-Kutta method.
    The velocity field v is a multilayer perceptron of (z, t) with a zero-initialized output layer, so the flow starts
    as the identity."""

    def __init__(self, d: int, hidden: int = 64, layers: int = 2, activation: str = 'silu', steps: int = 4):
        super().__init__()
        self.v = mlp(d+1, d, hidden, layers, activation)
        self.steps = steps

    def _velocity(self, z: torch.Tensor, t: float) -> torch.Tensor:
        return self.v(torch.cat([z, torch.full_like(z[:, :1], t)], -1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = 1./self.steps
        for k in range(self.steps):
            t = k*h
            k1 = self._velocity(z, t)
            k2 = self._velocity(z+h/2*k1, t+h/2)
            k3 = self._velocity(z+h/2*k2, t+h/2)
            k4 = self._velocity(z+h*k3, t+h)
            z = z+h/6*(k1+2*k2+2*k3+k4)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Backward integration from t = 1 to t = 0 (the inverse of the time-1 map up to discretization error)."""
        h = 1./self.steps
        for k in reversed(range(self.steps)):
            t = (k+1)*h
            k1 = self._velocity(z, t)
            k2 = self._velocity(z-h/2*k1, t-h/2)
            k3 = self._velocity(z-h/2*k2, t-h/2)
            k4 = self._velocity(z-h*k3, t-h)
            z = z-h/6*(k1+2*k2+2*k3+k4)
        return z


class ClonePotential(nn.Module):
    """Potential of the lineage-constrained problem: a network of the transformed state and of a learned embedding of
    the block (labeled clone or the unlabeled cells) of each training source cell."""

    def __init__(self, inputs: int, blocks: torch.Tensor, dim: int, hidden: int, layers: int, activation: str = 'silu'):
        super().__init__()
        self.register_buffer('block', blocks)
        self.embed = nn.Embedding(int(blocks.max())+1, dim)
        nn.init.zeros_(self.embed.weight)
        self.net = mlp(inputs+dim, 1, hidden, layers, activation)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Potential at all training source cells, in their order."""
        return self.net(torch.cat([z, self.embed(self.block)], -1))
