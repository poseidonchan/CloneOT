# Method

CloneOT learns the transport cost from lineage tracing and predicts the responses of new cells.

**Data.** Source cells $\{x_i\}$ are measured before the perturbation and target cells $\{y_j\}$ after it, represented
by the latent means of a VAE. For a fraction of the clones, lineage barcodes identify the source cells $S_k$ and the
target cells $Y_k$ of clone $k$, but not which source cell gave rise to which target cell.

**Neural cost.** The cost is a neural network,

$$
c_\theta(x,y)=h_\theta\big(\varphi_\theta(x)-\psi_\theta(y)\big),\qquad
h_\theta(d)=g_\theta(d)+g_\theta(-d)+\tfrac{\alpha}{2}\|d\|^2,
$$

where $\varphi_\theta$ and $\psi_\theta$ are neural ODE flows and $g_\theta$ is an input convex neural network. The
cost starts at the squared Euclidean cost and satisfies the twist condition, so it defines a unique transport map.

**Learning from lineage.** Under the cost, the entropic OT problem gives the coupling $\pi_\theta$ with optimal value
$\mathrm{OT}_\theta$. The same problem under the lineage constraint, in which a target cell of clone $k$ may only descend
from source cells of clone $k$, gives $\mathrm{OT}^A_\theta$. CloneOT minimizes the excess cost

$$
\frac{L(\theta)}{\varepsilon}=\frac{\mathrm{OT}^A_\theta-\mathrm{OT}_\theta}{\varepsilon}
=\min_{\pi\in\Pi^A}\mathrm{KL}(\pi\,\Vert\,\pi_\theta)\ \ge 0,
$$

which is zero when the coupling already respects the lineage and reduces to the classical inverse OT likelihood when
every clone is one pair of cells. Both problems are solved with neural semi-dual potentials whose $c$-transforms have a
closed form, and the cost follows the envelope gradient
$\nabla_\theta L=\mathbb{E}_{\pi^A_\theta}[\nabla_\theta c_\theta]-\mathbb{E}_{\pi_\theta}[\nabla_\theta c_\theta]$,
so no solver needs to be unrolled.

**Prediction.** A neural map $T(x)=x+F(x)$ is fitted to the barycentric projection $\mathbb{E}_{\pi_\theta}[y\mid x]$
of the learned coupling and predicts the response of a new cell from its expression alone.

See the paper and its supplement for the proofs, the implementation details and the settings.
