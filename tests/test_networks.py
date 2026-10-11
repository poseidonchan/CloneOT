import torch

from cloneot.networks import ConvexH, ODEFlow, QuadraticH, squared_euclidean


def test_h_is_symmetric_convex_and_minimal_at_zero():
    torch.manual_seed(0)
    h = ConvexH(5, hidden=16, layers=2)
    with torch.no_grad():                      # move away from the near-quadratic initialization
        for p in h.parameters():
            p.add_(torch.randn_like(p))
    zero = torch.zeros(1, 5)
    d1, d2 = torch.randn(200, 5), torch.randn(200, 5)
    f = lambda d: h(d, torch.zeros_like(d))
    assert torch.allclose(f(d1), f(-d1), atol=1e-5)
    assert (f(d1) >= f(zero)-1e-5).all()
    assert (f((d1+d2)/2) <= (f(d1)+f(d2))/2+1e-4).all()


def test_pairwise_matches_paired_evaluation():
    torch.manual_seed(0)
    for h in (ConvexH(3, hidden=8, layers=2), QuadraticH()):
        z, w = torch.randn(7, 3), torch.randn(5, 3)
        pairs = h(z[:, None].expand(-1, 5, -1).reshape(-1, 3), w[None].expand(7, -1, -1).reshape(-1, 3)).reshape(7, 5)
        assert torch.allclose(h.pairwise(z, w), pairs, atol=1e-5)


def test_squared_euclidean():
    z, w = torch.randn(4, 2), torch.randn(3, 2)
    assert torch.allclose(squared_euclidean(z, w), .5*torch.cdist(z, w)**2, atol=1e-5)


def test_ode_flow_starts_at_identity_and_is_invertible():
    torch.manual_seed(0)
    flow = ODEFlow(4, hidden=16, layers=2, steps=8)
    x = torch.randn(50, 4)
    assert torch.allclose(flow(x), x)
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(.1*torch.randn_like(p))
    assert not torch.allclose(flow(x), x)
    assert torch.allclose(flow.inverse(flow(x)), x, atol=1e-3)
