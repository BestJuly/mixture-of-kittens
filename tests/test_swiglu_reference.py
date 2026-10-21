"""Pin MOK's inclusive clamp subgradient independently of torch.clamp backward."""

import pytest
import torch

from .utils import run_swiglu_reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("component", ["gate", "up"])
def test_clamped_swiglu_reference_boundary_gradient(context, dtype, component):
    device = context[2]
    limit = 4.0
    lo = torch.tensor(-limit, device=device, dtype=dtype)
    hi = -lo
    negative_inf = torch.full_like(lo, -float("inf"))
    positive_inf = -negative_inf
    # Include both exact boundaries and their nearest representable neighbors.
    values = torch.stack((
        2 * lo, torch.nextafter(lo, negative_inf), lo,
        torch.nextafter(lo, positive_inf), torch.zeros_like(lo),
        torch.nextafter(hi, negative_inf), hi,
        torch.nextafter(hi, positive_inf), 2 * hi,
    ))
    gate = (values if component == "gate" else torch.ones_like(values)).requires_grad_()
    up = (values if component == "up" else torch.ones_like(values)).requires_grad_()
    dy = torch.tensor([1, -2, 0.5, 2, -1, 0.25, -0.5, 4, 1], device=device, dtype=dtype)
    output = run_swiglu_reference(gate, up, limit)
    d_gate, d_up = torch.autograd.grad(output, (gate, up), dy)

    # torch.clamp is used only for forward values, never its backward rule.
    gate_value = gate.detach().float().clamp(max=limit)
    up_value = up.detach().float().clamp(min=-limit, max=limit)
    silu_value = torch.nn.functional.silu(gate_value)
    expected_output = (silu_value * up_value).to(dtype)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    # Hand-written masks pin the inclusive boundary contract, not a version probe.
    if component == "gate":
        gate_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        up_mask = [1] * 9
    else:
        gate_mask = [1] * 9
        up_mask = [0, 0, 1, 1, 1, 1, 1, 0, 0]
    sigmoid = torch.sigmoid(gate_value)
    dsilu = sigmoid * (1 + gate_value * (1 - sigmoid))
    expected_gate = (dy.float() * up_value * dsilu
                     * torch.tensor(gate_mask, device=device)).to(dtype)
    expected_up = (dy.float() * silu_value
                   * torch.tensor(up_mask, device=device)).to(dtype)
    tolerance = dict(rtol=2e-6, atol=1e-7) if dtype == torch.float32 else dict(rtol=0, atol=0)
    torch.testing.assert_close(d_gate, expected_gate, **tolerance)
    torch.testing.assert_close(d_up, expected_up, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unclamped_swiglu_reference_unchanged(context, dtype):
    gate = torch.tensor([-5, -4, 0, 4, 5], device=context[2], dtype=dtype, requires_grad=True)
    up = torch.tensor([-4, 2, 1, 4, 5], device=context[2], dtype=dtype, requires_grad=True)
    actual = run_swiglu_reference(gate, up)
    expected = torch.nn.functional.silu(gate) * up
    actual_grads = torch.autograd.grad(actual.sum(), (gate, up))
    expected_grads = torch.autograd.grad(expected.sum(), (gate, up))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
