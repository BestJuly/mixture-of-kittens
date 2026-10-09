"""Shared output-gate correctness through the torchrun/pytest CUDA fixture."""

import pytest
import torch
import torch.distributed as dist

from mok import functional, ops

from .utils import (
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_reference_bf16,
    run_shared_output_gate_reference,
    run_swiglu_reference,
)


GATED_RESULT_NAMES = (
    "output", "d_x", "d_router_weights",
    "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down",
    "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down",
    "d_w_shared_output_gate",
)


def _dense_inputs(context):
    rank, world_size, device = context
    inputs = generate_inputs(rank, device, 2 * world_size, 2, 2, 512, 256, 256)
    generator = torch.Generator(device=device).manual_seed(919 + rank)
    weight = torch.randn(1, 256, generator=generator, device=device, dtype=torch.bfloat16) / 16
    return inputs, weight


def _routed_weight_args(weights, precision):
    if precision == "bf16":
        return weights, weights
    gate, up, down = [ops.mxfp8_quantize(weight, True, True) for weight in weights]
    return (gate[:2], up[:2], down[:2]), (gate, up, down[2:])


def _bf16_product_rounding_witness(device, rows, hidden):
    # Both products are near +/-1. BF16 rounds the positive product down,
    # yielding exact cancellation. FP32 retains its 1/16384 residual.
    x = torch.zeros(rows, hidden, device=device, dtype=torch.bfloat16)
    x[:, :2] = 1
    shared = torch.zeros_like(x)
    shared[:, 0] = 1 + 1 / 128
    shared[:, 1] = 1
    dy = torch.zeros_like(x)
    dy[:, 0] = 1 + 1 / 128
    dy[:, 1] = -(1 + 1 / 64)
    weight = torch.zeros(1, hidden, device=device, dtype=torch.bfloat16)
    weight[0, 0], weight[0, 1] = 1, -1  # Z=0 and G=0.5, but Wg is nonzero.
    return x, shared, weight, dy


@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
def test_gate_wgrad_fp32_contribution_and_accumulation(context, main_grad_dtype):
    rows, hidden = 32, 64
    x = torch.ones(rows, hidden, device=context[2], dtype=torch.bfloat16)
    x[-1].fill_(1 / 64)
    main_grad = torch.full((1, hidden), -rows, device=context[2], dtype=main_grad_dtype)
    for scale in (1, 1 / rows):
        dz = torch.full((rows, 1), scale, device=context[2], dtype=torch.bfloat16)
        # The exact dot is representable in FP32, but not BF16.
        contribution = torch.full_like(main_grad, (rows - 1 + 1 / 64) * scale, dtype=torch.float32)
        assert not torch.equal(contribution, contribution.bfloat16().float())
        fresh = functional._shared_output_gate_wgrad(dz, x)
        assert fresh.dtype == torch.float32
        torch.testing.assert_close(fresh, contribution, rtol=0, atol=0)
        expected = (main_grad.float() + contribution).to(main_grad_dtype)
        if main_grad_dtype == torch.bfloat16:
            pre_rounded = (main_grad.float() + contribution.bfloat16().float()).bfloat16()
            assert not torch.equal(expected, pre_rounded)
        returned = functional._shared_output_gate_wgrad(dz, x, main_grad)
        assert returned is main_grad
        torch.testing.assert_close(returned, expected, rtol=0, atol=0)
    main_grad.zero_()
    returned = functional._shared_output_gate_wgrad(dz, x, main_grad)
    assert returned is main_grad
    torch.testing.assert_close(returned, contribution.to(main_grad_dtype), rtol=0, atol=0)


def _mok_schedule(context, inputs, macrobatch_size=4096):
    _, _, device = context
    x, experts, probs, *_ = inputs
    config = functional.MoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2,
        minibatch_size=256, macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device,
        num_local_tokens=x.shape[0], hidden_size=x.shape[1], topk=probs.shape[1],
    )
    schedule = functional.build_schedule(workspace, config, experts, num_local_experts=2)
    assert (schedule.num_tokens.item() > macrobatch_size) == (macrobatch_size == 512)
    return config, workspace, schedule


def test_mok_gate_backward_uses_bf16_product_rounding(context):
    """Exact cancellation distinguishes BF16-product from FP32-product dG."""
    inputs, _ = _dense_inputs(context)
    x, experts, probs, *rest = inputs
    weights = rest[:-1]
    x, shared, weight, dy = _bf16_product_rounding_witness(
        context[2], x.shape[0], x.shape[1]
    )
    for matrix in weights:
        matrix.zero_()
    # Only one shared intermediate channel is active. BF16(silu(16))=16,
    # so the down GEMM produces our exact S; routed experts remain zero.
    weights[0][0, 0] = 16
    weights[1][0, 0] = 1
    weights[2][0, 0] = (1 + 1 / 128) / 16
    weights[2][1, 0] = 1 / 16
    inputs = (x, experts, probs, *weights, dy)
    config, workspace, schedule = _mok_schedule(context, inputs)
    output, saved = functional.forward(
        config, workspace, schedule, x, probs, *weights,
        shared_output_gate_weight=weight,
    )
    torch.testing.assert_close(saved.shared_output, shared, rtol=0, atol=0)
    torch.testing.assert_close(output, shared * 0.5, rtol=0, atol=0)
    _, _, _, _, _, golden_dw = run_shared_output_gate_reference(x, shared, weight, dy)
    gradients = functional.backward(
        config, workspace, schedule, saved, dy, x, probs, *weights,
        shared_output_gate_weight=weight,
    )
    assert len(gradients) == 9 and gradients[-1].dtype == torch.float32
    torch.testing.assert_close(golden_dw, torch.zeros_like(golden_dw), rtol=0, atol=0)
    torch.testing.assert_close(gradients[-1].double(), golden_dw, rtol=0, atol=0)
    # FP32-product dG would yield dZ=1/65536 per token instead of zero.


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("main_grad_dtype", [None, torch.float32, torch.bfloat16])
def test_mok_gated_matches_reference(context, precision, main_grad_dtype):
    inputs, weight = _dense_inputs(context)
    weights = inputs[3:9]
    forward_weights, backward_weights = _routed_weight_args(weights[3:], precision)
    accumulate = main_grad_dtype is not None
    state = _mok_schedule(context, inputs, 512 if accumulate else 4096)
    weight.requires_grad_(True)
    inputs[0].requires_grad_(True)
    second = (inputs[0].detach().mul(-0.5).requires_grad_(), *inputs[1:-1], inputs[-1] * 2)
    main_grads = gate_main_grad = None
    if accumulate:
        main_grads = tuple(
            torch.zeros_like(weights[i], dtype=main_grad_dtype) for i in (0, 3, 1, 4, 2, 5)
        )
        gate_main_grad = torch.full_like(weight, 0.25, dtype=main_grad_dtype)
        # Returned MLP gradients are routed-first, unlike the main_grads ABI.
        returned_buffers = tuple(main_grads[i] for i in (1, 3, 5, 0, 2, 4)) + (gate_main_grad,)
        expected_buffers = [buffer.clone() for buffer in returned_buffers]

    for window in range(2 if accumulate else 1):
        if window:
            for buffer in (*returned_buffers, *expected_buffers):
                buffer.zero_()
        pending = []
        for batch in (inputs, second):
            x, _, probs = batch[:3]
            output, saved = functional.forward(
                *state, x, probs, *weights[:3], *forward_weights,
                shared_output_gate_weight=weight,
            )
            assert saved.shared_output.shape == x.shape
            assert saved.shared_output.dtype == saved.shared_output_gate.dtype == torch.bfloat16
            assert saved.shared_output_gate.shape == (x.shape[0], 1)
            assert not saved.shared_output_gate.requires_grad and saved.shared_output_gate.grad_fn is None
            gate = torch.sigmoid(torch.nn.functional.linear(x, weight))
            torch.testing.assert_close(saved.shared_output_gate, gate, rtol=0, atol=0)
            shared = run_swiglu_reference(x.detach() @ weights[0].T, x.detach() @ weights[1].T) @ weights[2].T
            check_correctness("shared_output", shared, saved.shared_output, BF16_TOLERANCE)
            snapshots = (saved.shared_output.clone(), saved.shared_output_gate.clone())
            pending.append((batch, output, saved, snapshots))

        # Both contexts must survive another forward on the same workspace.
        for batch, output, saved, snapshots in reversed(pending):
            torch.testing.assert_close(saved.shared_output, snapshots[0], rtol=0, atol=0)
            torch.testing.assert_close(saved.shared_output_gate, snapshots[1], rtol=0, atol=0)
            gradients = functional.backward(
                *state, saved, batch[-1], batch[0], batch[2], *weights[:3], *backward_weights,
                main_grads=main_grads, shared_output_gate_weight=weight,
                shared_output_gate_main_grad=gate_main_grad,
            )
            reference = list(run_reference_bf16(*batch, shared_output_gate_weight=weight))
            assert len(gradients) == 9 and gradients[-1].dtype == (main_grad_dtype or torch.float32)
            if accumulate:
                for actual, buffer, expected, contribution in zip(
                    gradients[2:], returned_buffers, expected_buffers, reference[3:], strict=True,
                ):
                    assert actual is buffer and actual.dtype == main_grad_dtype
                    expected.add_(contribution)
                reference[3:] = expected_buffers
            for index, (name, actual, golden) in enumerate(zip(GATED_RESULT_NAMES, (output, *gradients), reference, strict=True)):
                tolerance = MXFP8_TOLERANCE if precision == "mxfp8" and index < 6 else BF16_TOLERANCE
                check_correctness(name, golden, actual, tolerance, print_stats=context[0] == 0)
