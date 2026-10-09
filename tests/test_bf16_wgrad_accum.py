import os

import pytest
import torch

from mok import functional
from mok.ops import mxfp8_quantize
from .utils import (
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_reference_bf16,
)


@pytest.mark.parametrize("leave_one_expert_empty", [False, True])
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "routed_weight_storage",
    ["separate_legacy", "single_grouped_legacy", "single_grouped_native"],
)
def test_wgrad_accumulates_directly_into_main_grad(
    context: tuple[int, int, torch.device],
    leave_one_expert_empty: bool,
    main_grad_dtype: torch.dtype,
    precision: str,
    routed_weight_storage: str,
) -> None:
    rank, world_size, device = context
    single_grouped = routed_weight_storage != "separate_legacy"
    native_columnwise = routed_weight_storage == "single_grouped_native"
    # Exercise both the gate/up offset within FC1 and the expert stride between
    # adjacent slices in one single-grouped parameter.
    num_local_experts = 2
    num_experts = world_size * num_local_experts
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=int(os.getenv("MOK_TEST_MACROBATCH_SIZE", "256")),
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        torch.distributed.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    inputs = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    (
        x,
        topk_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    ) = inputs
    if precision == "mxfp8":
        if single_grouped:
            routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
            routed_fc1_quantized = mxfp8_quantize(routed_fc1, True, True)
            routed_down_quantized = mxfp8_quantize(w_routed_down, True, True)
            forward_routed_gate = routed_fc1_quantized[:2]
            forward_routed_up = forward_routed_gate
            forward_routed_down = routed_down_quantized[:2]
            if native_columnwise:
                native_fc1_columnwise = (
                    routed_fc1_quantized[2]
                    .transpose(-2, -1)
                    .contiguous()
                )
                backward_routed_gate = (
                    routed_fc1_quantized[0],
                    routed_fc1_quantized[1],
                    native_fc1_columnwise,
                    routed_fc1_quantized[3],
                    True,
                )
            else:
                backward_routed_gate = routed_fc1_quantized
            backward_routed_up = backward_routed_gate
            if native_columnwise:
                backward_routed_down = (
                    routed_down_quantized[2].transpose(-2, -1).contiguous(),
                    routed_down_quantized[3],
                    True,
                )
            else:
                backward_routed_down = routed_down_quantized[2:]
        else:
            routed_gate_quantized = mxfp8_quantize(w_routed_gate, True, True)
            routed_up_quantized = mxfp8_quantize(w_routed_up, True, True)
            routed_down_quantized = mxfp8_quantize(w_routed_down, True, True)
            forward_routed_gate = routed_gate_quantized[:2]
            forward_routed_up = routed_up_quantized[:2]
            forward_routed_down = routed_down_quantized[:2]
            backward_routed_gate = routed_gate_quantized
            backward_routed_up = routed_up_quantized
            backward_routed_down = routed_down_quantized[2:]
        tolerance = MXFP8_TOLERANCE
    else:
        if single_grouped:
            routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
            forward_routed_gate = routed_fc1
            forward_routed_up = routed_fc1
        else:
            forward_routed_gate = w_routed_gate
            forward_routed_up = w_routed_up
        forward_routed_down = w_routed_down
        backward_routed_gate = forward_routed_gate
        backward_routed_up = forward_routed_up
        backward_routed_down = w_routed_down
        tolerance = BF16_TOLERANCE
    if leave_one_expert_empty:
        topk_experts[:, 0] = (
            torch.arange(num_local_tokens, device=device) + rank
        ) % (num_experts - 1)
    schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )

    if single_grouped:
        routed_fc1_main_grad = torch.full_like(routed_fc1, 0.25, dtype=main_grad_dtype)
        main_grads = (
            torch.full_like(w_shared_gate, 0.25, dtype=main_grad_dtype),
            routed_fc1_main_grad,
            torch.full_like(w_shared_up, 0.25, dtype=main_grad_dtype),
            routed_fc1_main_grad,
            torch.full_like(w_shared_down, 0.25, dtype=main_grad_dtype),
            torch.full_like(w_routed_down, 0.25, dtype=main_grad_dtype),
        )
    else:
        main_grads = tuple(
            torch.full_like(weight, 0.25, dtype=main_grad_dtype)
            for weight in (
                w_shared_gate, w_routed_gate, w_shared_up,
                w_routed_up, w_shared_down, w_routed_down,
            )
        )
    # MCore calls the fused backward once per microbatch and expects every call
    # to add into the same main_grad buffers. Exercise that lifecycle,
    # rather than validating only one add into a pre-filled tensor.
    # Production recipes may accumulate hundreds of microbatches. Keep regular
    # CI fast, while allowing focused runs to stress the real accumulation depth.
    num_accumulations = int(os.getenv("MOK_TEST_NUM_ACCUMULATIONS", "3"))
    if num_accumulations <= 0:
        raise ValueError("MOK_TEST_NUM_ACCUMULATIONS must be positive")

    for _ in range(num_accumulations):
        output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            forward_routed_gate,
            forward_routed_up,
            forward_routed_down,
        )
        d_x, d_router_weights, *weight_gradients = functional.backward(
            config,
            workspace,
            schedule,
            forward_context,
            d_output,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            backward_routed_gate,
            backward_routed_up,
            backward_routed_down,
            main_grads=main_grads,
        )
        assert len(weight_gradients) == 7 and weight_gradients[-1] is None

    reference_all = run_reference_bf16(*inputs)
    check_correctness("output", reference_all[0], output, tolerance, rank == 0)
    reference = reference_all[1:]
    check_correctness("d_x", reference[0], d_x, tolerance, rank == 0)
    check_correctness(
        "d_router_weights", reference[1], d_router_weights, tolerance, rank == 0
    )
    if single_grouped:
        reference_wgrads = (
            reference[5],
            torch.cat((reference[2], reference[3]), dim=1),
            reference[6],
            reference[7],
            reference[4],
        )
        actual_main_grads = (
            main_grads[0],
            main_grads[1],
            main_grads[2],
            main_grads[4],
            main_grads[5],
        )
        weight_names = (
            "shared_gate", "routed_fc1", "shared_up",
            "shared_down", "routed_down",
        )
    else:
        reference_wgrads = (
            reference[5], reference[2], reference[6],
            reference[3], reference[7], reference[4],
        )
        actual_main_grads = main_grads
        weight_names = (
            "shared_gate", "routed_gate", "shared_up",
            "routed_up", "shared_down", "routed_down",
        )
    for name, expected, actual in zip(
        weight_names, reference_wgrads, actual_main_grads, strict=True,
    ):
        expected_accum = torch.full_like(expected, 0.25, dtype=main_grad_dtype)
        for _ in range(num_accumulations):
            expected_accum.add_(expected.to(main_grad_dtype))
        check_correctness(
            f"main_grad/{name}",
            expected_accum,
            actual,
            (
                tolerance[0] * num_accumulations * (2 if num_accumulations > 3 else 1),
                tolerance[1] * (2 if num_accumulations > 3 else 1),
            ),
            rank == 0,
        )
