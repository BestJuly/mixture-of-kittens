import os

import torch
from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm

from mok import functional
from .utils import BF16_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16


def test_canonical_fc12_wgrad_accum(
    context: tuple[int, int, torch.device],
) -> None:
    """Keep FC1/FC2 optimizer owners while using the six-weight MOK ABI."""
    rank, world_size, device = context
    num_local_experts = int(os.getenv("MOK_TEST_NUM_LOCAL_EXPERTS", "2"))
    hidden_dim = int(os.getenv("MOK_TEST_HIDDEN_DIM", "256"))
    intermediate_dim = int(os.getenv("MOK_TEST_INTERMEDIATE_DIM", "256"))
    topk = int(os.getenv("MOK_TEST_TOPK", "1"))
    num_local_tokens = int(os.getenv("MOK_TEST_NUM_LOCAL_TOKENS", "512"))
    num_experts = world_size * num_local_experts
    config = functional.MoKConfig(
        fwd_num_comm_sms=int(os.getenv("MOK_TEST_FWD_NUM_COMM_SMS", "2")),
        bwd_num_comm_sms=int(os.getenv("MOK_TEST_BWD_NUM_COMM_SMS", "2")),
        minibatch_size=int(os.getenv("MOK_TEST_MINIBATCH_SIZE", "256")),
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
        top_experts,
        router_weights,
        shared_gate,
        shared_up,
        shared_fc2,
        routed_gate,
        routed_up,
        routed_fc2,
        grad_output,
    ) = inputs
    shared_fc1 = torch.cat((shared_gate, shared_up), dim=0)
    routed_fc1 = torch.cat((routed_gate, routed_up), dim=1)
    shared_gate_view = shared_fc1[:intermediate_dim]
    shared_up_view = shared_fc1[intermediate_dim:]
    assert shared_gate_view.is_contiguous() and shared_up_view.is_contiguous()
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=num_local_experts,
    )
    main_grads = tuple(
        torch.full_like(weight, 0.25, dtype=torch.bfloat16)
        for weight in (shared_fc1, routed_fc1, shared_fc2, routed_fc2)
    )
    # The optimizer receives four full owners, without counting FC1 twice.
    # MOK's gate/up descriptors select disjoint halves of the same routed FC1.
    abi_main_grads = (
        main_grads[0][:intermediate_dim],
        main_grads[1],
        main_grads[0][intermediate_dim:],
        main_grads[1],
        main_grads[2],
        main_grads[3],
    )
    assert all(gradient.is_contiguous() for gradient in abi_main_grads)

    output, forward_context = functional.forward(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        shared_gate_view,
        shared_up_view,
        shared_fc2,
        routed_fc1,
        routed_fc1,
        routed_fc2,
    )
    backward_outputs = functional.backward(
        config,
        workspace,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        shared_gate_view,
        shared_up_view,
        shared_fc2,
        routed_fc1,
        routed_fc1,
        routed_fc2,
        main_grads=abi_main_grads,
    )
    assert len(backward_outputs) == 9 and backward_outputs[-1] is None
    d_x, d_router_weights = backward_outputs[:2]
    for actual, expected_alias in zip(
        backward_outputs[2:8],
        (abi_main_grads[1], abi_main_grads[3], abi_main_grads[5],
         abi_main_grads[0], abi_main_grads[2], abi_main_grads[4]),
        strict=True,
    ):
        assert actual is expected_alias

    # Match the optimizer operation that exposed the end-to-end regression.
    noop = torch.zeros(1, dtype=torch.int32, device=device)
    multi_tensor_applier(multi_tensor_l2norm, noop, [list(main_grads)], False)
    torch.cuda.synchronize()

    reference_all = run_reference_bf16(*inputs)
    reference = reference_all[1:]
    check_correctness("output", reference_all[0], output, BF16_TOLERANCE, rank == 0)
    check_correctness("d_x", reference[0], d_x, BF16_TOLERANCE, rank == 0)
    check_correctness(
        "d_router_weights", reference[1], d_router_weights, BF16_TOLERANCE, rank == 0
    )
    expected_grads = (
        torch.cat((reference[5], reference[6]), dim=0),
        torch.cat((reference[2], reference[3]), dim=1),
        reference[7],
        reference[4],
    )
    for name, expected, actual in zip(
        ("shared_fc1", "routed_fc1", "shared_fc2", "routed_fc2"),
        expected_grads,
        main_grads,
        strict=True,
    ):
        check_correctness(
            f"main_grad/{name}",
            expected.to(torch.bfloat16).add(0.25),
            actual,
            BF16_TOLERANCE,
            rank == 0,
        )
