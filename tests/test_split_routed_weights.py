import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from mok import functional, ops
from .utils import BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, generate_inputs, ungated_backward_gradients


def _current_cuda_context() -> int | None:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxGetCurrent.restype = ctypes.c_int
    context = ctypes.c_void_p()
    assert driver.cuCtxGetCurrent(ctypes.byref(context)) == 0
    return context.value


def test_storage_table_builds_on_fresh_host_thread(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    expert_grads = [
        torch.empty((256, 256), dtype=torch.float32, device=device)
        for _ in range(2)
    ]

    def build_table() -> tuple[torch.Tensor, int | None, int | None]:
        before = _current_cuda_context()
        table = ops.make_routed_d_weight_storage_table(expert_grads)
        return table, before, _current_cuda_context()

    with ThreadPoolExecutor(max_workers=1) as executor:
        table, before, after = executor.submit(build_table).result(timeout=30)

    assert table.device == device
    assert (before, after) == (None, None)


def _quantize_experts(
    experts: list[torch.Tensor],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    quantized = [ops.mxfp8_quantize(expert, True, True) for expert in experts]
    rowwise = [value[0] for value in quantized]
    rowwise_scale = [value[1] for value in quantized]
    # MCore/TE keeps a native columnwise payload with the logical matrix shape.
    columnwise = [value[2].transpose(-2, -1).contiguous() for value in quantized]
    columnwise_scale = [value[3] for value in quantized]
    return rowwise, rowwise_scale, columnwise, columnwise_scale


@pytest.mark.parametrize("precision,gated", [
    ("bf16", False), ("mxfp8", False), ("mxfp8", True),
])
def test_split_routed_weights_match_dense(
    context: tuple[int, int, torch.device], precision: str, gated: bool
) -> None:
    rank, world_size, device = context
    num_local_experts = 2
    num_experts = world_size * num_local_experts
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.MoKConfig(
        fwd_num_comm_sms=int(os.getenv("MOK_TEST_FWD_COMM_SMS", "2")),
        bwd_num_comm_sms=int(os.getenv("MOK_TEST_BWD_COMM_SMS", "2")),
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
    gate_kwargs = {}
    if gated:
        gate_kwargs["shared_output_gate_weight"] = x.new_full((1, hidden_dim), hidden_dim ** -0.5)
    routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
    fc1_experts = [routed_fc1[expert].clone() for expert in range(num_local_experts)]
    down_experts = [w_routed_down[expert].clone() for expert in range(num_local_experts)]

    if precision == "mxfp8":
        dense_fc1_quantized = ops.mxfp8_quantize(routed_fc1, True, True)
        dense_down_quantized = ops.mxfp8_quantize(w_routed_down, True, True)
        dense_forward_gate = dense_fc1_quantized[:2]
        dense_forward_up = dense_forward_gate
        dense_forward_down = dense_down_quantized[:2]
        dense_backward_gate = (
            dense_fc1_quantized[0],
            dense_fc1_quantized[1],
            dense_fc1_quantized[2].transpose(-2, -1).contiguous(),
            dense_fc1_quantized[3],
            True,
        )
        dense_backward_up = dense_backward_gate
        dense_backward_down = (
            dense_down_quantized[2].transpose(-2, -1).contiguous(),
            dense_down_quantized[3],
            True,
        )

        fc1_row, fc1_scale, fc1_col, fc1_col_scale = _quantize_experts(fc1_experts)
        down_row, down_scale, down_col, down_col_scale = _quantize_experts(down_experts)
        fc1_row_table = ops.make_routed_weight_storage_table_mxfp8(fc1_row)
        fc1_col_table = ops.make_routed_weight_storage_table_mxfp8(fc1_col)
        down_row_table = ops.make_routed_weight_storage_table_mxfp8(down_row)
        down_col_table = ops.make_routed_weight_storage_table_mxfp8(down_col)
        fc1_scale_table = ops.make_routed_scale_storage_table(fc1_scale)
        fc1_col_scale_table = ops.make_routed_scale_storage_table(fc1_col_scale)
        down_scale_table = ops.make_routed_scale_storage_table(down_scale)
        down_col_scale_table = ops.make_routed_scale_storage_table(down_col_scale)
        split_fc1 = functional.SplitRoutedWeight(
            data=fc1_row[0],
            storage_table=fc1_row_table,
            scale=fc1_scale[0],
            scale_storage_table=fc1_scale_table,
            scale_tensors=tuple(fc1_scale),
            transposed_data=fc1_col[0],
            transposed_scale=fc1_col_scale[0],
            transposed_storage_table=fc1_col_table,
            transposed_scale_storage_table=fc1_col_scale_table,
            transposed_scale_tensors=tuple(fc1_col_scale),
            native_columnwise=True,
        )
        split_down = functional.SplitRoutedWeight(
            data=down_row[0],
            storage_table=down_row_table,
            scale=down_scale[0],
            scale_storage_table=down_scale_table,
            scale_tensors=tuple(down_scale),
            transposed_data=down_col[0],
            transposed_scale=down_col_scale[0],
            transposed_storage_table=down_col_table,
            transposed_scale_storage_table=down_col_scale_table,
            transposed_scale_tensors=tuple(down_col_scale),
            native_columnwise=True,
        )
        tolerance = MXFP8_TOLERANCE
    else:
        dense_forward_gate = routed_fc1
        dense_forward_up = routed_fc1
        dense_forward_down = w_routed_down
        dense_backward_gate = routed_fc1
        dense_backward_up = routed_fc1
        dense_backward_down = w_routed_down
        fc1_table = ops.make_routed_weight_storage_table_bf16(fc1_experts)
        down_table = ops.make_routed_weight_storage_table_bf16(down_experts)
        split_fc1 = functional.SplitRoutedWeight(fc1_experts[0], fc1_table)
        split_down = functional.SplitRoutedWeight(down_experts[0], down_table)
        tolerance = BF16_TOLERANCE

    schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    dense_fc1_main_grad = torch.zeros_like(routed_fc1, dtype=torch.float32)
    dense_down_main_grad = torch.zeros_like(w_routed_down, dtype=torch.float32)
    dense_main_grads = (
        torch.zeros_like(w_shared_gate, dtype=torch.float32),
        dense_fc1_main_grad,
        torch.zeros_like(w_shared_up, dtype=torch.float32),
        dense_fc1_main_grad,
        torch.zeros_like(w_shared_down, dtype=torch.float32),
        dense_down_main_grad,
    )
    dense_output, dense_context = functional.forward(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        dense_forward_gate,
        dense_forward_up,
        dense_forward_down,
        **gate_kwargs,
    )
    dense_backward = functional.backward(
        config,
        workspace,
        schedule,
        dense_context,
        d_output,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        dense_backward_gate,
        dense_backward_up,
        dense_backward_down,
        main_grads=dense_main_grads,
        **gate_kwargs,
    )

    split_fc1_main_grads = [
        torch.zeros_like(weight, dtype=torch.float32) for weight in fc1_experts
    ]
    split_down_main_grads = [
        torch.zeros_like(weight, dtype=torch.float32) for weight in down_experts
    ]
    split_fc1_main_grad_table = ops.make_routed_d_weight_storage_table(
        split_fc1_main_grads
    )
    split_down_main_grad_table = ops.make_routed_d_weight_storage_table(
        split_down_main_grads
    )
    split_main_grads = (
        torch.zeros_like(w_shared_gate, dtype=torch.float32),
        split_fc1_main_grads[0],
        torch.zeros_like(w_shared_up, dtype=torch.float32),
        split_fc1_main_grads[0],
        torch.zeros_like(w_shared_down, dtype=torch.float32),
        split_down_main_grads[0],
    )
    split_output, split_context = functional.forward(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        split_fc1,
        split_fc1,
        split_down,
        **gate_kwargs,
    )
    split_backward = functional.backward(
        config,
        workspace,
        schedule,
        split_context,
        d_output,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        split_fc1,
        split_fc1,
        split_down,
        main_grads=split_main_grads,
        main_grad_storage_tables=(
            split_fc1_main_grad_table,
            split_fc1_main_grad_table,
            split_down_main_grad_table,
        ),
        **gate_kwargs,
    )

    if gated:
        assert len(dense_backward) == len(split_backward) == 9
        assert dense_backward[-1].dtype == split_backward[-1].dtype == torch.float32
        check_correctness("split/d_output_gate", dense_backward[-1], split_backward[-1], BF16_TOLERANCE, rank == 0)
    else:
        ungated_backward_gradients(dense_backward)
        ungated_backward_gradients(split_backward)
    check_correctness("split/output", dense_output, split_output, tolerance, rank == 0)
    check_correctness("split/d_x", dense_backward[0], split_backward[0], tolerance, rank == 0)
    check_correctness(
        "split/d_router_weights",
        dense_backward[1],
        split_backward[1],
        tolerance,
        rank == 0,
    )
    check_correctness(
        "split/main_grad_routed_fc1",
        dense_fc1_main_grad,
        torch.stack(split_fc1_main_grads),
        tolerance,
        rank == 0,
    )
    check_correctness(
        "split/main_grad_routed_down",
        dense_down_main_grad,
        torch.stack(split_down_main_grads),
        tolerance,
        rank == 0,
    )
    for name, dense_grad, split_grad in (
        ("shared_gate", dense_main_grads[0], split_main_grads[0]),
        ("shared_up", dense_main_grads[2], split_main_grads[2]),
        ("shared_down", dense_main_grads[4], split_main_grads[4]),
    ):
        check_correctness(
            f"split/main_grad_{name}", dense_grad, split_grad, tolerance, rank == 0
        )
