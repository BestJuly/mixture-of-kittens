from typing import Any

import torch

from . import _C


@torch.library.custom_op("mok::all_gather_top_experts", mutates_args=("all_gather_top_experts_buffer",))
def all_gather_top_experts(
    top_experts: torch.Tensor,
    all_gather_top_experts_buffer: torch.Tensor,
    all_gather_top_experts_buffer_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
) -> None:
    """All-gathers top-expert assignments across expert-parallel ranks.

    Inputs:
        top_experts:                                 int32 [num_local_tokens, topk]
        all_gather_top_experts_buffer:               int32 [ep_size, num_local_tokens, topk]
        all_gather_top_experts_buffer_multicast_ptr: int
        rank:                                        int
        chunk_bytes:                                 int

    Outputs:
        None
    """
    if (not top_experts.is_cuda or top_experts.dtype != torch.int32 or not top_experts.is_contiguous() or top_experts.ndim != 2):
        raise ValueError("top_experts must be contiguous CUDA int32 [num_local_tokens, topk]")
    if not all_gather_top_experts_buffer.is_cuda:
        raise ValueError("all_gather_top_experts_buffer must be a CUDA tensor")
    if all_gather_top_experts_buffer.dtype != torch.int32:
        raise TypeError("all_gather_top_experts_buffer must have dtype torch.int32")
    if not all_gather_top_experts_buffer.is_contiguous():
        raise ValueError("all_gather_top_experts_buffer must be contiguous")
    if all_gather_top_experts_buffer.ndim != 3:
        raise ValueError("all_gather_top_experts_buffer must have shape (ep_size, num_local_tokens, topk)")
    if any(size <= 0 for size in top_experts.shape):
        raise ValueError("top_experts dimensions must be positive")
    if any(size <= 0 for size in all_gather_top_experts_buffer.shape):
        raise ValueError("all_gather_top_experts_buffer dimensions must be positive")
    ep_size = all_gather_top_experts_buffer.shape[0]
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("all_gather_top_experts_buffer ep_size must be one of 1, 4, 8, 16, 32, 64")
    if (all_gather_top_experts_buffer.device != top_experts.device
            or tuple(all_gather_top_experts_buffer.shape[1:]) != tuple(top_experts.shape)):
        raise ValueError("all_gather_top_experts_buffer must match top_experts shape and device")
    if type(all_gather_top_experts_buffer_multicast_ptr) is not int or all_gather_top_experts_buffer_multicast_ptr <= 0:
        raise TypeError("all_gather_top_experts_buffer_multicast_ptr must be a positive integer")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if chunk_bytes % 16 != 0:
        raise ValueError("chunk_bytes must be divisible by 16")
    rank_buffer_bytes = top_experts.numel() * top_experts.element_size()
    if rank_buffer_bytes % chunk_bytes != 0:
        raise ValueError("chunk_bytes must divide one rank's route-buffer bytes")

    if ep_size == 1:
        all_gather_top_experts_buffer[0].copy_(top_experts)
    else:
        _C.all_gather_top_experts(top_experts, all_gather_top_experts_buffer, all_gather_top_experts_buffer_multicast_ptr, rank, chunk_bytes)


@torch.library.custom_op("mok::barrier_all", mutates_args=("barrier_buffer", "target"))
def barrier_all(
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    target: torch.Tensor,
) -> None:
    """Synchronizes all expert-parallel ranks at device-side.

    Inputs:
        barrier_buffer:               int32 [1]
        barrier_buffer_ptrs:          list[int] [ep_size]
        barrier_buffer_multicast_ptr: int
        target:                       int32 [1]

    Outputs:
        None
    """
    if not barrier_buffer.is_cuda:
        raise ValueError("barrier_buffer must be a CUDA tensor")
    if barrier_buffer.dtype != torch.int32:
        raise TypeError("barrier_buffer must have dtype torch.int32")
    if not barrier_buffer.is_contiguous():
        raise ValueError("barrier_buffer must be contiguous")
    if tuple(barrier_buffer.shape) != (1,):
        raise ValueError("barrier_buffer must have shape (1,)")
    if not isinstance(barrier_buffer_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in barrier_buffer_ptrs):
        raise TypeError("barrier_buffer_ptrs must be a list of positive integers")
    ep_size = len(barrier_buffer_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("barrier_buffer_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if type(barrier_buffer_multicast_ptr) is not int or barrier_buffer_multicast_ptr <= 0:
        raise TypeError("barrier_buffer_multicast_ptr must be a positive integer")
    if (not target.is_cuda or target.device != barrier_buffer.device
            or target.dtype != torch.int32 or not target.is_contiguous()
            or tuple(target.shape) != (1,)):
        raise ValueError("target must be contiguous int32 [1] on the barrier CUDA device")

    if ep_size == 1:
        return
    else:
        _C.barrier_all(barrier_buffer, barrier_buffer_ptrs, barrier_buffer_multicast_ptr, target)


@torch.library.custom_op("mok::schedule", mutates_args=())
def schedule(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Builds the routed-token schedule for the current expert-parallel rank.

    Inputs:
        topk_all:          int32 [ep_size, num_local_tokens, topk]
        num_local_experts: int
        schedule_capacity: int
        rank:              int

    Outputs:
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
    """
    if topk_all.ndim != 3:
        raise ValueError("topk_all must have shape (ep_size, num_local_tokens, topk)")
    ep_size, num_local_tokens, topk = topk_all.shape
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("topk_all ep_size must be one of 1, 4, 8, 16, 32, 64")
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError(
            "topk_all num_local_tokens must be at least 512 and divisible by 256"
        )
    if not 0 < topk <= 255:
        raise ValueError("topk_all topk must be in [1, 255]")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if (type(schedule_capacity) is not int or schedule_capacity <= 0
            or schedule_capacity % 256 != 0):
        raise ValueError("schedule_capacity must be positive and divisible by 256")
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule_capacity must hold at least one rank's routed tokens")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")

    return _C.schedule(topk_all, num_local_experts, schedule_capacity, rank)


@torch.library.custom_op(
    "mok::mxfp8_quantize", mutates_args=(),
    schema="(Tensor x_bf16, bool return_normal, bool return_transposed) -> (Tensor?, Tensor?, Tensor?, Tensor?)",
)
def mxfp8_quantize(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Quantizes BF16 matrices to MXFP8 in normal and/or transposed layouts.

    Inputs:
        x_bf16:           bfloat16 [M, N] or [E, M, N]
        return_normal:     bool
        return_transposed: bool

    Outputs:
        x_fp8:   float8_e4m3fn [M, N] or [E, M, N] | None
        x_sc:    uint8 [E * M // 128, N // 128, 32, 16] | None
        x_fp8_t: float8_e4m3fn [N, M] or [E, N, M] | None
        x_sc_t:  uint8 [E * N // 128, M // 128, 32, 16] | None
    """
    if x_bf16.ndim not in (2, 3):
        raise ValueError("x_bf16 must have shape (M, N) or (E, M, N)")
    if any(size <= 0 for size in x_bf16.shape):
        raise ValueError("x_bf16 dimensions must be positive")
    if x_bf16.shape[-2] % 128 != 0 or x_bf16.shape[-1] % 128 != 0:
        raise ValueError("x_bf16 M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("at least one quantized layout must be requested")

    return _C.mxfp8_quantize(x_bf16, return_normal, return_transposed)


def _make_routed_storage_table(
    expert_tensors: list[torch.Tensor], *, precision: str
) -> torch.Tensor:
    """Build a reusable device table of per-expert TK/TMA layouts.

    This is a one-time setup operation and must run before CUDA graph capture.
    The returned tensor owns only descriptors; it does not copy weight data.
    """
    if not isinstance(expert_tensors, list) or not expert_tensors:
        raise TypeError("expert_tensors must be a nonempty list of tensors")
    first = expert_tensors[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("each expert tensor must be a torch.Tensor")
    if not first.is_cuda or not first.is_contiguous() or first.ndim != 2:
        raise ValueError("each expert tensor must be a contiguous CUDA matrix")
    for tensor in expert_tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != first.device
            or tensor.dtype != first.dtype
            or tensor.shape != first.shape
            or not tensor.is_contiguous()
        ):
            raise ValueError("all expert tensors must have identical dtype, shape, and device")
    if precision == "mxfp8":
        if first.dtype != torch.float8_e4m3fn:
            raise TypeError("MXFP8 expert tensors must have dtype float8_e4m3fn")
        return _C.make_routed_weight_storage_table_mxfp8(expert_tensors)
    if precision == "bf16":
        if first.dtype != torch.bfloat16:
            raise TypeError("BF16 expert tensors must have dtype bfloat16")
        return _C.make_routed_weight_storage_table_bf16(expert_tensors)
    if precision == "wgrad":
        if first.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("expert main grads must have dtype float32 or bfloat16")
        return _C.make_routed_d_weight_storage_table(expert_tensors)
    raise ValueError(f"unsupported routed storage-table precision: {precision}")


def make_routed_weight_storage_table_mxfp8(
    expert_tensors: list[torch.Tensor],
) -> torch.Tensor:
    return _make_routed_storage_table(expert_tensors, precision="mxfp8")


def make_routed_weight_storage_table_bf16(
    expert_tensors: list[torch.Tensor],
) -> torch.Tensor:
    return _make_routed_storage_table(expert_tensors, precision="bf16")


def make_routed_scale_storage_table(
    expert_tensors: list[torch.Tensor],
) -> torch.Tensor:
    """Build a reusable table of per-expert tcgen05 scale layouts."""
    if not isinstance(expert_tensors, list) or not expert_tensors:
        raise TypeError("expert_tensors must be a nonempty list of tensors")
    first = expert_tensors[0]
    if (
        not isinstance(first, torch.Tensor)
        or not first.is_cuda
        or first.dtype != torch.uint8
        or not first.is_contiguous()
        or first.ndim != 4
        or tuple(first.shape[-2:]) != (32, 16)
    ):
        raise ValueError(
            "each expert scale must be contiguous CUDA uint8 shaped "
            "[row_blocks, col_blocks, 32, 16]"
        )
    for tensor in expert_tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != first.device
            or tensor.dtype != torch.uint8
            or tensor.shape != first.shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "all expert scales must have identical dtype, shape, and device"
            )
    return _C.make_routed_scale_storage_table(expert_tensors)


def make_routed_d_weight_storage_table(
    expert_tensors: list[torch.Tensor],
) -> torch.Tensor:
    return _make_routed_storage_table(expert_tensors, precision="wgrad")


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_mxfp8", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_mxfp8(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    w_routed_down_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
    w_routed_gate_storage_table: torch.Tensor | None = None,
    w_routed_up_storage_table: torch.Tensor | None = None,
    w_routed_down_storage_table: torch.Tensor | None = None,
    w_routed_gate_sc_storage_table: torch.Tensor | None = None,
    w_routed_up_sc_storage_table: torch.Tensor | None = None,
    w_routed_down_sc_storage_table: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused MXFP8 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [num_local_tokens * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_down_sc:        uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        y_shared:            bfloat16 [num_local_tokens, hidden_size]
        y_routed:            bfloat16 [macrobatch_size, hidden_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (("x_ptrs", x_ptrs),
                                   ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    split_storage = w_routed_gate_storage_table is not None
    if (
        split_storage != (w_routed_up_storage_table is not None)
        or split_storage != (w_routed_down_storage_table is not None)
    ):
        raise ValueError("all three routed storage tables must be provided together")
    split_scale_storage = w_routed_gate_sc_storage_table is not None
    if (
        split_scale_storage != (w_routed_up_sc_storage_table is not None)
        or split_scale_storage != (w_routed_down_sc_storage_table is not None)
    ):
        raise ValueError("all three routed scale storage tables must be provided together")
    if split_scale_storage != split_storage:
        raise ValueError(
            "split MXFP8 weights and scales must provide descriptor tables together"
        )
    if split_storage:
        num_local_experts = tokens_per_expert.numel()
        if num_local_experts <= 0 or w_routed_gate.ndim != 2:
            raise ValueError("split w_routed_gate must be one expert matrix")
    else:
        if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
            raise ValueError("w_routed_gate must have shape "
                             "(num_local_experts, intermediate_size, hidden_size)")
        num_local_experts = w_routed_gate.shape[0]
    combined_fc1_payload = w_routed_gate is w_routed_up
    combined_fc1_scale_payload = w_routed_gate_sc is w_routed_up_sc
    if combined_fc1_payload != combined_fc1_scale_payload:
        raise ValueError(
            "combined FC1 data and scale tensors must alias consistently"
        )
    if split_storage:
        if combined_fc1_payload != (
            w_routed_gate_storage_table is w_routed_up_storage_table
        ):
            raise ValueError(
                "combined FC1 weights and weight descriptor tables must alias consistently"
            )
        if combined_fc1_scale_payload != (
            w_routed_gate_sc_storage_table is w_routed_up_sc_storage_table
        ):
            raise ValueError(
                "combined FC1 scales and scale descriptor tables must alias consistently"
            )
    routed_fc1_shape = (
        ((2 if combined_fc1_payload else 1) * intermediate_size, hidden_size)
        if split_storage
        else (
            num_local_experts,
            (2 if combined_fc1_payload else 1) * intermediate_size,
            hidden_size,
        )
    )
    routed_fc1_sc_shape = (
        (
            (2 if combined_fc1_payload else 1) * intermediate_size // 128,
            hidden_size // 128, 32, 16,
        )
        if split_scale_storage
        else (
            num_local_experts * (2 if combined_fc1_payload else 1) * intermediate_size // 128,
            hidden_size // 128, 32, 16,
        )
    )
    expected_shapes = (
        ("combine_buffer", combine_buffer, (num_local_tokens * topk, hidden_size)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, routed_fc1_shape),
        ("w_routed_gate_sc", w_routed_gate_sc, routed_fc1_sc_shape),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, routed_fc1_shape),
        ("w_routed_up_sc", w_routed_up_sc, routed_fc1_sc_shape),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down,
         ((hidden_size, intermediate_size) if split_storage
          else (num_local_experts, hidden_size, intermediate_size))),
        ("w_routed_down_sc", w_routed_down_sc,
         ((hidden_size // 128, intermediate_size // 128, 32, 16)
          if split_scale_storage
          else (num_local_experts * hidden_size // 128,
                intermediate_size // 128, 32, 16))),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("w_routed_down_sc", w_routed_down_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if split_storage:
        for tensor_name, table in (
            ("w_routed_gate_storage_table", w_routed_gate_storage_table),
            ("w_routed_up_storage_table", w_routed_up_storage_table),
            ("w_routed_down_storage_table", w_routed_down_storage_table),
            ("w_routed_gate_sc_storage_table", w_routed_gate_sc_storage_table),
            ("w_routed_up_sc_storage_table", w_routed_up_sc_storage_table),
            ("w_routed_down_sc_storage_table", w_routed_down_sc_storage_table),
        ):
            if (
                table.device != x.device
                or table.dtype != torch.uint8
                or not table.is_contiguous()
                or table.numel() == 0
            ):
                raise ValueError(
                    f"{tensor_name} must be nonempty contiguous CUDA uint8 on {x.device}"
                )
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.dispatch_mlp_swiglu_combine_fwd_mxfp8(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        w_shared_down, w_routed_down, w_routed_down_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
        w_routed_gate_storage_table, w_routed_up_storage_table,
        w_routed_down_storage_table,
        w_routed_gate_sc_storage_table, w_routed_up_sc_storage_table,
        w_routed_down_sc_storage_table,
    )


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_bf16", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_bf16(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
    w_routed_gate_storage_table: torch.Tensor | None = None,
    w_routed_up_storage_table: torch.Tensor | None = None,
    w_routed_down_storage_table: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused BF16 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [num_local_tokens * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           bfloat16 [num_local_experts, hidden_size, intermediate_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        y_shared:      bfloat16 [num_local_tokens, hidden_size]
        y_routed:      bfloat16 [macrobatch_size, hidden_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    split_storage = w_routed_gate_storage_table is not None
    if (
        split_storage != (w_routed_up_storage_table is not None)
        or split_storage != (w_routed_down_storage_table is not None)
    ):
        raise ValueError("all three routed storage tables must be provided together")
    if split_storage:
        num_local_experts = tokens_per_expert.numel()
        if num_local_experts <= 0 or w_routed_gate.ndim != 2:
            raise ValueError("split w_routed_gate must be one expert matrix")
    else:
        if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
            raise ValueError(
                "w_routed_gate must have shape "
                "(num_local_experts, intermediate_size, hidden_size)"
            )
        num_local_experts = w_routed_gate.shape[0]
    combined_fc1_payload = w_routed_gate is w_routed_up
    if split_storage and combined_fc1_payload != (
        w_routed_gate_storage_table is w_routed_up_storage_table
    ):
        raise ValueError(
            "combined FC1 weights and weight descriptor tables must alias consistently"
        )
    routed_fc1_shape = (
        ((2 if combined_fc1_payload else 1) * intermediate_size, hidden_size)
        if split_storage
        else (
            num_local_experts,
            (2 if combined_fc1_payload else 1) * intermediate_size,
            hidden_size,
        )
    )
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (("x_ptrs", x_ptrs), ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("combine_buffer", combine_buffer, (num_local_tokens * topk, hidden_size)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, routed_fc1_shape),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, routed_fc1_shape),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down,
         ((hidden_size, intermediate_size) if split_storage
          else (num_local_experts, hidden_size, intermediate_size))),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if split_storage:
        for tensor_name, table in (
            ("w_routed_gate_storage_table", w_routed_gate_storage_table),
            ("w_routed_up_storage_table", w_routed_up_storage_table),
            ("w_routed_down_storage_table", w_routed_down_storage_table),
        ):
            if (
                table.device != x.device
                or table.dtype != torch.uint8
                or not table.is_contiguous()
                or table.numel() == 0
            ):
                raise ValueError(
                    f"{tensor_name} must be nonempty contiguous CUDA uint8 on {x.device}"
                )

    return _C.dispatch_mlp_swiglu_combine_fwd_bf16(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
        w_routed_gate_storage_table, w_routed_up_storage_table,
        w_routed_down_storage_table,
    )


@torch.library.custom_op("mok::recompute_forward_context_mxfp8", mutates_args=())
def recompute_forward_context_mxfp8(
    x: torch.Tensor,
    x_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Recomputes the MXFP8 MoE forward context.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if not isinstance(x_ptrs, list) or any(type(pointer) is not int or pointer <= 0 for pointer in x_ptrs):
        raise TypeError("x_ptrs must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.recompute_forward_context_mxfp8(
        x, x_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::recompute_forward_context_bf16", mutates_args=())
def recompute_forward_context_bf16(
    x: torch.Tensor,
    x_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Recomputes the BF16 MoE forward context.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if not isinstance(x_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in x_ptrs
    ):
        raise TypeError("x_ptrs must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor, _ in expected_shapes:
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.device != x.device:
        raise ValueError(f"schedule_peer_rank must be on {x.device}")

    return _C.recompute_forward_context_bf16(
        x, x_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_mxfp8",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer",
                  "x_fp8_t_routed", "x_sc_t_routed",
                  "gate_fp8_routed", "gate_sc_routed", "up_fp8_routed", "up_sc_routed",
                  "hidden_fp8_t_routed", "hidden_sc_t_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_mxfp8(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate_T: torch.Tensor,
    w_routed_gate_T_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up_T: torch.Tensor,
    w_routed_up_T_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down_T: torch.Tensor,
    w_routed_down_T_sc: torch.Tensor,
    x_fp8_t_routed: torch.Tensor,
    x_sc_t_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_fp8_routed: torch.Tensor,
    gate_sc_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_fp8_routed: torch.Tensor,
    up_sc_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_fp8_t_routed: torch.Tensor,
    hidden_sc_t_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
    routed_weights_are_native_columnwise: bool = False,
    shared_grad_output: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused MXFP8 MoE backward pass.

    ``shared_grad_output`` optionally supplies BF16 [T, H] gradients for
    the shared MLP only. Routed dispatch/quantization and router backward
    retain the original ``d_y_buffer``. None preserves the ungated path.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [num_local_tokens * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [num_local_tokens, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [num_local_tokens, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate_T:               legacy: float8_e4m3fn [num_local_experts, hidden_size, intermediate_size];
                                         native single-grouped: shared FC1 columnwise payload [num_local_experts, 2 * intermediate_size, hidden_size]
        w_routed_gate_T_sc:            legacy: uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16];
                                         native single-grouped: uint8 [num_local_experts * hidden_size // 128, 2 * intermediate_size // 128, 32, 16]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up_T:                 legacy: float8_e4m3fn [num_local_experts, hidden_size, intermediate_size];
                                         native single-grouped: alias of w_routed_gate_T
        w_routed_up_T_sc:              legacy: uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16];
                                         native single-grouped: alias of w_routed_gate_T_sc
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down_T:               legacy: float8_e4m3fn [num_local_experts, intermediate_size, hidden_size];
                                         native single-grouped: float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_down_T_sc:            uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        x_fp8_t_routed:                float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:                 uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:               float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:                uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:                 float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:                  uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed:           float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:            uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        w_routed_gate:                 float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:              uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_routed_up:                   float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:                uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int
        routed_weights_are_native_columnwise:
                                        bool; selects native TE/MCore single-grouped columnwise payloads

    Outputs:
        d_x_shared:          bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:          bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:       bfloat16 [num_local_tokens, intermediate_size]
        d_gate_fp8_routed:   float8_e4m3fn [macrobatch_size, intermediate_size]
        d_gate_sc_routed:    uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_up_shared:         bfloat16 [num_local_tokens, intermediate_size]
        d_up_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        d_up_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_hidden_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_y_fp8_routed:      float8_e4m3fn [macrobatch_size, hidden_size]
        d_y_sc_routed:       uint8 [macrobatch_size // 128, hidden_size // 128, 32, 16]
        d_w_shared_gate:     bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate:     bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:       bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:       bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down:     bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down:     bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if type(routed_weights_are_native_columnwise) is not bool:
        raise TypeError("routed_weights_are_native_columnwise must be a bool")

    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    ):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
    ):
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    combined_fc1_payload = w_routed_gate is w_routed_up
    combined_fc1_scale_payload = w_routed_gate_sc is w_routed_up_sc
    combined_fc1_transposed_payload = w_routed_gate_T is w_routed_up_T
    combined_fc1_transposed_scale_payload = (
        w_routed_gate_T_sc is w_routed_up_T_sc
    )
    if combined_fc1_payload != combined_fc1_scale_payload:
        raise ValueError(
            "combined rowwise FC1 data and scales must alias consistently"
        )
    if combined_fc1_transposed_payload != combined_fc1_transposed_scale_payload:
        raise ValueError(
            "combined columnwise FC1 data and scales must alias consistently"
        )
    if routed_weights_are_native_columnwise and not (
        combined_fc1_payload and combined_fc1_scale_payload
        and combined_fc1_transposed_payload and combined_fc1_transposed_scale_payload
    ):
        raise ValueError("native columnwise weights require combined FC1 aliases")

    mb_i_sc = (macrobatch_size // 128, intermediate_size // 128, 32, 16)
    i_mb_sc = (intermediate_size // 128, macrobatch_size // 128, 32, 16)
    h_mb_sc = (hidden_size // 128, macrobatch_size // 128, 32, 16)
    e_i_h_sc = (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)
    routed_fc1_shape = (
        num_local_experts, (2 if combined_fc1_payload else 1) * intermediate_size, hidden_size
    )
    routed_fc1_sc_shape = (
        num_local_experts * (2 if combined_fc1_payload else 1) * intermediate_size // 128,
        hidden_size // 128, 32, 16,
    )
    routed_fc1_t_shape = (
        routed_fc1_shape if routed_weights_are_native_columnwise else
        (num_local_experts, hidden_size,
         (2 if combined_fc1_transposed_payload else 1) * intermediate_size)
    )
    routed_fc1_t_sc_shape = (
        num_local_experts * hidden_size // 128,
        (2 if combined_fc1_transposed_payload else 1) * intermediate_size // 128, 32, 16,
    )
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (num_local_tokens * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (num_local_tokens, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (num_local_tokens, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate_T", w_routed_gate_T, routed_fc1_t_shape),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc, routed_fc1_t_sc_shape),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up_T", w_routed_up_T, routed_fc1_t_shape),
        ("w_routed_up_T_sc", w_routed_up_T_sc, routed_fc1_t_sc_shape),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down_T", w_routed_down_T,
         ((num_local_experts, hidden_size, intermediate_size)
          if routed_weights_are_native_columnwise else
          (num_local_experts, intermediate_size, hidden_size))),
        ("w_routed_down_T_sc", w_routed_down_T_sc, e_i_h_sc),
        ("x_fp8_t_routed", x_fp8_t_routed, (hidden_size, macrobatch_size)),
        ("x_sc_t_routed", x_sc_t_routed, h_mb_sc),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_fp8_routed", gate_fp8_routed, (macrobatch_size, intermediate_size)),
        ("gate_sc_routed", gate_sc_routed, mb_i_sc),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_fp8_routed", up_fp8_routed, (macrobatch_size, intermediate_size)),
        ("up_sc_routed", up_sc_routed, mb_i_sc),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed, (intermediate_size, macrobatch_size)),
        ("hidden_sc_t_routed", hidden_sc_t_routed, i_mb_sc),
        ("w_routed_gate", w_routed_gate, routed_fc1_shape),
        ("w_routed_gate_sc", w_routed_gate_sc, routed_fc1_sc_shape),
        ("w_routed_up", w_routed_up, routed_fc1_shape),
        ("w_routed_up_sc", w_routed_up_sc, routed_fc1_sc_shape),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("d_y_buffer", d_y_buffer),
        ("d_x_routed_buffer", d_x_routed_buffer),
        ("router_weight_buffer", router_weight_buffer),
        ("d_router_weight_buffer", d_router_weight_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate_T", w_routed_gate_T),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up_T", w_routed_up_T),
        ("w_routed_up_T_sc", w_routed_up_T_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down_T", w_routed_down_T),
        ("w_routed_down_T_sc", w_routed_down_T_sc),
        ("x_fp8_t_routed", x_fp8_t_routed),
        ("x_sc_t_routed", x_sc_t_routed),
        ("gate_shared", gate_shared),
        ("gate_fp8_routed", gate_fp8_routed),
        ("gate_sc_routed", gate_sc_routed),
        ("up_shared", up_shared),
        ("up_fp8_routed", up_fp8_routed),
        ("up_sc_routed", up_sc_routed),
        ("hidden_shared", hidden_shared),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed),
        ("hidden_sc_t_routed", hidden_sc_t_routed),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    if shared_grad_output is not None:
        _validate_shared_grad_output(shared_grad_output, d_y_buffer)

    return _C.dispatch_mlp_swiglu_combine_bwd_mxfp8(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
        w_shared_up, w_routed_up_T, w_routed_up_T_sc,
        w_shared_down, w_routed_down_T, w_routed_down_T_sc,
        x_fp8_t_routed, x_sc_t_routed,
        gate_shared, gate_fp8_routed, gate_sc_routed,
        up_shared, up_fp8_routed, up_sc_routed,
        hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
        x, x_ptrs, w_routed_gate, w_routed_gate_sc, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
        routed_weights_are_native_columnwise,
        shared_grad_output=shared_grad_output,
    )


def dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(
    *args: Any,
    main_grads: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    weight_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
    scale_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
    main_grad_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
    shared_grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Runs MXFP8 backward and accumulates wgrad directly into FP32 or BF16 buffers.

    ``args`` are the inputs of :func:`dispatch_mlp_swiglu_combine_bwd_mxfp8`.
    ``main_grads`` are ordered as shared gate, routed gate, shared up, routed
    up, shared down, routed down. The returned tuple contains only the twelve
    non-weight-gradient outputs; the six weight gradients are side effects.

    This eager wrapper is separate from the original custom op because custom
    op outputs may not alias mutated inputs.
    ``shared_grad_output`` follows the fresh op's BF16 shared-only contract.
    """
    if len(main_grads) != 6:
        raise ValueError("main_grads must contain six tensors")
    main_grad_dtype = main_grads[0].dtype
    if main_grad_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("main_grads must have dtype float32 or bfloat16")
    for name, main_grad in zip(
        (
            "shared_gate", "routed_gate", "shared_up",
            "routed_up", "shared_down", "routed_down",
        ),
        main_grads,
        strict=True,
    ):
        if not main_grad.is_cuda:
            raise TypeError(f"main_grad_{name} must be a CUDA tensor")
        if main_grad.dtype != main_grad_dtype:
            raise TypeError("all main_grads must have the same dtype")
        if not main_grad.is_contiguous():
            raise ValueError(f"main_grad_{name} must be contiguous")
    split_tables = weight_storage_tables is not None
    if (
        split_tables != (scale_storage_tables is not None)
        or split_tables != (main_grad_storage_tables is not None)
    ):
        raise ValueError(
            "split weight, scale, and main-grad storage tables must be provided together"
        )
    if shared_grad_output is not None:
        _validate_shared_grad_output(shared_grad_output, args[0])
    if weight_storage_tables is None:
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(
            *args, *main_grads, shared_grad_output=shared_grad_output
        )
    else:
        if (
            len(weight_storage_tables) != 5
            or len(scale_storage_tables) != 5
            or len(main_grad_storage_tables) != 3
        ):
            raise ValueError(
                "MXFP8 split storage tables must contain five weight, five scale, "
                "and three main-grad tables"
            )
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(
            *args, *main_grads, *weight_storage_tables, *scale_storage_tables,
            *main_grad_storage_tables, shared_grad_output=shared_grad_output
        )
    return outputs[:12]


def _validate_shared_grad_output(shared_grad_output: torch.Tensor, d_y_buffer: torch.Tensor) -> None:
    if (
        not shared_grad_output.is_cuda
        or shared_grad_output.dtype != torch.bfloat16
        or not shared_grad_output.is_contiguous()
        or shared_grad_output.device != d_y_buffer.device
        or shared_grad_output.shape != d_y_buffer.shape
    ):
        raise ValueError("shared_grad_output must be contiguous CUDA BF16 with d_y_buffer's shape/device")


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_bf16",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer", "x_routed", "gate_routed", "up_routed", "hidden_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_bf16(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    x_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
    shared_grad_output: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused BF16 MoE backward pass.

    ``shared_grad_output`` optionally supplies BF16 [T, H] upstream gradients
    for the shared MLP only. Routed dispatch and router-weight backward keep
    using the original ``d_y_buffer``. None preserves the ungated path.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [num_local_tokens * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [num_local_tokens, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [num_local_tokens, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:                 bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up:                   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down:                 bfloat16 [num_local_experts, hidden_size, intermediate_size]
        x_routed:                      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:                   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:                     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed:                 bfloat16 [macrobatch_size, intermediate_size]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:      bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        d_gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        d_up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        d_y_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_w_shared_gate: bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:   bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down: bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down: bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    combined_fc1_payload = w_routed_gate is w_routed_up
    routed_fc1_shape = (
        num_local_experts,
        (2 if combined_fc1_payload else 1) * intermediate_size,
        hidden_size,
    )
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    pointer_lists = (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    )
    for pointer_name, pointers in pointer_lists:
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    for pointer_name, pointers in pointer_lists[:-1]:
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (num_local_tokens * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (num_local_tokens, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (num_local_tokens, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, routed_fc1_shape),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, routed_fc1_shape),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("x_routed", x_routed, (macrobatch_size, hidden_size)),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_routed", gate_routed, (macrobatch_size, intermediate_size)),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_routed", up_routed, (macrobatch_size, intermediate_size)),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_routed", hidden_routed, (macrobatch_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor, _ in expected_shapes:
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.device != x.device:
        raise ValueError(f"schedule_peer_rank must be on {x.device}")

    if shared_grad_output is not None:
        _validate_shared_grad_output(shared_grad_output, d_y_buffer)

    return _C.dispatch_mlp_swiglu_combine_bwd_bf16(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        x_routed, gate_shared, gate_routed, up_shared, up_routed,
        hidden_shared, hidden_routed,
        x, x_ptrs,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
        shared_grad_output=shared_grad_output,
    )


def dispatch_mlp_swiglu_combine_bwd_bf16_accum(
    *args: Any,
    main_grads: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    weight_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
    main_grad_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
    shared_grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Runs BF16 backward and accumulates wgrad directly into FP32 or BF16 buffers.

    ``args`` are the inputs of :func:`dispatch_mlp_swiglu_combine_bwd_bf16`.
    ``main_grads`` are ordered as shared gate, routed gate, shared up, routed
    up, shared down, routed down. The returned tuple contains only the nine
    non-weight-gradient outputs; the six weight gradients are side effects.

    This eager wrapper is intentionally separate from the original custom op:
    PyTorch custom-op outputs may not alias mutated inputs, while the C++ entry
    point returns the supplied buffers for convenient internal bookkeeping.
    ``shared_grad_output`` follows the fresh op's shared-only gradient contract.
    """
    if len(main_grads) != 6:
        raise ValueError("main_grads must contain six tensors")
    main_grad_dtype = main_grads[0].dtype
    if main_grad_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("main_grads must have dtype float32 or bfloat16")
    for name, main_grad in zip(
        ("shared_gate", "routed_gate", "shared_up", "routed_up", "shared_down", "routed_down"),
        main_grads,
        strict=True,
    ):
        if not main_grad.is_cuda:
            raise TypeError(f"main_grad_{name} must be a CUDA tensor")
        if main_grad.dtype != main_grad_dtype:
            raise TypeError("all main_grads must have the same dtype")
        if not main_grad.is_contiguous():
            raise ValueError(f"main_grad_{name} must be contiguous")
    if (weight_storage_tables is None) != (main_grad_storage_tables is None):
        raise ValueError(
            "split weight and main-grad storage tables must be provided together"
        )
    if shared_grad_output is not None:
        _validate_shared_grad_output(shared_grad_output, args[0])
    if weight_storage_tables is None:
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_bf16(
            *args, *main_grads, shared_grad_output=shared_grad_output
        )
    else:
        if len(weight_storage_tables) != 3 or len(main_grad_storage_tables) != 3:
            raise ValueError(
                "BF16 split storage tables must contain three weight and three main-grad tables"
            )
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_bf16(
            *args, *main_grads, *weight_storage_tables, *main_grad_storage_tables,
            shared_grad_output=shared_grad_output,
        )
    return outputs[:9]


@torch.library.custom_op("mok::fwd_epilogue", mutates_args=())
def fwd_epilogue(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
    top_experts: torch.Tensor,
    shared_output_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combines shared and router-weighted routed expert outputs.

    Inputs:
        y_shared:       bfloat16 [num_local_tokens, hidden_size]
        combine_buffer: bfloat16 [num_local_tokens * topk, hidden_size]
        topk_weights:   float32 [num_local_tokens, topk]
        top_experts:    int32 [num_local_tokens, topk], with -1 for invalid routes
        shared_output_gate: optional contiguous bfloat16 [num_local_tokens, 1].
                            Scales shared output in FP32 before routed accumulation.

    Outputs:
        output: bfloat16 [num_local_tokens, hidden_size]
    """
    if y_shared.ndim != 2:
        raise ValueError("y_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = y_shared.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if (topk_weights.device != y_shared.device
            or combine_buffer.device != y_shared.device
            or top_experts.device != y_shared.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if topk_weights.ndim != 2 or topk_weights.shape[0] != num_local_tokens:
        raise ValueError("topk_weights must have shape (num_local_tokens, topk)")
    topk = topk_weights.shape[1]
    if not 0 < topk <= 255:
        raise ValueError("topk must be in [1, 255]")
    if tuple(combine_buffer.shape) != (num_local_tokens * topk, hidden_size):
        raise ValueError("combine_buffer must have shape (num_local_tokens * topk, hidden_size)")
    if (top_experts.dtype != torch.int32 or not top_experts.is_contiguous()
            or tuple(top_experts.shape) != (num_local_tokens, topk)):
        raise ValueError("top_experts must be contiguous int32 (num_local_tokens, topk)")


    if shared_output_gate is None:
        return _C.fwd_epilogue(y_shared, combine_buffer, topk_weights, top_experts)
    if (
        not shared_output_gate.is_cuda
        or shared_output_gate.dtype != torch.bfloat16
        or not shared_output_gate.is_contiguous()
        or shared_output_gate.device != y_shared.device
        or tuple(shared_output_gate.shape) != (num_local_tokens, 1)
    ):
        raise ValueError("shared_output_gate must be contiguous CUDA BF16 [T, 1] on y_shared's device")
    return _C.fwd_epilogue(
        y_shared, combine_buffer, topk_weights, top_experts,
        shared_output_gate=shared_output_gate,
    )


@torch.library.custom_op("mok::bwd_epilogue", mutates_args=())
def bwd_epilogue(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
    top_experts: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and routed input gradients.

    Inputs:
        d_x_shared:        bfloat16 [num_local_tokens, hidden_size]
        d_x_routed_buffer: bfloat16 [num_local_tokens * topk, hidden_size]
        top_experts:       int32 [num_local_tokens, topk], with -1 for invalid routes

    Outputs:
        d_x: bfloat16 [num_local_tokens, hidden_size]
    """
    if d_x_shared.ndim != 2:
        raise ValueError("d_x_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = d_x_shared.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if (d_x_routed_buffer.device != d_x_shared.device
            or top_experts.device != d_x_shared.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if d_x_routed_buffer.ndim != 2:
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")
    if (d_x_routed_buffer.shape[1] != hidden_size
            or d_x_routed_buffer.shape[0] % num_local_tokens != 0
            or d_x_routed_buffer.shape[0] == 0
            or d_x_routed_buffer.shape[0] > num_local_tokens * 255):
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")
    topk = d_x_routed_buffer.shape[0] // num_local_tokens
    if (top_experts.device != d_x_shared.device
            or top_experts.dtype != torch.int32
            or not top_experts.is_contiguous()
            or tuple(top_experts.shape) != (num_local_tokens, topk)):
        raise ValueError("top_experts must be contiguous int32 (num_local_tokens, topk)")


    return _C.bwd_epilogue(d_x_shared, d_x_routed_buffer, top_experts)
