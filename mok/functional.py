import math
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils import _python_dispatch

from . import _C
from . import _gate_fused
from .ops import (
    _check_accumulation_operands,
    all_gather_top_experts,
    barrier_all,
    bwd_epilogue,
    dispatch_mlp_swiglu_combine_bwd_mxfp8,
    dispatch_mlp_swiglu_combine_bwd_bf16,
    dispatch_mlp_swiglu_combine_bwd_mxfp8_accum,
    dispatch_mlp_swiglu_combine_bwd_bf16_accum,
    recompute_forward_context_mxfp8,
    recompute_forward_context_bf16,
    dispatch_mlp_swiglu_combine_fwd_mxfp8,
    dispatch_mlp_swiglu_combine_fwd_bf16,
    fwd_epilogue,
    schedule,
    stage_into_buffers,
)


@dataclass(frozen=True, slots=True)
class MoKConfig:
    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 0.5
    all_gather_top_experts_chunk_bytes: int = 2048


@dataclass(frozen=True, slots=True)
class MoKSchedule:
    peer_rank: torch.Tensor          # (schedule_capacity,) int32
    peer_token_idx: torch.Tensor     # (schedule_capacity,) int32
    num_tokens: torch.Tensor         # (1,) int32
    tokens_per_expert: torch.Tensor  # (num_local_experts,) int32
    top_experts: torch.Tensor  # (num_local_tokens, topk) int32; -1 marks no route


@dataclass(frozen=True, slots=True)
class MoKForwardContext:
    x_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    up_shared: torch.Tensor
    up_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    # Per-forward tensors, never workspace/ring-buffer views. Ungated keeps neither.
    shared_output: torch.Tensor | None = None
    shared_output_gate: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class SplitRoutedWeight:
    """One logical routed matrix stored in independent per-expert allocations.

    ``data`` and ``scale`` are representative expert tensors required by the
    Tensor-based custom-op interface; the descriptor tables select the actual
    expert allocation at runtime. The tables own neither payload nor scale
    storage. MCore parameters already own every data payload, so no parallel
    ``data_tensors`` collection is needed. In contrast, the tcgen05-layout
    scales are generated specifically for MOK and have no other owner, so
    ``scale_tensors`` retains every expert scale for the descriptor lifetime.

    MXFP8 backward also provides a columnwise payload through the transposed
    fields. ``native_columnwise=False`` means the legacy payload is physically
    transposed and uses the ABt dgrad path. ``native_columnwise=True`` means it
    is TE's native columnwise-quantized payload in the original logical matrix
    shape and uses the AB dgrad path without materializing a transpose.

    Gate and up may intentionally share the same full ``[2I, H]`` data,
    descriptor table, and scale tensor. MoK then applies its existing logical
    gate/up row offsets without repacking that FC1 parameter.
    """

    data: torch.Tensor
    storage_table: torch.Tensor
    scale: torch.Tensor | None = None
    scale_storage_table: torch.Tensor | None = None
    scale_tensors: tuple[torch.Tensor, ...] | None = None
    transposed_data: torch.Tensor | None = None
    transposed_scale: torch.Tensor | None = None
    transposed_storage_table: torch.Tensor | None = None
    transposed_scale_storage_table: torch.Tensor | None = None
    transposed_scale_tensors: tuple[torch.Tensor, ...] | None = None
    native_columnwise: bool = False


@dataclass(frozen=True, slots=True)
class _BF16BackwardWeightArgs:
    """Parsed BF16 routed-weight arguments for ``backward``."""

    gate_data: torch.Tensor
    up_data: torch.Tensor
    down_data: torch.Tensor
    storage_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class _MXFP8BackwardWeightArgs:
    """Parsed MXFP8 routed-weight arguments for ``backward``."""

    gate_row_data: torch.Tensor
    gate_row_scale: torch.Tensor
    gate_column_data: torch.Tensor
    gate_column_scale: torch.Tensor
    up_row_data: torch.Tensor
    up_row_scale: torch.Tensor
    up_column_data: torch.Tensor
    up_column_scale: torch.Tensor
    down_column_data: torch.Tensor
    down_column_scale: torch.Tensor
    native_columnwise: bool
    storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None
    scale_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None


@dataclass(slots=True)
class MoKWorkspace:
    group_name: str                                   # () str
    ep_rank: int                                      # () int
    ep_size: int                                      # () int
    device: torch.device                              # () torch.device
    num_local_tokens: int                             # () int
    hidden_size: int                                  # () int
    topk: int                                         # () int
    schedule_capacity: int                            # () int
    x_buffer: torch.Tensor                            # (num_local_tokens, hidden_size) bfloat16
    x_buffer_handle: Any                              # () SymmetricMemory handle
    x_buffer_ptrs: list[int]                          # (ep_size,) uintptr64
    combine_buffer: torch.Tensor                      # (num_local_tokens * topk, hidden_size) bfloat16
    combine_buffer_handle: Any                        # () SymmetricMemory handle
    combine_buffer_ptrs: list[int]                    # (ep_size,) uintptr64
    d_y_buffer: torch.Tensor                          # (num_local_tokens, hidden_size) bfloat16
    d_y_buffer_handle: Any                            # () SymmetricMemory handle
    d_y_buffer_ptrs: list[int]                        # (ep_size,) uintptr64
    d_x_routed_buffer: torch.Tensor                   # (num_local_tokens * topk, hidden_size) bfloat16
    d_x_routed_buffer_handle: Any                     # () SymmetricMemory handle
    d_x_routed_buffer_ptrs: list[int]                 # (ep_size,) uintptr64
    router_weight_buffer: torch.Tensor                # (num_local_tokens, topk) float32
    router_weight_buffer_handle: Any                  # () SymmetricMemory handle
    router_weight_buffer_ptrs: list[int]              # (ep_size,) uintptr64
    d_router_weight_buffer: torch.Tensor              # (num_local_tokens, topk) float32
    d_router_weight_buffer_handle: Any                # () SymmetricMemory handle
    d_router_weight_buffer_ptrs: list[int]            # (ep_size,) uintptr64
    all_gather_top_experts_buffer: torch.Tensor       # (ep_size, num_local_tokens, topk) int32
    all_gather_top_experts_buffer_handle: Any         # () SymmetricMemory handle
    all_gather_top_experts_buffer_multicast_ptr: int  # () uintptr64
    barrier_buffer: torch.Tensor                      # (1,) int32
    barrier_buffer_handle: Any                        # () SymmetricMemory handle
    barrier_buffer_ptrs: list[int]                    # (ep_size,) uintptr64
    barrier_buffer_multicast_ptr: int                 # () uintptr64
    barrier_target: torch.Tensor                      # (1,) int32


_WORKSPACE_CACHE: dict[tuple[str, int, int, int, int, int], MoKWorkspace] = {}
# get_workspace arguments that already passed validate_workspace_args, mapped to their cached workspace, so a
# repeated lookup skips the device-property and process-group queries. The key holds the arguments by value
# (the group by identity) plus the current CUDA device. A hit is returned only while its group is still registered
# (_group_is_registered): once the group is destroyed, the entry is dropped and the arguments are validated again.
_WORKSPACE_LOOKUP: dict[tuple, MoKWorkspace] = {}
# (config, device, num_local_tokens, topk) combinations whose schedule configuration build_schedule validated.
_VALIDATED_SCHEDULE_CONFIGS: set[tuple] = set()

# Eager dispatch. In eager mode every custom-op call spends host time on a dispatcher round trip and the op's Python
# argument checks, and an MoE forward and backward make a dozen of them, so this module launches the same C++ entry
# points directly when that is safe (_eager_dispatch): only if every tensor operand of the call is a plain torch.Tensor
# or nn.Parameter (activations, routes, router weights, schedule, forward context, weights and scales, descriptor
# tables, main grads, gate weight and gate main grad), and only outside tracers and transforms: not under
# torch.compile, a dispatch mode (FakeTensorMode, make_fx, FunctionalTensorMode, ...), pre-dispatch tracing
# (make_fx(pre_dispatch=True), torch.export) or a torch.func transform (functionalize, vmap, grad, ...). A transform's
# wrapper tensors have the exact type torch.Tensor but hold no data a kernel can use, and inside functionalize even
# the outputs that a C++ entry point allocates for plain operands are such wrappers. Otherwise every C++ launch of the
# call goes through the custom ops in mok.ops, which dispatch through tensor subclasses and which tracers record; under
# a torch.func transform, which none of them supports, the call raises instead (_reject_func_transforms).
# Arguments derived from the workspace and schedule are valid by construction, and validate_inputs checks the caller's
# activations and routes on every call. On the direct path, the caller's weights, main grads and descriptor tables go
# through the custom op (or accumulating wrapper) and its checks on first use; _VALIDATED_LAUNCHES then remembers that
# combination of tensor objects, all of them plain.
#
# Staging. Peers read x, router weights and grad_output from this rank's symmetric workspace buffers through raw
# pointers. Only the direct path reads data pointers: an x or grad_output that is 256-byte aligned stands in for its
# buffer as the shared expert's operand (x_shared), so only its routed rows are copied (_can_stand_in_for_buffer), and
# a misaligned one is copied in full with Tensor.copy_. On the direct path the forward gate and the gate-weight
# gradient read x_shared too, so their cuBLAS results do not depend on x's alignment. A call off the direct path
# copies the inputs in full, at any alignment, through one call of the custom op stage_into_buffers per direction: x
# and router weights in forward; grad_output first, then x and router weights, in backward. Unlike Tensor.copy_ into a
# workspace buffer, which AOTAutograd would apply at the end of a compiled graph, after the kernels that read the
# buffer, the custom op writes in place where it runs, ahead of the barrier in program order. Because the backward
# call takes the incoming gradient, AOTAutograd's partitioner keeps it in the backward graph of a compiled
# autograd.Function; separate copies of x and router weights would not depend on the gradient and could move into the
# forward graph (as the backward's barriers do, see Known limitations).
#
# Gate backward. The fused Triton kernels of a gated backward (_gate_fused) read raw pointers too, but nothing checks
# them. They run on plain operands outside tracers and torch.func transforms, and under torch.compile, which traces
# them; otherwise the reference torch operations they replace run instead (_fused_gate_kernels). These round like the
# kernels, except for the FP32 summation order of the gate gradient.
#
# Accumulation. backward with main_grads (fused weight-gradient accumulation) has no custom op: off the direct path
# it calls the eager accumulating wrappers in mok.ops, which hand their operands to C++ unchanged. So with main_grads,
# backward raises before launching anything if an operand is a tensor subclass (TypeError) or if make_fx,
# FakeTensorMode, FunctionalTensorMode, pre-dispatch tracing or a torch.func transform is active (RuntimeError).
# torch.compile cannot trace the wrappers either: fullgraph=True fails, and otherwise the graph breaks there.
#
# Known limitations, none of them new with the direct path:
# * Order of the MoK ops under torch.compile. The custom ops tie each barrier to the work around it only through
#   data dependencies, and a barrier depends on nothing but the barrier counter: the staging op and the megakernels
#   do not depend on the barrier before them. Inductor keeps the program order of the MoK ops in the graphs that the
#   tests compile, but a pass that reorders independent nodes, such as its peak-memory reordering with large
#   unrelated work in the same graph, can move a barrier past the staging op or a megakernel.
# * A torch.autograd.Function that calls forward and backward with main_grads=None hangs under torch.compile, even
#   with one layer and one microbatch: AOTAutograd's partitioner moves the backward's barriers, which do not depend on
#   the incoming gradient, into the compiled forward graph, where they run on a copy of the barrier state that peers
#   never signal. With main_grads (MCore's pattern), Dynamo breaks the graph at the accumulating backward instead and
#   the call runs correctly (fullgraph=True fails). A compiled function that calls forward and backward directly, in
#   one graph, is not affected.
# * torch.func transforms and torch.jit.trace are unsupported. Under a transform (functionalize, vmap, grad, vjp, jvp,
#   ...), build_schedule, forward, recompute_forward_context and backward raise a RuntimeError before they launch
#   anything: the custom ops implement no transform, and some transforms would otherwise return zero derivatives for
#   MoK's part without an error. torch.jit.trace does not record the direct launches.
#
# _PLAIN_TENSOR_TYPES holds the exact types of a plain operand; NoneType admits an absent optional one.
_PLAIN_TENSOR_TYPES = frozenset((torch.Tensor, torch.nn.Parameter, type(None)))
_dispatch_stack_len = torch._C._len_torch_dispatch_stack
_dispatch_key_included = torch._C._dispatch_tls_is_dispatch_key_included
_PRE_DISPATCH = torch._C.DispatchKey.PreDispatch
_functorch_transforms_active = torch._C._are_functorch_transforms_active


def _eager_dispatch(tensors: tuple[torch.Tensor | None, ...]) -> bool:
    """Whether a call launches MoK kernels through their C++ entry points instead of the custom ops.

    ``tensors`` are all the tensor operands of the call, None for an absent optional one.
    """
    # is_dynamo_compiling() comes first: Dynamo folds it to True and so never traces the queries after it. Pre-dispatch
    # tracing keeps its modes off the dispatch-mode stack and sets the PreDispatch key instead; its modes are entered
    # like every TorchDispatchMode, which sets the Python flag that screens the key query. A torch.func transform
    # neither enters a dispatch mode nor changes the type of its wrapper tensors.
    if (
        torch.compiler.is_dynamo_compiling()
        or _dispatch_stack_len()
        or (_python_dispatch._is_in_torch_dispatch_mode and _dispatch_key_included(_PRE_DISPATCH))
        or _functorch_transforms_active()
    ):
        return False
    return _PLAIN_TENSOR_TYPES.issuperset(map(type, tensors))


def _fused_gate_kernels(*tensors: torch.Tensor) -> bool:
    """Whether a gated backward launches its fused Triton kernels (_gate_fused) on these operands.

    Like the direct C++ launches, the kernels read raw data pointers, so outside torch.compile, whose Dynamo traces
    them, they need plain tensors and no tracer or torch.func transform (the checks of _eager_dispatch). Otherwise the
    reference torch operations they replace run instead, which dispatch through tensor subclasses and which tracers
    record.
    """
    if torch.compiler.is_dynamo_compiling():
        return True
    if (
        _dispatch_stack_len()
        or (_python_dispatch._is_in_torch_dispatch_mode and _dispatch_key_included(_PRE_DISPATCH))
        or _functorch_transforms_active()
    ):
        return False
    return _PLAIN_TENSOR_TYPES.issuperset(map(type, tensors))


def _reject_func_transforms() -> None:
    """Raises a RuntimeError while a torch.func transform is active, before anything is launched.

    No MoK kernel supports a transform (see Known limitations). The direct path rules transforms out already, so only
    the custom-op path calls this; under Dynamo, which folds is_dynamo_compiling() to True, it is a no-op.
    """
    if not torch.compiler.is_dynamo_compiling() and _functorch_transforms_active():
        raise RuntimeError(
            "MoK does not support torch.func transforms (functionalize, vmap, grad, vjp, jvp, ...): build_schedule, "
            "forward, recompute_forward_context and backward raise under one before launching anything"
        )


class _ValidatedLaunches:
    """Kernel argument combinations already validated, keyed by the identities of their tensors.

    An entry keeps weak references to its tensors and is dropped as soon as one of them is freed, so a tensor
    that later reuses a freed tensor's id never matches a stale entry.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, tuple[weakref.ref, ...]] = {}

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries

    def add(self, key: tuple, tensors: tuple[torch.Tensor | None, ...]) -> None:
        entries = self._entries

        def evict(_: weakref.ref) -> None:
            entries.pop(key, None)

        entries[key] = tuple(weakref.ref(tensor, evict) for tensor in tensors if tensor is not None)

    def clear(self) -> None:
        self._entries.clear()


_VALIDATED_LAUNCHES = _ValidatedLaunches()


def _launch(validated_op, entrypoint, args: tuple, tensors: tuple, key: tuple, **kwargs: Any) -> Any:
    """Launch ``entrypoint(*args, **kwargs)``, validating with ``validated_op`` on first use.

    ``validated_op`` checks its arguments and then calls the same entry point. ``tensors`` are the arguments the
    caller has not validated itself (weights, main grads, descriptor tables, workspace buffers); ``key`` holds
    their ids and any other argument the op checks that can differ between calls with the same tensors.
    """
    key = (*key, *map(id, tensors))
    if key in _VALIDATED_LAUNCHES:
        return entrypoint(*args, **kwargs)
    outputs = validated_op(*args, **kwargs)
    _VALIDATED_LAUNCHES.add(key, tensors)
    return outputs


def _all_gather_top_experts(
    workspace: MoKWorkspace, config: MoKConfig, top_experts: torch.Tensor, direct: bool
) -> None:
    if not direct:
        all_gather_top_experts(
            top_experts, workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr, workspace.ep_rank,
            config.all_gather_top_experts_chunk_bytes,
        )
    elif workspace.ep_size == 1:
        workspace.all_gather_top_experts_buffer[0].copy_(top_experts)
    else:
        _C.all_gather_top_experts(
            top_experts, workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr, workspace.ep_rank,
            config.all_gather_top_experts_chunk_bytes,
        )


def _barrier(workspace: MoKWorkspace, direct: bool) -> None:
    if not direct:
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                    workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    elif workspace.ep_size > 1:
        _C.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                       workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)


def _bwd_mxfp8_accum_entrypoint(
    *args: Any,
    main_grads: tuple[torch.Tensor, ...],
    weight_storage_tables: tuple[torch.Tensor, ...] | None = None,
    scale_storage_tables: tuple[torch.Tensor, ...] | None = None,
    main_grad_storage_tables: tuple[torch.Tensor, ...] | None = None,
    shared_grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """The C++ launch of ops.dispatch_mlp_swiglu_combine_bwd_mxfp8_accum, without its argument checks."""
    if weight_storage_tables is None:
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(
            *args, *main_grads, shared_grad_output=shared_grad_output
        )
    else:
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(
            *args, *main_grads, *weight_storage_tables, *scale_storage_tables,
            *main_grad_storage_tables, shared_grad_output=shared_grad_output
        )
    return outputs[:12]


def _bwd_bf16_accum_entrypoint(
    *args: Any,
    main_grads: tuple[torch.Tensor, ...],
    weight_storage_tables: tuple[torch.Tensor, ...] | None = None,
    main_grad_storage_tables: tuple[torch.Tensor, ...] | None = None,
    shared_grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """The C++ launch of ops.dispatch_mlp_swiglu_combine_bwd_bf16_accum, without its argument checks."""
    if weight_storage_tables is None:
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_bf16(
            *args, *main_grads, shared_grad_output=shared_grad_output
        )
    else:
        outputs = _C.dispatch_mlp_swiglu_combine_bwd_bf16(
            *args, *main_grads, *weight_storage_tables, *main_grad_storage_tables,
            shared_grad_output=shared_grad_output,
        )
    return outputs[:9]


def validate_workspace_args(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> None:
    """Validates the arguments used to create or retrieve a workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        None
    """
    if (
        type(config.schedule_capacity_multiplier) not in (int, float)
        or not math.isfinite(config.schedule_capacity_multiplier)
        or config.schedule_capacity_multiplier <= 0
    ):
        raise ValueError("schedule_capacity_multiplier must be a positive finite number")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError("MoK workspace device must be the current CUDA device")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("MoK currently requires an SM100 or SM103 GPU")
    device_properties = torch.cuda.get_device_properties(device)
    if type(num_local_tokens) is not int or num_local_tokens < 512:
        raise ValueError("num_local_tokens must be an integer at least 512")
    if num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be divisible by 256")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if hidden_size < 256 or hidden_size % 128 != 0:
        raise ValueError("hidden_size must be at least 256 and divisible by 128")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    fwd_epilogue_smem_bytes = 2 * ((topk + 1) * 2048 + 2 * topk * 4) + 1024
    if fwd_epilogue_smem_bytes > device_properties.shared_memory_per_block_optin:
        raise ValueError("topk requires more dynamic shared memory than the device supports")

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("MoK EP size must be one of 1, 4, 8, 16, 32, 64")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError("current process is not a member of the EP process group")


def create_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Creates a new caller-owned workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    local_shape = torch.tensor([num_local_tokens, hidden_size, topk], dtype=torch.int64, device=device)
    gathered_shapes = torch.empty(ep_size * local_shape.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(gathered_shapes == local_shape).item():  # .item() here is fine since this is one-time setup
        raise ValueError("MoK requires identical token, hidden, and top-k shapes on every EP rank")
    if ep_size > 1:
        symm_mem.enable_symm_mem_for_group(group_name)

    schedule_capacity = num_local_tokens * topk * schedule_capacity_factor

    def allocate_buffer(*shape: int, dtype: torch.dtype, zero: bool = False) -> tuple[torch.Tensor, Any, list[int]]:
        if ep_size == 1:
            if zero:
                buffer = torch.zeros(*shape, dtype=dtype, device=device)
            else:
                buffer = torch.empty(*shape, dtype=dtype, device=device)
            handle = None
            ptrs = [int(buffer.data_ptr())]
        else:
            buffer = symm_mem.empty(*shape, dtype=dtype, device=device)
            if zero:
                buffer.zero_()
            handle = symm_mem.rendezvous(buffer, group_name)
            ptrs = [int(handle.buffer_ptrs[peer_rank]) for peer_rank in range(ep_size)]
        return buffer, handle, ptrs

    x_buffer, x_buffer_handle, x_buffer_ptrs = allocate_buffer(num_local_tokens, hidden_size, dtype=torch.bfloat16)
    combine_buffer, combine_buffer_handle, combine_buffer_ptrs = allocate_buffer(num_local_tokens * topk, hidden_size, dtype=torch.bfloat16)
    d_y_buffer, d_y_buffer_handle, d_y_buffer_ptrs = allocate_buffer(num_local_tokens, hidden_size, dtype=torch.bfloat16)
    d_x_routed_buffer, d_x_routed_buffer_handle, d_x_routed_buffer_ptrs = allocate_buffer(num_local_tokens * topk, hidden_size, dtype=torch.bfloat16)
    router_weight_buffer, router_weight_buffer_handle, router_weight_buffer_ptrs = allocate_buffer(num_local_tokens, topk, dtype=torch.float32)
    # Zeroed once: the backward kernels write only routed slots, so with zero_unrouted_router_grads=False an
    # unrouted slot reads back zero or a gradient an earlier call wrote there, never uninitialized memory.
    d_router_weight_buffer, d_router_weight_buffer_handle, d_router_weight_buffer_ptrs = allocate_buffer(num_local_tokens, topk, dtype=torch.float32, zero=True)

    all_gather_top_experts_buffer, all_gather_top_experts_buffer_handle, _ = allocate_buffer(ep_size, num_local_tokens, topk, dtype=torch.int32)
    all_gather_top_experts_buffer_multicast_ptr = int(
        all_gather_top_experts_buffer.data_ptr()
        if all_gather_top_experts_buffer_handle is None
        else all_gather_top_experts_buffer_handle.multicast_ptr
    )

    barrier_buffer, barrier_buffer_handle, barrier_buffer_ptrs = allocate_buffer(1, dtype=torch.int32, zero=True)
    barrier_buffer_multicast_ptr = int(
        barrier_buffer.data_ptr()
        if barrier_buffer_handle is None
        else barrier_buffer_handle.multicast_ptr
    )
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)

    dist.barrier(group=group, async_op=True, device_ids=[device_index]).block_current_stream()

    workspace = MoKWorkspace(
        group_name=group_name, ep_rank=ep_rank, ep_size=ep_size, device=device,
        num_local_tokens=num_local_tokens, hidden_size=hidden_size, topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=x_buffer, x_buffer_handle=x_buffer_handle, x_buffer_ptrs=x_buffer_ptrs,
        combine_buffer=combine_buffer, combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        d_y_buffer=d_y_buffer, d_y_buffer_handle=d_y_buffer_handle,
        d_y_buffer_ptrs=d_y_buffer_ptrs,
        d_x_routed_buffer=d_x_routed_buffer, d_x_routed_buffer_handle=d_x_routed_buffer_handle,
        d_x_routed_buffer_ptrs=d_x_routed_buffer_ptrs,
        router_weight_buffer=router_weight_buffer,
        router_weight_buffer_handle=router_weight_buffer_handle,
        router_weight_buffer_ptrs=router_weight_buffer_ptrs,
        d_router_weight_buffer=d_router_weight_buffer,
        d_router_weight_buffer_handle=d_router_weight_buffer_handle,
        d_router_weight_buffer_ptrs=d_router_weight_buffer_ptrs,
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=all_gather_top_experts_buffer_handle,
        all_gather_top_experts_buffer_multicast_ptr=all_gather_top_experts_buffer_multicast_ptr,
        barrier_buffer=barrier_buffer, barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
    )
    return workspace


def _group_is_registered(group: dist.ProcessGroup) -> bool:
    """Whether torch.distributed is initialized and ``group`` is still one of its process groups (not destroyed).

    get_workspace checks this on a lookup hit instead of rerunning validate_workspace_args, whose process-group queries
    raise for a destroyed group. The check reads c10d's group registry, the map destroy_process_group removes a group
    from; a torch without that private registry falls back to the public dist.get_rank, which raises for such a group.
    """
    if not dist.is_initialized():
        return False
    try:
        registered_groups = dist.distributed_c10d._world.pg_map
    except AttributeError:
        try:
            dist.get_rank(group=group)
        except (RuntimeError, ValueError):
            return False
        return True
    return group in registered_groups


def get_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Returns a cached workspace, creating and caching one if absent.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    lookup_key = None
    if (
        type(num_local_tokens) is int and type(hidden_size) is int and type(topk) is int
        and isinstance(config, MoKConfig) and isinstance(device, torch.device)
        and not torch.compiler.is_compiling()
    ):
        lookup_key = (config, group, device, torch.cuda.current_device(), num_local_tokens, hidden_size, topk)
        workspace = _WORKSPACE_LOOKUP.get(lookup_key)
        if workspace is not None:
            if _group_is_registered(group):
                return workspace
            # The group was destroyed after this lookup was recorded (or torch.distributed was shut down): forget the
            # entry, which also releases the group, and validate below, which raises for such a group.
            _WORKSPACE_LOOKUP.pop(lookup_key, None)

    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    cache_key = (group_name, device_index, num_local_tokens, hidden_size, topk, schedule_capacity_factor)
    workspace = _WORKSPACE_CACHE.get(cache_key)
    if workspace is None:
        workspace = create_workspace(
            config,
            group,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_size,
            topk=topk,
        )
        _WORKSPACE_CACHE[cache_key] = workspace
    if lookup_key is not None:
        _WORKSPACE_LOOKUP[lookup_key] = workspace
    return workspace


def clear_workspace_cache() -> None:
    """Clears cached workspaces after all participating ranks synchronize.

    Inputs:
        None

    Outputs:
        None
    """
    for workspace in _WORKSPACE_CACHE.values():
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                    workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        torch.cuda.synchronize(workspace.device)
    _WORKSPACE_CACHE.clear()
    _WORKSPACE_LOOKUP.clear()
    _VALIDATED_SCHEDULE_CONFIGS.clear()
    _VALIDATED_LAUNCHES.clear()


def build_schedule(
    workspace: MoKWorkspace,
    config: MoKConfig,
    top_experts: torch.Tensor,
    *,
    num_local_experts: int,
) -> MoKSchedule:
    """All-gathers routing choices and builds this rank's padded expert schedule.

    Inputs:
        workspace:         MoKWorkspace
        config:            MoKConfig
        top_experts:       int32 or int64 [num_local_tokens, topk]
        num_local_experts: int

    Outputs:
        schedule: MoKSchedule
    """
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    direct = _eager_dispatch((top_experts,))
    if not direct:
        _reject_func_transforms()
        _validate_schedule_config(workspace, config)
    else:
        config_key = (config, workspace.device, workspace.num_local_tokens, workspace.topk)
        if config_key not in _VALIDATED_SCHEDULE_CONFIGS:
            _validate_schedule_config(workspace, config)
            _VALIDATED_SCHEDULE_CONFIGS.add(config_key)
    if not top_experts.is_cuda or top_experts.device != workspace.device:
        raise ValueError("top_experts must be on the workspace CUDA device")
    if top_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_experts must have dtype torch.int32 or torch.int64")
    if not top_experts.is_contiguous():
        raise ValueError("top_experts must be contiguous")
    if tuple(top_experts.shape) != (workspace.num_local_tokens, workspace.topk):
        raise ValueError("top_experts must have shape (num_local_tokens, topk)")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")

    top_experts_int32 = (
        top_experts if top_experts.dtype == torch.int32 else top_experts.to(torch.int32)
    )
    _all_gather_top_experts(workspace, config, top_experts_int32, direct)
    _barrier(workspace, direct)
    schedule_args = (workspace.all_gather_top_experts_buffer, num_local_experts,
                     workspace.schedule_capacity, workspace.ep_rank)
    (schedule_peer_rank, schedule_peer_token_idx,
     num_tokens, tokens_per_expert) = _C.schedule(*schedule_args) if direct else schedule(*schedule_args)
    return MoKSchedule(
        peer_rank=schedule_peer_rank, peer_token_idx=schedule_peer_token_idx,
        num_tokens=num_tokens, tokens_per_expert=tokens_per_expert,
        top_experts=top_experts_int32,
    )


def _validate_schedule_config(workspace: MoKWorkspace, config: MoKConfig) -> None:
    """Validates the scheduling and communication options of ``config`` for ``workspace``'s device and shapes."""
    device_properties = torch.cuda.get_device_properties(workspace.device)
    if type(config.fwd_num_comm_sms) is not int or config.fwd_num_comm_sms <= 0:
        raise ValueError("fwd_num_comm_sms must be a positive integer")
    if config.fwd_num_comm_sms % 2 != 0:
        raise ValueError("fwd_num_comm_sms must be even")
    if type(config.bwd_num_comm_sms) is not int or config.bwd_num_comm_sms <= 0:
        raise ValueError("bwd_num_comm_sms must be a positive integer")
    if config.bwd_num_comm_sms % 2 != 0:
        raise ValueError("bwd_num_comm_sms must be even")
    if (config.fwd_num_comm_sms >= device_properties.multi_processor_count
            or config.bwd_num_comm_sms >= device_properties.multi_processor_count):
        raise ValueError("communication-SM counts must leave at least one compute SM")
    if (type(config.minibatch_size) is not int or config.minibatch_size <= 0
            or config.minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(config.macrobatch_size) is not int or config.macrobatch_size <= 0
            or config.macrobatch_size % config.minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if (type(config.all_gather_top_experts_chunk_bytes) is not int
            or config.all_gather_top_experts_chunk_bytes <= 0
            or config.all_gather_top_experts_chunk_bytes % 16 != 0):
        raise ValueError("all_gather_top_experts_chunk_bytes must be a positive multiple of 16")
    if (config.all_gather_top_experts_chunk_bytes + 1024
            > device_properties.shared_memory_per_block_optin):
        raise ValueError("all_gather_top_experts_chunk_bytes exceeds device dynamic shared-memory capacity")
    route_buffer_bytes = workspace.num_local_tokens * workspace.topk * 4
    if route_buffer_bytes % config.all_gather_top_experts_chunk_bytes != 0:
        raise ValueError("all_gather_top_experts_chunk_bytes must divide one rank's route-buffer bytes")


def validate_inputs(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor | None = None,
    grad_output: torch.Tensor | None = None,
) -> None:
    """Validates runtime inputs against the workspace and schedule.

    Inputs:
        config:         MoKConfig
        workspace:      MoKWorkspace
        schedule:       MoKSchedule
        x:              bfloat16 [num_local_tokens, hidden_size]
        router_weights: float32 [num_local_tokens, topk] | None
        grad_output:    bfloat16 [num_local_tokens, hidden_size] | None

    Outputs:
        None
    """
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    expected_activation_shape = (workspace.num_local_tokens, workspace.hidden_size)
    tensors = [("x", x, torch.bfloat16, expected_activation_shape)]
    if grad_output is not None:
        tensors.append(("grad_output", grad_output, torch.bfloat16, expected_activation_shape))
    if router_weights is not None:
        tensors.append(("router_weights", router_weights, torch.float32, (workspace.num_local_tokens, workspace.topk)))
    tensors.append(("schedule.top_experts", schedule.top_experts, torch.int32,
                    (workspace.num_local_tokens, workspace.topk)))
    for tensor_name, tensor, expected_dtype, expected_shape in tensors:
        if not tensor.is_cuda or tensor.device != workspace.device or tensor.dtype != expected_dtype or not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous {expected_dtype} on the workspace CUDA device")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} shape does not match the workspace")
    if schedule.peer_rank.numel() != workspace.schedule_capacity:
        raise ValueError("schedule capacity does not match the workspace")


def _can_stand_in_for_buffer(tensor: torch.Tensor, direct: bool) -> bool:
    """Whether a validated caller activation can replace its workspace buffer as a kernel's local operand.

    The megakernels read the local operand (x, or grad_output via shared_grad_output) only for the shared expert,
    through TMA descriptors; peers read the symmetric buffers only at routed rows, through the buffer pointers.
    Only a direct launch reads the data pointer: under a tracer the pointer is unknown, unreadable (a fake tensor has
    no data) or not the one the traced graph will run with, and a tensor subclass's pointer need not address its data,
    so such calls always stage full copies (_stage_in_full).
    """
    return direct and tensor.data_ptr() % 256 == 0


def _stage_in_full(direct: bool, sources: list[torch.Tensor], buffers: list[torch.Tensor]) -> None:
    """Copies each source in full into its workspace buffer, at any alignment.

    Off the direct path the copies are one call of the stage_into_buffers custom op, which declares its writes: under
    torch.compile, Tensor.copy_ into a workspace buffer would run at the end of the compiled graph, after the barrier
    and the kernels whose peers read the buffer.
    """
    if direct:
        for source, buffer in zip(sources, buffers):
            buffer.copy_(source)
    else:
        stage_into_buffers(sources, buffers)


def _stage_forward_inputs(
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    direct: bool,
) -> torch.Tensor:
    """Fills the symmetric buffers peers read in forward and returns the shared-expert operand.

    On the direct path with an aligned x, only rows with a route are copied into x_buffer and the shared expert reads
    x itself; the other x_buffer rows keep stale data that no kernel reads. Otherwise x is copied in full and the
    shared expert reads x_buffer.
    """
    if _can_stand_in_for_buffer(x, direct):
        _C.copy_routed_rows(schedule.top_experts, [x], [workspace.x_buffer],
                            router_weights, workspace.router_weight_buffer)
        return x
    _stage_in_full(direct, [x, router_weights], [workspace.x_buffer, workspace.router_weight_buffer])
    return workspace.x_buffer


def _gate_bwd_pre_reference(
    grad_output: torch.Tensor, shared_output: torch.Tensor, gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The torch operations that _gate_fused.gate_bwd_pre replaces: d_shared = dY * G and d_z = sigmoid_backward(dG, G)
    with dG = (dY * S).sum(-1), at the same BF16 tensor boundaries."""
    d_shared = grad_output * gate
    d_gate = (grad_output * shared_output).sum(-1, keepdim=True)
    return d_shared, torch.ops.aten.sigmoid_backward.default(d_gate, gate)


def _validate_shared_output_gate_weight(weight: torch.Tensor, x: torch.Tensor) -> None:
    """Checks the output-gate weight that forward and backward take: contiguous BF16 [1, H] on x's device."""
    if (
        weight.dtype != torch.bfloat16
        or weight.device != x.device
        or tuple(weight.shape) != (1, x.shape[1])
        or not weight.is_contiguous()
    ):
        raise ValueError("shared_output_gate_weight must be contiguous BF16 [1, H] on the input device")


def _shared_output_gate_wgrad(
    d_z: torch.Tensor, x: torch.Tensor, main_grad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute an FP32 gate-weight gradient from BF16 d_z and x.

    If main_grad is supplied, add the contribution in place and return that
    buffer. A BF16 buffer rounds after the addition, not before it.
    """
    contribution = torch.mm(d_z.T, x, out_dtype=torch.float32)
    if main_grad is None:
        return contribution
    main_grad.add_(contribution)
    return main_grad


def forward(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | SplitRoutedWeight,
    routed_up_weights: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | SplitRoutedWeight,
    routed_down_weights: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | SplitRoutedWeight,
    swiglu_limit: float | None = None,
    *,
    shared_output_gate_weight: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    MoKForwardContext,
]:
    """Runs the MoE forward pass.

    Routed weights use one of three representations:
      * ``torch.Tensor``: contiguous single-grouped BF16 payload.
      * ``(data, scale)``: contiguous single-grouped MXFP8 forward payload.
      * ``SplitRoutedWeight``: descriptor-backed non-single BF16 or MXFP8
        payloads stored in independent per-expert allocations.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 representation
        swiglu_limit:        float | None
        shared_output_gate_weight: optional BF16 [1, H] output gate for BF16 or MXFP8 routed experts

    Outputs:
        output:          bfloat16 [num_local_tokens, hidden_size]
        forward_context: MoKForwardContext
    """
    validate_inputs(config, workspace, schedule, x, router_weights)
    if shared_output_gate_weight is not None:
        _validate_shared_output_gate_weight(shared_output_gate_weight, x)

    split_weights = isinstance(routed_gate_weights, SplitRoutedWeight)
    if split_weights != isinstance(
        routed_up_weights, SplitRoutedWeight
    ) or split_weights != isinstance(routed_down_weights, SplitRoutedWeight):
        raise TypeError(
            "routed gate/up/down weights must use the same storage representation"
        )
    routed_precision_is_mxfp8 = (
        routed_gate_weights.scale is not None
        if split_weights
        else isinstance(routed_gate_weights, tuple)
    )
    if split_weights:
        split_triplet = (routed_gate_weights, routed_up_weights, routed_down_weights)
        if any(
            (weight.scale is not None) != routed_precision_is_mxfp8
            for weight in split_triplet
        ):
            raise ValueError("split routed gate/up/down weights must use one precision")

    if routed_precision_is_mxfp8:
        if split_weights:
            if any(weight.scale_storage_table is None for weight in split_triplet):
                raise ValueError(
                    "all split MXFP8 routed weights must provide scale descriptor tables"
                )
            routed_gate_weights_fp8 = routed_gate_weights.data
            routed_gate_weights_sc = routed_gate_weights.scale
            routed_up_weights_fp8 = routed_up_weights.data
            routed_up_weights_sc = routed_up_weights.scale
            routed_down_weights_fp8 = routed_down_weights.data
            routed_down_weights_sc = routed_down_weights.scale
            routed_storage_tables = (
                routed_gate_weights.storage_table,
                routed_up_weights.storage_table,
                routed_down_weights.storage_table,
            )
            routed_scale_storage_tables = (
                routed_gate_weights.scale_storage_table,
                routed_up_weights.scale_storage_table,
                routed_down_weights.scale_storage_table,
            )
        else:
            routed_gate_weights_fp8, routed_gate_weights_sc = routed_gate_weights
            routed_up_weights_fp8, routed_up_weights_sc = routed_up_weights
            routed_down_weights_fp8, routed_down_weights_sc = routed_down_weights
            routed_storage_tables = (None, None, None)
            routed_scale_storage_tables = (None, None, None)
        fwd_weights = (
            shared_gate_weights,
            routed_gate_weights_fp8,
            routed_gate_weights_sc,
            shared_up_weights,
            routed_up_weights_fp8,
            routed_up_weights_sc,
            shared_down_weights,
            routed_down_weights_fp8,
            routed_down_weights_sc,
        )
        routed_tables = (*routed_storage_tables, *routed_scale_storage_tables)
    else:
        if split_weights:
            routed_gate_data = routed_gate_weights.data
            routed_up_data = routed_up_weights.data
            routed_down_data = routed_down_weights.data
            routed_storage_tables = (
                routed_gate_weights.storage_table,
                routed_up_weights.storage_table,
                routed_down_weights.storage_table,
            )
        else:
            routed_gate_data = routed_gate_weights
            routed_up_data = routed_up_weights
            routed_down_data = routed_down_weights
            routed_storage_tables = (None, None, None)
        fwd_weights = (
            shared_gate_weights,
            routed_gate_data,
            shared_up_weights,
            routed_up_data,
            shared_down_weights,
            routed_down_data,
        )
        routed_tables = routed_storage_tables

    # One dispatch decision covers every launch of this call (see _eager_dispatch).
    direct = _eager_dispatch((
        x, router_weights, shared_output_gate_weight, schedule.top_experts, schedule.peer_rank,
        schedule.peer_token_idx, schedule.num_tokens, schedule.tokens_per_expert, *fwd_weights, *routed_tables,
    ))
    if not direct:
        _reject_func_transforms()
    x_shared = _stage_forward_inputs(workspace, schedule, x, router_weights, direct)
    _barrier(workspace, direct)

    shared_output_gate = None
    if shared_output_gate_weight is not None:
        # This functional API has a manual backward; do not retain an inner
        # autograd graph (and Z) when called with trainable X or weights.
        # On the direct path the gate reads x_shared, like the shared expert:
        # x if it is 256-byte aligned and its staged copy otherwise, so the
        # cuBLAS result does not depend on x's alignment.
        with torch.no_grad():
            shared_output_gate = torch.sigmoid(
                torch.nn.functional.linear(x_shared if direct else x, shared_output_gate_weight)
            )

    fwd_args = (
        x_shared,
        workspace.x_buffer_ptrs,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        *fwd_weights,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        workspace.topk,
        swiglu_limit,
        config.fwd_num_comm_sms,
        config.macrobatch_size,
        config.minibatch_size,
        *routed_tables,
    )
    if routed_precision_is_mxfp8:
        if direct:
            fwd_outputs = _launch(
                dispatch_mlp_swiglu_combine_fwd_mxfp8,
                _C.dispatch_mlp_swiglu_combine_fwd_mxfp8,
                fwd_args,
                (workspace.combine_buffer, *fwd_weights, *routed_tables),
                ("fwd_mxfp8", config, swiglu_limit, schedule.tokens_per_expert.shape),
            )
        else:
            fwd_outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(*fwd_args)
        (
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            up_shared,
            up_fp8_routed,
            up_sc_routed,
            hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            y_shared,
            y_routed,
        ) = fwd_outputs
        forward_context = MoKForwardContext(
            x_routed=(x_fp8_t_routed, x_sc_t_routed),
            gate_shared=gate_shared,
            gate_routed=(gate_fp8_routed, gate_sc_routed),
            up_shared=up_shared,
            up_routed=(up_fp8_routed, up_sc_routed),
            hidden_shared=hidden_shared,
            hidden_routed=(hidden_fp8_t_routed, hidden_sc_t_routed),
            shared_output=y_shared if shared_output_gate is not None else None,
            shared_output_gate=shared_output_gate,
        )
    else:
        if direct:
            fwd_outputs = _launch(
                dispatch_mlp_swiglu_combine_fwd_bf16,
                _C.dispatch_mlp_swiglu_combine_fwd_bf16,
                fwd_args,
                (workspace.combine_buffer, *fwd_weights, *routed_tables),
                ("fwd_bf16", config, swiglu_limit, schedule.tokens_per_expert.shape),
            )
        else:
            fwd_outputs = dispatch_mlp_swiglu_combine_fwd_bf16(*fwd_args)
        (
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
            y_shared,
            y_routed,
        ) = fwd_outputs
        forward_context = MoKForwardContext(
            x_routed=x_routed,
            gate_shared=gate_shared,
            gate_routed=gate_routed,
            up_shared=up_shared,
            up_routed=up_routed,
            hidden_shared=hidden_shared,
            hidden_routed=hidden_routed,
            shared_output=y_shared if shared_output_gate is not None else None,
            shared_output_gate=shared_output_gate,
        )

    _barrier(workspace, direct)
    if direct:
        # y_shared and the gate come from the kernels above; the workspace and routes are validated.
        if shared_output_gate is None:
            output = _C.fwd_epilogue(y_shared, workspace.combine_buffer,
                                     workspace.router_weight_buffer, schedule.top_experts)
        else:
            output = _C.fwd_epilogue(
                y_shared, workspace.combine_buffer, workspace.router_weight_buffer,
                schedule.top_experts, shared_output_gate=shared_output_gate,
            )
    elif shared_output_gate is None:
        output = fwd_epilogue(y_shared, workspace.combine_buffer,
                              workspace.router_weight_buffer, schedule.top_experts)
    else:
        output = fwd_epilogue(
            y_shared, workspace.combine_buffer, workspace.router_weight_buffer,
            schedule.top_experts, shared_output_gate=shared_output_gate,
        )
    return output, forward_context


def recompute_forward_context(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
) -> MoKForwardContext:
    """Recomputes the intermediates needed by the MoK backward pass.

    ``SplitRoutedWeight`` is intentionally unsupported here because the
    recompute custom ops do not yet accept per-expert descriptor tables.
    This API remains ungated: it lacks the shared down and output-gate weights
    needed to recreate S/G for gated backward.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        swiglu_limit:        float | None

    Outputs:
        forward_context: MoKForwardContext
    """
    validate_inputs(config, workspace, schedule, x)
    # Fail explicitly instead of passing a descriptor-backed object to the
    # contiguous BF16 custom op.
    if isinstance(routed_gate_weights, SplitRoutedWeight) or isinstance(
        routed_up_weights, SplitRoutedWeight
    ):
        raise TypeError("recompute_forward_context does not support SplitRoutedWeight")
    if isinstance(routed_gate_weights, tuple) != isinstance(routed_up_weights, tuple):
        raise TypeError("routed gate and up weights must use the same precision representation")
    if isinstance(routed_gate_weights, tuple) and (len(routed_gate_weights) != 2 or len(routed_up_weights) != 2):
        raise ValueError("MXFP8 routed gate and up weights must be data/scale pairs")

    # The barrier and the recompute kernels always launch through the custom ops. Only the staging copy depends on
    # the path: a plain Tensor.copy_ on the direct path, otherwise the staging op, which keeps the write ahead of the
    # barrier under torch.compile (see _stage_in_full).
    if isinstance(routed_gate_weights, tuple):
        routed_weights = (*routed_gate_weights, *routed_up_weights)
    else:
        routed_weights = (routed_gate_weights, routed_up_weights)
    direct = _eager_dispatch((
        x, shared_gate_weights, shared_up_weights, *routed_weights, schedule.peer_rank, schedule.peer_token_idx,
        schedule.num_tokens, schedule.tokens_per_expert,
    ))
    if not direct:
        _reject_func_transforms()
    _stage_in_full(direct, [x], [workspace.x_buffer])
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

    if isinstance(routed_gate_weights, tuple):
        routed_gate_weights_fp8, routed_gate_weights_sc = routed_gate_weights
        routed_up_weights_fp8, routed_up_weights_sc = routed_up_weights
        (x_fp8_t_routed, x_sc_t_routed,
         gate_shared, gate_fp8_routed, gate_sc_routed,
         up_shared, up_fp8_routed, up_sc_routed,
         hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed) = recompute_forward_context_mxfp8(
            workspace.x_buffer, workspace.x_buffer_ptrs,
            shared_gate_weights, routed_gate_weights_fp8, routed_gate_weights_sc,
            shared_up_weights, routed_up_weights_fp8, routed_up_weights_sc,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.fwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )
        x_routed = (x_fp8_t_routed, x_sc_t_routed)
        gate_routed = (gate_fp8_routed, gate_sc_routed)
        up_routed = (up_fp8_routed, up_sc_routed)
        hidden_routed = (hidden_fp8_t_routed, hidden_sc_t_routed)
    else:
        (x_routed, gate_shared, gate_routed, up_shared, up_routed,
         hidden_shared, hidden_routed) = recompute_forward_context_bf16(
            workspace.x_buffer, workspace.x_buffer_ptrs,
            shared_gate_weights, routed_gate_weights,
            shared_up_weights, routed_up_weights,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.fwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )

    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    return MoKForwardContext(
        x_routed=x_routed,
        gate_shared=gate_shared,
        gate_routed=gate_routed,
        up_shared=up_shared,
        up_routed=up_routed,
        hidden_shared=hidden_shared,
        hidden_routed=hidden_routed,
    )


def _parse_backward_weight_arguments(
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    main_grads: tuple[torch.Tensor, ...] | None,
    main_grad_storage_tables: tuple[torch.Tensor, ...] | None,
) -> _BF16BackwardWeightArgs | _MXFP8BackwardWeightArgs:
    """Parse all public routed-weight encodings used by ``backward``.

    Tuple-length compatibility is intentionally confined to this boundary.
    The backward implementation below consumes only named fields.
    """
    split_weights = isinstance(routed_gate_weights, SplitRoutedWeight)
    if split_weights != isinstance(
        routed_up_weights, SplitRoutedWeight
    ) or split_weights != isinstance(routed_down_weights, SplitRoutedWeight):
        raise TypeError(
            "routed gate/up/down weights must use the same storage representation"
        )
    if split_weights and main_grads is None:
        raise ValueError("split routed weights require fused main-grad accumulation")
    if (main_grad_storage_tables is not None) != split_weights:
        raise ValueError(
            "split routed weights require three split main-grad storage tables"
        )

    if split_weights:
        split_gate = routed_gate_weights
        split_up = routed_up_weights
        split_down = routed_down_weights
        split_triplet = (split_gate, split_up, split_down)
        split_is_mxfp8 = split_gate.scale is not None
        if not split_is_mxfp8:
            if any(weight.scale is not None for weight in split_triplet):
                raise ValueError("split BF16 routed weights must not provide scales")
            return _BF16BackwardWeightArgs(
                gate_data=split_gate.data,
                up_data=split_up.data,
                down_data=split_down.data,
                storage_tables=(
                    split_gate.storage_table,
                    split_up.storage_table,
                    split_down.storage_table,
                ),
            )

        if any(weight.scale is None for weight in split_triplet):
            raise ValueError("all split MXFP8 routed weights must provide scales")
        if any(
            weight.transposed_data is None
            or weight.transposed_scale is None
            or weight.transposed_storage_table is None
            or weight.scale_storage_table is None
            or weight.transposed_scale_storage_table is None
            for weight in split_triplet
        ):
            raise ValueError(
                "split MXFP8 routed weights require rowwise/columnwise data, "
                "scales, and descriptors"
            )
        native_columnwise_values = {
            weight.native_columnwise for weight in split_triplet
        }
        if len(native_columnwise_values) != 1:
            raise ValueError(
                "split MXFP8 routed weights must agree on columnwise layout"
            )
        return _MXFP8BackwardWeightArgs(
            gate_row_data=split_gate.data,
            gate_row_scale=split_gate.scale,
            gate_column_data=split_gate.transposed_data,
            gate_column_scale=split_gate.transposed_scale,
            up_row_data=split_up.data,
            up_row_scale=split_up.scale,
            up_column_data=split_up.transposed_data,
            up_column_scale=split_up.transposed_scale,
            down_column_data=split_down.transposed_data,
            down_column_scale=split_down.transposed_scale,
            native_columnwise=split_gate.native_columnwise,
            storage_tables=(
                split_gate.storage_table,
                split_up.storage_table,
                split_gate.transposed_storage_table,
                split_up.transposed_storage_table,
                split_down.transposed_storage_table,
            ),
            scale_storage_tables=(
                split_gate.scale_storage_table,
                split_up.scale_storage_table,
                split_gate.transposed_scale_storage_table,
                split_up.transposed_scale_storage_table,
                split_down.transposed_scale_storage_table,
            ),
        )

    tuple_weights = isinstance(routed_gate_weights, tuple)
    if tuple_weights != isinstance(
        routed_up_weights, tuple
    ) or tuple_weights != isinstance(routed_down_weights, tuple):
        raise TypeError(
            "routed gate/up/down weights must use the same storage representation"
        )
    if not tuple_weights:
        if not all(
            isinstance(weight, torch.Tensor)
            for weight in (routed_gate_weights, routed_up_weights, routed_down_weights)
        ):
            raise TypeError("contiguous BF16 routed weights must be torch.Tensor objects")
        return _BF16BackwardWeightArgs(
            gate_data=routed_gate_weights,
            up_data=routed_up_weights,
            down_data=routed_down_weights,
            storage_tables=None,
        )

    if len(routed_gate_weights) == 5:
        # Single-grouped MXFP8 with TE-native columnwise payloads:
        # gate/up=(row, row_scale, column, column_scale, True),
        # down=(column, column_scale, True).
        if len(routed_up_weights) != 5 or len(routed_down_weights) != 3:
            raise ValueError(
                "native-columnwise MXFP8 gate/up/down tuples must have lengths 5/5/3"
            )
        (
            gate_row_data,
            gate_row_scale,
            gate_column_data,
            gate_column_scale,
            gate_native_columnwise,
        ) = routed_gate_weights
        (
            up_row_data,
            up_row_scale,
            up_column_data,
            up_column_scale,
            up_native_columnwise,
        ) = routed_up_weights
        (
            down_column_data,
            down_column_scale,
            down_native_columnwise,
        ) = routed_down_weights
        if (
            gate_native_columnwise is not True
            or up_native_columnwise is not True
            or down_native_columnwise is not True
        ):
            raise ValueError("all native-columnwise MXFP8 flags must be True")
        return _MXFP8BackwardWeightArgs(
            gate_row_data=gate_row_data,
            gate_row_scale=gate_row_scale,
            gate_column_data=gate_column_data,
            gate_column_scale=gate_column_scale,
            up_row_data=up_row_data,
            up_row_scale=up_row_scale,
            up_column_data=up_column_data,
            up_column_scale=up_column_scale,
            down_column_data=down_column_data,
            down_column_scale=down_column_scale,
            native_columnwise=True,
            storage_tables=None,
            scale_storage_tables=None,
        )

    if len(routed_gate_weights) == 4:
        # Legacy single-grouped MXFP8 with explicitly transposed payloads:
        # gate/up=(row, row_scale, transposed, transposed_scale),
        # down=(transposed, transposed_scale).
        if len(routed_up_weights) != 4 or len(routed_down_weights) != 2:
            raise ValueError(
                "legacy MXFP8 gate/up/down tuples must have lengths 4/4/2"
            )
        (
            gate_row_data,
            gate_row_scale,
            gate_column_data,
            gate_column_scale,
        ) = routed_gate_weights
        (
            up_row_data,
            up_row_scale,
            up_column_data,
            up_column_scale,
        ) = routed_up_weights
        down_column_data, down_column_scale = routed_down_weights
        return _MXFP8BackwardWeightArgs(
            gate_row_data=gate_row_data,
            gate_row_scale=gate_row_scale,
            gate_column_data=gate_column_data,
            gate_column_scale=gate_column_scale,
            up_row_data=up_row_data,
            up_row_scale=up_row_scale,
            up_column_data=up_column_data,
            up_column_scale=up_column_scale,
            down_column_data=down_column_data,
            down_column_scale=down_column_scale,
            native_columnwise=False,
            storage_tables=None,
            scale_storage_tables=None,
        )

    raise ValueError("MXFP8 gate weight tuple must contain four or five values")


def backward(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    forward_context: MoKForwardContext,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    swiglu_limit: float | None = None,
    main_grads: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    main_grad_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    *,
    shared_output_gate_weight: torch.Tensor | None = None,
    shared_output_gate_main_grad: torch.Tensor | None = None,
    zero_unrouted_router_grads: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Runs the MoE backward pass.

    Contiguous BF16 weights use ``torch.Tensor``. Contiguous MXFP8 gate/up
    weights use either the legacy four-tensor tuple or the native-columnwise
    five-item tuple; down uses the corresponding two- or three-item tuple.
    Descriptor-backed non-single BF16/MXFP8 weights use
    ``SplitRoutedWeight``.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        forward_context:     MoKForwardContext
        grad_output:         bfloat16 [num_local_tokens, hidden_size]
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 tensor tuple
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 tensor tuple
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 tensor tuple
        swiglu_limit:        float | None
        main_grads:          optional BF16/FP32 buffers in MOK ABI order:
                             shared gate, routed gate, shared up, routed up,
                             shared down, routed down. If provided, wgrad is
                             accumulated in-place instead of returned freshly.
        main_grad_storage_tables:
                             None for contiguous single-grouped main grads;
                             for non-single weights, descriptor tables ordered
                             as routed gate, routed up, routed down. Gate/up may
                             share the same combined-FC1 table.
        shared_output_gate_weight: the same unchanged BF16 [1, H] weight passed to forward, or None
        shared_output_gate_main_grad: optional BF16/FP32 [1, H] additive accumulation buffer
        zero_unrouted_router_grads: zero d_router_weights where top_experts is -1 (default). A caller that
                             reads d_router_weights only at routed slots may pass False to skip that masking;
                             the unrouted entries are then unspecified (zero, or a gradient an earlier call
                             computed for that slot).

    Outputs:
        The six returned weight-gradient entries are fresh gradients when
        ``main_grads is None``; otherwise they alias the supplied accumulation
        buffers. For split weights, routed entries are representative tensors
        while the storage tables address every expert buffer.

        d_x:                   bfloat16 [num_local_tokens, hidden_size]
        d_router_weights:      float32 [num_local_tokens, topk]
        d_routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size]
        d_shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        d_shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        d_shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        d_shared_output_gate_weight: fresh float32 [1, H], or an alias of the
                                    BF16/FP32 accumulation buffer; None for ungated
    """
    validate_inputs(config, workspace, schedule, x, router_weights, grad_output)
    if not isinstance(forward_context, MoKForwardContext):
        raise TypeError("forward_context must be a MoKForwardContext")

    gate = forward_context.shared_output_gate
    shared_output = forward_context.shared_output
    if shared_output_gate_weight is None:
        if gate is not None or shared_output is not None:
            raise ValueError("a gated forward context requires shared_output_gate_weight in backward")
        if shared_output_gate_main_grad is not None:
            raise ValueError("shared output gate main-grad requires shared_output_gate_weight")
    else:
        if gate is None or shared_output is None:
            raise ValueError("gated backward requires saved S/G from the original gated forward context")
        # Checked again here, before any launch: the fused gate kernels read the weight as BF16 [1, H] unchecked.
        # Their other operands are valid by construction: grad_output passed validate_inputs, S and G come from the
        # gated forward, and d_z and d_x are fresh outputs of the operations below.
        _validate_shared_output_gate_weight(shared_output_gate_weight, x)
        if shared_output_gate_main_grad is not None and (
            shared_output_gate_main_grad.dtype not in (torch.bfloat16, torch.float32)
            or shared_output_gate_main_grad.device != x.device
            or tuple(shared_output_gate_main_grad.shape) != (1, x.shape[1])
            or not shared_output_gate_main_grad.is_contiguous()
        ):
            raise ValueError("shared output gate main-grad must be contiguous BF16/FP32 [1, H] on the input device")

    weight_args = _parse_backward_weight_arguments(
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        main_grads,
        main_grad_storage_tables,
    )
    if isinstance(weight_args, _MXFP8BackwardWeightArgs):
        x_fp8_t_routed, x_sc_t_routed = forward_context.x_routed
        gate_fp8_routed, gate_sc_routed = forward_context.gate_routed
        up_fp8_routed, up_sc_routed = forward_context.up_routed
        hidden_fp8_t_routed, hidden_sc_t_routed = forward_context.hidden_routed
        routed_context = (
            x_fp8_t_routed, x_sc_t_routed, gate_fp8_routed, gate_sc_routed,
            up_fp8_routed, up_sc_routed, hidden_fp8_t_routed, hidden_sc_t_routed,
        )
        bwd_weights = (
            shared_gate_weights,
            weight_args.gate_column_data,
            weight_args.gate_column_scale,
            shared_up_weights,
            weight_args.up_column_data,
            weight_args.up_column_scale,
            shared_down_weights,
            weight_args.down_column_data,
            weight_args.down_column_scale,
            weight_args.gate_row_data,
            weight_args.gate_row_scale,
            weight_args.up_row_data,
            weight_args.up_row_scale,
        )
        weight_tables = (*(weight_args.storage_tables or ()), *(weight_args.scale_storage_tables or ()))
    else:
        routed_context = (
            forward_context.x_routed, forward_context.gate_routed,
            forward_context.up_routed, forward_context.hidden_routed,
        )
        bwd_weights = (
            shared_gate_weights,
            weight_args.gate_data,
            shared_up_weights,
            weight_args.up_data,
            shared_down_weights,
            weight_args.down_data,
        )
        weight_tables = weight_args.storage_tables or ()

    # One dispatch decision covers every C++ launch of this call (see _eager_dispatch).
    operands = (
        grad_output, x, router_weights, schedule.top_experts, schedule.peer_rank, schedule.peer_token_idx,
        schedule.num_tokens, schedule.tokens_per_expert, *bwd_weights, *weight_tables, *(main_grads or ()),
        *(main_grad_storage_tables or ()), *routed_context, forward_context.gate_shared, forward_context.up_shared,
        forward_context.hidden_shared, shared_output, gate, shared_output_gate_weight, shared_output_gate_main_grad,
    )
    direct = _eager_dispatch(operands)
    if main_grads is not None and not direct and not torch.compiler.is_dynamo_compiling():
        # The accumulating backward has no custom op, so fail before launching anything (see the module comment).
        _check_accumulation_operands(operands)
    if not direct:
        _reject_func_transforms()

    shared_grad_kwargs = {}
    # The direct path implies the conditions of the fused gate kernels: plain operands, no tracer and no transform.
    fused_gate = gate is not None and (direct or _fused_gate_kernels(grad_output, shared_output, gate))
    if gate is not None:
        with torch.no_grad():
            # Native-style BF16 tensor boundaries: dS = dY * G uses FP32 opmath, dG sums BF16 products with an FP32
            # accumulator, and dZ = sigmoid_backward(dG, G) rounds like aten on BF16. The fused kernel makes one pass
            # over the caller's full grad_output and S; it differs from the reference operations only in the FP32
            # summation order of dG (see _gate_fused).
            if fused_gate:
                d_shared, d_z = _gate_fused._launch_gate_bwd_pre(grad_output, shared_output, gate)
            else:
                d_shared, d_z = _gate_bwd_pre_reference(grad_output, shared_output, gate)
        shared_grad_kwargs["shared_grad_output"] = d_shared

    if _can_stand_in_for_buffer(grad_output, direct) and _can_stand_in_for_buffer(x, direct):
        # Peers read d_y_buffer and x_buffer only at routed rows; the shared expert reads grad_output (or the gated
        # d_shared) and x themselves.
        _C.copy_routed_rows(schedule.top_experts, [grad_output, x], [workspace.d_y_buffer, workspace.x_buffer],
                            router_weights, workspace.router_weight_buffer)
        shared_grad_kwargs.setdefault("shared_grad_output", grad_output)
        x_shared = x
    else:
        # One staging call that takes grad_output first (see the module comment).
        _stage_in_full(
            direct, [grad_output, x, router_weights],
            [workspace.d_y_buffer, workspace.x_buffer, workspace.router_weight_buffer],
        )
        x_shared = workspace.x_buffer
    _barrier(workspace, direct)
    if isinstance(weight_args, _MXFP8BackwardWeightArgs):
        mxfp8_bwd_args = (
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights,
            weight_args.gate_column_data,
            weight_args.gate_column_scale,
            shared_up_weights,
            weight_args.up_column_data,
            weight_args.up_column_scale,
            shared_down_weights,
            weight_args.down_column_data,
            weight_args.down_column_scale,
            x_fp8_t_routed,
            x_sc_t_routed,
            forward_context.gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            forward_context.up_shared,
            up_fp8_routed,
            up_sc_routed,
            forward_context.hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            x_shared,
            workspace.x_buffer_ptrs,
            weight_args.gate_row_data,
            weight_args.gate_row_scale,
            weight_args.up_row_data,
            weight_args.up_row_scale,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            weight_args.native_columnwise,
        )
        if main_grads is None:
            # Original MOK API: materialize and return six fresh weight gradients.
            if direct:
                bwd_outputs = _launch(
                    dispatch_mlp_swiglu_combine_bwd_mxfp8,
                    _C.dispatch_mlp_swiglu_combine_bwd_mxfp8,
                    mxfp8_bwd_args,
                    (workspace.d_y_buffer, *bwd_weights),
                    ("bwd_mxfp8", config, swiglu_limit, schedule.tokens_per_expert.shape,
                     weight_args.native_columnwise),
                    **shared_grad_kwargs,
                )
            else:
                bwd_outputs = dispatch_mlp_swiglu_combine_bwd_mxfp8(*mxfp8_bwd_args, **shared_grad_kwargs)
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_fp8_routed,
                d_gate_sc_routed,
                d_up_shared,
                d_up_fp8_routed,
                d_up_sc_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_fp8_routed,
                d_y_sc_routed,
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_down,
                d_w_routed_down,
            ) = bwd_outputs
        else:
            # Fused accumulation: mutate the six supplied main-grad buffers.
            # Descriptor tables are present only for non-single expert storage.
            accum_kwargs = dict(
                main_grads=main_grads,
                weight_storage_tables=weight_args.storage_tables,
                scale_storage_tables=weight_args.scale_storage_tables,
                main_grad_storage_tables=main_grad_storage_tables,
                **shared_grad_kwargs,
            )
            if direct:
                bwd_outputs = _launch(
                    dispatch_mlp_swiglu_combine_bwd_mxfp8_accum,
                    _bwd_mxfp8_accum_entrypoint,
                    mxfp8_bwd_args,
                    (*main_grads, *(weight_args.storage_tables or ()), *(weight_args.scale_storage_tables or ()),
                     *(main_grad_storage_tables or ())),
                    ("bwd_mxfp8_accum", len(main_grads), weight_args.storage_tables is None,
                     main_grad_storage_tables is None),
                    **accum_kwargs,
                )
            else:
                bwd_outputs = dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(*mxfp8_bwd_args, **accum_kwargs)
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_fp8_routed,
                d_gate_sc_routed,
                d_up_shared,
                d_up_fp8_routed,
                d_up_sc_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_fp8_routed,
                d_y_sc_routed,
            ) = bwd_outputs
            (
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_down,
                d_w_routed_down,
            ) = main_grads
    else:
        # BF16 has no scale or rowwise/columnwise representation pair.
        x_routed = forward_context.x_routed
        gate_routed = forward_context.gate_routed
        up_routed = forward_context.up_routed
        hidden_routed = forward_context.hidden_routed
        bwd_args = (
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights,
            weight_args.gate_data,
            shared_up_weights,
            weight_args.up_data,
            shared_down_weights,
            weight_args.down_data,
            x_routed,
            forward_context.gate_shared,
            gate_routed,
            forward_context.up_shared,
            up_routed,
            forward_context.hidden_shared,
            hidden_routed,
            x_shared,
            workspace.x_buffer_ptrs,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        if main_grads is None:
            # Original MOK API: materialize and return six fresh weight gradients.
            if direct:
                bwd_outputs = _launch(
                    dispatch_mlp_swiglu_combine_bwd_bf16,
                    _C.dispatch_mlp_swiglu_combine_bwd_bf16,
                    bwd_args,
                    (workspace.d_y_buffer, *bwd_weights),
                    ("bwd_bf16", config, swiglu_limit, schedule.tokens_per_expert.shape),
                    **shared_grad_kwargs,
                )
            else:
                bwd_outputs = dispatch_mlp_swiglu_combine_bwd_bf16(*bwd_args, **shared_grad_kwargs)
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_routed,
                d_up_shared,
                d_up_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_routed,
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_down,
                d_w_routed_down,
            ) = bwd_outputs
        else:
            # Fused accumulation: mutate the six supplied main-grad buffers.
            # Descriptor tables are present only for non-single expert storage.
            accum_kwargs = dict(
                main_grads=main_grads,
                weight_storage_tables=weight_args.storage_tables,
                main_grad_storage_tables=main_grad_storage_tables,
                **shared_grad_kwargs,
            )
            if direct:
                bwd_outputs = _launch(
                    dispatch_mlp_swiglu_combine_bwd_bf16_accum,
                    _bwd_bf16_accum_entrypoint,
                    bwd_args,
                    (*main_grads, *(weight_args.storage_tables or ()), *(main_grad_storage_tables or ())),
                    ("bwd_bf16_accum", len(main_grads), weight_args.storage_tables is None,
                     main_grad_storage_tables is None),
                    **accum_kwargs,
                )
            else:
                bwd_outputs = dispatch_mlp_swiglu_combine_bwd_bf16_accum(*bwd_args, **accum_kwargs)
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_routed,
                d_up_shared,
                d_up_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_routed,
            ) = bwd_outputs
            (
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_down,
                d_w_routed_down,
            ) = main_grads

    _barrier(workspace, direct)
    if direct:
        # d_x_shared comes from the kernel above; the workspace and routes are validated.
        d_x = _C.bwd_epilogue(d_x_shared, workspace.d_x_routed_buffer, schedule.top_experts)
    else:
        d_x = bwd_epilogue(d_x_shared, workspace.d_x_routed_buffer,
                           schedule.top_experts)
    d_w_output_gate = None
    if gate is not None:
        with torch.no_grad():
            # x_shared is x on the direct path if x is 256-byte aligned and its staged copy otherwise, so the cuBLAS
            # result does not depend on x's alignment.
            d_w_output_gate = _shared_output_gate_wgrad(d_z, x_shared, shared_output_gate_main_grad)
            # Deliberate BF16 merge of dZ @ Wg after the original MLP epilogue, in place on its fresh d_x.
            if fused_gate and (direct or _fused_gate_kernels(d_x, shared_output_gate_weight)):
                d_x = _gate_fused._launch_gate_bwd_post_(d_x, d_z, shared_output_gate_weight)
            else:
                d_x = d_x + d_z @ shared_output_gate_weight
    d_router_weights = workspace.d_router_weight_buffer.clone()  # TODO: we can remove this
    if zero_unrouted_router_grads:
        d_router_weights.masked_fill_(schedule.top_experts < 0, 0.0)
    return (
        d_x,
        d_router_weights,
        d_w_routed_gate,
        d_w_routed_up,
        d_w_routed_down,
        d_w_shared_gate,
        d_w_shared_up,
        d_w_shared_down,
        d_w_output_gate,
    )
