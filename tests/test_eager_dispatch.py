"""Eager dispatch: functional launches MoK's C++ entry points directly and validates caller tensors once."""

import contextlib
import dataclasses
import gc
import re

import pytest
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from mok import functional, ops

from .utils import generate_inputs

CUSTOM_OPS = (
    "all_gather_top_experts",
    "barrier_all",
    "schedule",
    "stage_into_buffers",
    "dispatch_mlp_swiglu_combine_fwd_mxfp8",
    "dispatch_mlp_swiglu_combine_fwd_bf16",
    "dispatch_mlp_swiglu_combine_bwd_mxfp8",
    "dispatch_mlp_swiglu_combine_bwd_bf16",
    "fwd_epilogue",
    "bwd_epilogue",
)
ACCUMULATING_WRAPPERS = (
    "dispatch_mlp_swiglu_combine_bwd_mxfp8_accum",
    "dispatch_mlp_swiglu_combine_bwd_bf16_accum",
)


def _problem(
    context, precision, *, gated=True, accumulate=True, single_grouped=True, num_local_tokens=512, hidden_dim=256,
):
    """One rank's MoE problem with unrouted slots; single-grouped is MCore's argument layout."""
    rank, world_size, device = context
    num_local_experts = 2
    intermediate_dim = 256
    topk = 2
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=1024,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    (
        x, top_experts, router_weights,
        w_shared_gate, w_shared_up, w_shared_down,
        w_routed_gate, w_routed_up, w_routed_down,
        d_output,
    ) = generate_inputs(
        rank, device, world_size * num_local_experts, num_local_experts, topk,
        num_local_tokens, hidden_dim, intermediate_dim,
    )
    top_experts = top_experts.to(torch.int32)
    top_experts[::7, 1] = -1
    top_experts[::11] = -1
    router_weights = router_weights.masked_fill(top_experts < 0, 0.0)
    if single_grouped:
        routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
        if precision == "mxfp8":
            fc1 = ops.mxfp8_quantize(routed_fc1, True, True)
            down = ops.mxfp8_quantize(w_routed_down, True, True)
            forward_routed = (fc1[:2], fc1[:2], down[:2])
            # TE-native columnwise payloads, as MCore single-grouped MXFP8 weights provide them.
            fc1 = (fc1[0], fc1[1], fc1[2].transpose(-2, -1).contiguous(), fc1[3], True)
            backward_routed = (fc1, fc1, (down[2].transpose(-2, -1).contiguous(), down[3], True))
        else:
            forward_routed = backward_routed = (routed_fc1, routed_fc1, w_routed_down)
        routed_main_grad_shapes = (routed_fc1, routed_fc1, w_routed_down)
    else:
        assert precision == "bf16"
        forward_routed = backward_routed = (w_routed_gate, w_routed_up, w_routed_down)
        routed_main_grad_shapes = forward_routed
    main_grads = gate_main_grad = None
    if accumulate:
        routed_gate_main_grad = torch.zeros_like(routed_main_grad_shapes[0], dtype=torch.float32)
        routed_up_main_grad = (
            routed_gate_main_grad if single_grouped
            else torch.zeros_like(routed_main_grad_shapes[1], dtype=torch.float32)
        )
        main_grads = (
            torch.zeros_like(w_shared_gate, dtype=torch.float32),
            routed_gate_main_grad,
            torch.zeros_like(w_shared_up, dtype=torch.float32),
            routed_up_main_grad,
            torch.zeros_like(w_shared_down, dtype=torch.float32),
            torch.zeros_like(w_routed_down, dtype=torch.float32),
        )
    gate_kwargs = {}
    if gated:
        generator = torch.Generator(device=device).manual_seed(77 + rank)
        gate_weight = torch.randn(1, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) / 16
        gate_kwargs = {"shared_output_gate_weight": gate_weight}
        if accumulate:
            gate_main_grad = torch.zeros_like(gate_weight, dtype=torch.float32)
    return dict(
        config=config, workspace=workspace, num_local_experts=num_local_experts, x=x,
        top_experts=top_experts, router_weights=router_weights, d_output=d_output,
        shared=(w_shared_gate, w_shared_up, w_shared_down),
        forward_routed=forward_routed, backward_routed=backward_routed,
        main_grads=main_grads, gate_main_grad=gate_main_grad, gate_kwargs=gate_kwargs,
    )


def _step(problem, **backward_kwargs):
    """build_schedule + forward + backward, as MCore's autograd bridge calls them."""
    p = problem
    schedule = functional.build_schedule(
        p["workspace"], p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    output, forward_context = functional.forward(
        p["config"], p["workspace"], schedule, p["x"], p["router_weights"], *p["shared"],
        *p["forward_routed"], **p["gate_kwargs"],
    )
    gate_kwargs = dict(p["gate_kwargs"])
    if p["gate_main_grad"] is not None:
        gate_kwargs["shared_output_gate_main_grad"] = p["gate_main_grad"]
    gradients = functional.backward(
        p["config"], p["workspace"], schedule, forward_context, p["d_output"], p["x"],
        p["router_weights"], *p["shared"], *p["backward_routed"], main_grads=p["main_grads"],
        **gate_kwargs, **backward_kwargs,
    )
    return output, gradients


def _count_calls(monkeypatch, names):
    counts = dict.fromkeys(names, 0)
    for name in names:
        op = getattr(functional, name)

        def counted(*args, _op=op, _name=name, **kwargs):
            counts[_name] += 1
            return _op(*args, **kwargs)

        monkeypatch.setattr(functional, name, counted)
    return counts


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_eager_dispatch_matches_custom_ops(context, monkeypatch, precision, gated):
    problem = _problem(context, precision, gated=gated)
    main_grads = [g for g in (*problem["main_grads"], problem["gate_main_grad"]) if g is not None]

    def run(direct):
        for main_grad in main_grads:
            main_grad.zero_()
        functional._VALIDATED_LAUNCHES.clear()
        with monkeypatch.context() as patch:
            if not direct:
                patch.setattr(functional, "_eager_dispatch", lambda tensors: False)
            counts = _count_calls(patch, CUSTOM_OPS + ACCUMULATING_WRAPPERS)
            results = []
            # Two microbatches accumulate into the same main grads, as in MCore.
            for _ in range(2):
                output, (d_x, d_router_weights, *_) = _step(problem)
                results += [output, d_x, d_router_weights]
        torch.cuda.synchronize()
        return results + [main_grad.clone() for main_grad in main_grads], counts

    expected, op_counts = run(direct=False)
    actual, direct_counts = run(direct=True)

    for index, (reference, result) in enumerate(zip(expected, actual, strict=True)):
        assert torch.equal(reference, result), f"result {index} differs between dispatch paths"
    # Every launch went through a custom op (or wrapper) before; now only the first megakernel
    # launch of each kind with these weights does, for its argument checks.
    accumulate = f"dispatch_mlp_swiglu_combine_bwd_{precision}_accum"
    assert op_counts["barrier_all"] == 2 * 5 and op_counts[accumulate] == 2
    # One staging call in forward (x, router weights) and one in backward (grad_output, x, router weights).
    assert op_counts["stage_into_buffers"] == 2 * 2
    assert {name: count for name, count in direct_counts.items() if count} == {
        f"dispatch_mlp_swiglu_combine_fwd_{precision}": 1,
        accumulate: 1,
    }


def test_eager_dispatch_validates_new_weights_on_first_use(context):
    problem = _problem(context, "bf16", gated=False)
    _step(problem)
    w_shared_gate, w_shared_up, w_shared_down = problem["shared"]
    problem["shared"] = (w_shared_gate, w_shared_up[:128], w_shared_down)
    with pytest.raises(ValueError, match="w_shared_up"):
        _step(problem)
    problem["shared"] = (w_shared_gate, w_shared_up.clone(), w_shared_down)
    _step(problem)
    torch.cuda.synchronize()


def test_validated_launches_drop_entries_with_their_tensors():
    launches = functional._ValidatedLaunches()
    weight, table = torch.empty(4), torch.empty(2)
    key = ("op", id(weight), id(None), id(table))
    launches.add(key, (weight, None, table))
    assert key in launches
    del weight
    gc.collect()
    assert key not in launches

    calls = []

    def validated(*args, **kwargs):
        calls.append("validated")
        if kwargs.get("bad"):
            raise ValueError("bad")
        return args

    def entrypoint(*args, **kwargs):
        calls.append("entrypoint")
        return args

    for _ in range(3):
        assert functional._launch(validated, entrypoint, (1, 2), (table,), ("op",)) == (1, 2)
    assert calls == ["validated", "entrypoint", "entrypoint"]
    calls.clear()
    other = torch.empty(2)
    for _ in range(2):
        with pytest.raises(ValueError):
            functional._launch(validated, entrypoint, (), (other,), ("op",), bad=True)
    # A failed validation is not remembered.
    assert calls == ["validated", "validated"]


def test_get_workspace_lookup_skips_revalidation(context, monkeypatch):
    _, _, device = context
    calls = []
    validate = functional.validate_workspace_args

    def counting_validate(*args, **kwargs):
        calls.append(kwargs["num_local_tokens"])
        return validate(*args, **kwargs)

    monkeypatch.setattr(functional, "validate_workspace_args", counting_validate)
    config = functional.MoKConfig(minibatch_size=256, macrobatch_size=512)
    kwargs = dict(device=device, num_local_tokens=1024, hidden_size=256, topk=2)
    first = functional.get_workspace(config, dist.group.WORLD, **kwargs)
    validated = len(calls)  # get_workspace, plus create_workspace if it built the workspace
    assert validated in (1, 2)
    assert functional.get_workspace(config, dist.group.WORLD, **kwargs) is first
    equal_config = functional.MoKConfig(minibatch_size=256, macrobatch_size=512)
    assert functional.get_workspace(equal_config, dist.group.WORLD, **kwargs) is first
    assert len(calls) == validated
    # Invalid arguments are never remembered: each call validates and raises again.
    for _ in range(2):
        with pytest.raises(ValueError, match="num_local_tokens"):
            functional.get_workspace(config, dist.group.WORLD, **{**kwargs, "num_local_tokens": 100})
    assert calls[validated:] == [100, 100]


def _isolate_workspace_caches(monkeypatch):
    """Empty workspace caches for one test, and a create_workspace that returns stand-in objects.

    The stand-ins allocate no symmetric memory (which must not outlive its process group) and never reach the
    barriers of clear_workspace_cache: monkeypatch restores the real caches afterwards.
    """
    monkeypatch.setattr(functional, "_WORKSPACE_CACHE", {})
    monkeypatch.setattr(functional, "_WORKSPACE_LOOKUP", {})
    created = []

    def create_stand_in(*args, **kwargs):
        created.append(object())
        return created[-1]

    monkeypatch.setattr(functional, "create_workspace", create_stand_in)
    return created


def test_get_workspace_lookup_rejects_destroyed_group(context, monkeypatch):
    """A lookup hit is returned only while its process group exists: after the group is destroyed, every call
    validates the arguments again and raises, as get_workspace did before it had the lookup."""
    _, world_size, device = context
    created = _isolate_workspace_caches(monkeypatch)
    calls = []
    validate = functional.validate_workspace_args

    def counting_validate(*args, **kwargs):
        calls.append(kwargs["num_local_tokens"])
        return validate(*args, **kwargs)

    monkeypatch.setattr(functional, "validate_workspace_args", counting_validate)
    config = functional.MoKConfig(minibatch_size=256, macrobatch_size=512)
    kwargs = dict(device=device, num_local_tokens=1024, hidden_size=256, topk=2)
    group = dist.new_group(list(range(world_size)))
    workspace = functional.get_workspace(config, group, **kwargs)
    assert functional.get_workspace(config, group, **kwargs) is workspace
    assert len(calls) == 1  # the second call was a lookup hit
    dist.destroy_process_group(group)
    for _ in range(2):
        with pytest.raises((RuntimeError, ValueError)):
            functional.get_workspace(config, group, **kwargs)
    assert len(calls) == 3
    # The stale entry is dropped, so the lookup no longer keeps the destroyed group alive.
    assert not any(group in key for key in functional._WORKSPACE_LOOKUP)
    assert created == [workspace]


def test_get_workspace_lookup_rejects_uninitialized_distributed(context, monkeypatch):
    """Once torch.distributed reports itself uninitialized (as after destroy_process_group()), a lookup hit is not
    returned: get_workspace validates and raises. Re-initialized, the arguments validate and hit the cache again."""
    _, _, device = context
    created = _isolate_workspace_caches(monkeypatch)
    config = functional.MoKConfig(minibatch_size=256, macrobatch_size=512)
    kwargs = dict(device=device, num_local_tokens=1024, hidden_size=256, topk=2)
    world = dist.group.WORLD
    workspace = functional.get_workspace(config, world, **kwargs)
    assert functional.get_workspace(config, world, **kwargs) is workspace
    with monkeypatch.context() as patch:
        patch.setattr(dist, "is_initialized", lambda: False)
        with pytest.raises(RuntimeError, match="must be initialized"):
            functional.get_workspace(config, world, **kwargs)
    assert functional.get_workspace(config, world, **kwargs) is workspace
    assert created == [workspace]


@pytest.mark.parametrize("bad_weight", ["fp32", "short", "column", "strided"])
def test_gate_weight_checked_by_forward_and_backward(context, bad_weight):
    """forward and backward both reject a gate weight that is not contiguous BF16 [1, H], before launching anything.

    backward used to hand any weight to the fused gate kernels, which read it as BF16 [1, H] without checks.
    """
    p = _problem(context, "mxfp8", gated=True)
    weight = p["gate_kwargs"]["shared_output_gate_weight"]
    hidden = weight.shape[1]
    bad = {
        "fp32": lambda: weight.float(),
        "short": lambda: torch.cat((weight, weight), dim=1)[:, : hidden // 2],  # contiguous [1, H/2]
        "column": lambda: weight.reshape(hidden, 1),
        "strided": lambda: torch.stack((weight, weight), dim=-1).reshape(1, 2 * hidden)[:, ::2],
    }[bad_weight]()
    schedule = functional.build_schedule(
        p["workspace"], p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    forward_args = (p["config"], p["workspace"], schedule, p["x"], p["router_weights"], *p["shared"],
                    *p["forward_routed"])
    with pytest.raises(ValueError, match="shared_output_gate_weight must be"):
        functional.forward(*forward_args, shared_output_gate_weight=bad)
    _, forward_context = functional.forward(*forward_args, shared_output_gate_weight=weight)
    backward_args = (p["config"], p["workspace"], schedule, forward_context, p["d_output"], p["x"],
                     p["router_weights"], *p["shared"], *p["backward_routed"])
    backward_kwargs = dict(main_grads=p["main_grads"], shared_output_gate_main_grad=p["gate_main_grad"])
    with pytest.raises(ValueError, match="shared_output_gate_weight must be"):
        functional.backward(*backward_args, shared_output_gate_weight=bad, **backward_kwargs)
    torch.cuda.synchronize()
    # Nothing was launched: every main grad is still zero, and the same context still runs its backward.
    assert not any(main_grad.any() for main_grad in (*p["main_grads"], p["gate_main_grad"]))
    functional.backward(*backward_args, shared_output_gate_weight=weight, **backward_kwargs)
    torch.cuda.synchronize()
    assert p["main_grads"][0].any() and p["gate_main_grad"].any()


def test_zero_unrouted_router_grads_opt_out(context):
    problem = _problem(context, "mxfp8", gated=True)
    unrouted = problem["top_experts"] < 0
    assert unrouted.any() and (~unrouted).any()
    output, (d_x, d_router_weights, *_) = _step(problem)
    output_opt_out, (d_x_opt_out, d_router_weights_opt_out, *_) = _step(
        problem, zero_unrouted_router_grads=False
    )
    torch.cuda.synchronize()

    assert torch.equal(output, output_opt_out) and torch.equal(d_x, d_x_opt_out)
    assert not d_router_weights[unrouted].any()
    assert torch.equal(d_router_weights[~unrouted], d_router_weights_opt_out[~unrouted])
    # Unrouted slots are unspecified but never uninitialized memory.
    assert torch.isfinite(d_router_weights_opt_out).all()


class _RecordingMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _mok_targets(names):
    """MoK custom ops in call order."""
    return [name for name in map(str, names) if name.startswith("mok.")]


def _expected_targets(precision, *, backward=True):
    """The custom-op path of build_schedule, forward and a backward without main_grads. The inputs are staged by full
    copies into the workspace buffers, one staging call per direction (x and router weights; then grad_output, x and
    router weights), because no data pointer is read off the direct path."""
    forward = [
        "mok.all_gather_top_experts.default",
        "mok.barrier_all.default",
        "mok.schedule.default",
        "mok.stage_into_buffers.default",
        "mok.barrier_all.default",
        f"mok.dispatch_mlp_swiglu_combine_fwd_{precision}.default",
        "mok.barrier_all.default",
        "mok.fwd_epilogue.default",
    ]
    if not backward:
        return forward
    return forward + [
        "mok.stage_into_buffers.default",
        "mok.barrier_all.default",
        f"mok.dispatch_mlp_swiglu_combine_bwd_{precision}.default",
        "mok.barrier_all.default",
        "mok.bwd_epilogue.default",
    ]


def test_dispatch_modes_see_the_custom_ops(context):
    problem = _problem(context, "bf16", gated=False, accumulate=False, single_grouped=False)
    with _RecordingMode() as mode:
        _step(problem)
    torch.cuda.synchronize()
    assert _mok_targets(mode.ops) == _expected_targets("bf16")
    assert "aten.copy_.default" not in mode.ops


class _Subclass(torch.Tensor):
    """A tensor subclass that otherwise behaves like a plain tensor: it shares its storage and wraps no results."""

    __torch_function__ = torch._C._disabled_torch_function_impl


class _WrapperTensor(torch.Tensor):
    """A __torch_dispatch__ wrapper subclass, like DTensor or a quantized-tensor subclass.

    It owns no storage, so data_ptr() is 0: a kernel launched on it directly reads address 0, an illegal memory
    access. Every operation runs on the plain tensor it wraps. With rewrap=True the results are wrapped again, so the
    subclass spreads to the forward context.
    """

    @staticmethod
    def __new__(cls, inner, rewrap=False):
        return torch.Tensor._make_wrapper_subclass(
            cls, inner.shape, strides=inner.stride(), storage_offset=inner.storage_offset(), dtype=inner.dtype,
            device=inner.device,
        )

    def __init__(self, inner, rewrap=False):
        self.inner = inner
        self.rewrap = rewrap

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        rewrap = any(
            tensor.rewrap for tensor in pytree.tree_leaves((args, kwargs)) if isinstance(tensor, _WrapperTensor)
        )
        args, kwargs = pytree.tree_map_only(_WrapperTensor, lambda tensor: tensor.inner, (args, kwargs or {}))
        outputs = func(*args, **kwargs)
        if rewrap:
            return pytree.tree_map_only(torch.Tensor, lambda tensor: _WrapperTensor(tensor, rewrap=True), outputs)
        return outputs


def _unwrap(value):
    return value.inner if isinstance(value, _WrapperTensor) else value


def test_eager_dispatch_takes_plain_operands_outside_tracing():
    """The direct path needs every operand to be a plain tensor or parameter (None stands for an absent optional one)
    and no tracer or transform: under torch.compile, a dispatch mode, pre-dispatch tracing or a torch.func transform,
    whose wrapper tensors have the exact type torch.Tensor, a call always gets the custom ops. The fused gate kernels
    follow the same rule, except that torch.compile traces them."""
    from torch.fx.experimental.proxy_tensor import make_fx

    class _ParameterSubclass(torch.nn.Parameter):
        pass

    tensor = torch.zeros(2)
    parameter = torch.nn.Parameter(torch.zeros(2))
    assert functional._eager_dispatch((tensor, None, parameter))
    assert functional._fused_gate_kernels(tensor, parameter)
    for subclass in (tensor.as_subclass(_Subclass), _ParameterSubclass(torch.zeros(2)), _WrapperTensor(tensor)):
        assert not functional._eager_dispatch((tensor, subclass, parameter))
        assert not functional._fused_gate_kernels(tensor, subclass)
    decisions = []

    def decide(operand):
        decisions.append((functional._eager_dispatch((operand,)), functional._fused_gate_kernels(operand)))
        return operand + 1

    with _RecordingMode():
        decide(tensor)
    make_fx(decide, tracing_mode="real")(tensor)
    make_fx(decide, tracing_mode="fake")(tensor)
    # Pre-dispatch tracing keeps its modes off the dispatch-mode stack: a plain real input is all the call sees.
    make_fx(decide, tracing_mode="real", pre_dispatch=True)(tensor)
    torch.func.vmap(decide)(torch.zeros(3, 2))
    torch.func.functionalize(decide)(tensor)
    torch.func.grad(lambda operand: decide(operand).sum())(tensor)
    assert decisions == [(False, False)] * 7
    decisions.clear()
    torch._dynamo.reset()
    try:
        torch.compile(decide, backend="eager", fullgraph=True)(tensor)
    finally:
        torch._dynamo.reset()
    assert decisions == [(False, True)]


class _LaunchRecorder:
    """Stands in for a module whose kernels functional launches directly (_C, _gate_fused) and records every launch
    that receives a tensor subclass."""

    def __init__(self, module, launches):
        self._module = module
        self._launches = launches

    def __getattr__(self, name):
        entrypoint = getattr(self._module, name)
        if not callable(entrypoint):
            return entrypoint

        def launch(*args, **kwargs):
            for value in (*args, *kwargs.values()):
                for tensor in value if isinstance(value, (list, tuple)) else (value,):
                    if isinstance(tensor, torch.Tensor) and type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                        self._launches.append(name)
            return entrypoint(*args, **kwargs)

        return launch


def _record_subclass_launches(monkeypatch):
    launches = []
    monkeypatch.setattr(functional, "_C", _LaunchRecorder(functional._C, launches))
    monkeypatch.setattr(functional, "_gate_fused", _LaunchRecorder(functional._gate_fused, launches))
    return launches


def _use_subclass(problem, operand, monkeypatch, wrap):
    """Replaces one tensor operand of ``problem`` (MXFP8, single grouped, gated), or of the schedule or forward
    context that _step passes on, by ``wrap(tensor)``."""
    p = problem
    if operand in ("x", "top_experts", "router_weights", "d_output", "gate_main_grad"):
        p[operand] = wrap(p[operand])
    elif operand == "shared_weight":
        w_gate, w_up, w_down = p["shared"]
        p["shared"] = (w_gate, w_up, wrap(w_down))
    elif operand == "routed_weight":
        fc1, _, (data, scale) = p["forward_routed"]
        p["forward_routed"] = (fc1, fc1, (data, wrap(scale)))
        fc1, _, (data, scale, native) = p["backward_routed"]
        p["backward_routed"] = (fc1, fc1, (wrap(data), scale, native))
    elif operand == "main_grad":
        p["main_grads"] = (*p["main_grads"][:5], wrap(p["main_grads"][5]))
    elif operand == "gate_weight":
        p["gate_kwargs"] = {"shared_output_gate_weight": wrap(p["gate_kwargs"]["shared_output_gate_weight"])}
    elif operand == "schedule":
        build_schedule = functional.build_schedule

        def build_schedule_with_subclass(*args, **kwargs):
            schedule = build_schedule(*args, **kwargs)
            return dataclasses.replace(schedule, peer_rank=wrap(schedule.peer_rank))

        monkeypatch.setattr(functional, "build_schedule", build_schedule_with_subclass)
    elif operand == "forward_context":
        forward = functional.forward

        def forward_with_subclass_context(*args, **kwargs):
            output, forward_context = forward(*args, **kwargs)
            return output, dataclasses.replace(forward_context, hidden_shared=wrap(forward_context.hidden_shared))

        monkeypatch.setattr(functional, "forward", forward_with_subclass_context)
    else:
        raise AssertionError(operand)


@pytest.mark.parametrize(
    "operand",
    ["x", "top_experts", "router_weights", "schedule", "shared_weight", "routed_weight", "forward_context",
     "gate_weight", "gate_main_grad"],
)
def test_subclass_operand_takes_custom_ops(context, monkeypatch, operand):
    """One tensor subclass among a call's operands sends every C++ launch of that call through the custom ops, on
    every call: a launch with a subclass operand is never remembered for direct launches, and no kernel that
    functional launches directly (C++ entry point or fused gate kernel) ever receives the subclass. The results equal
    the direct path's bitwise. The gate operands here are plain except the gate weight, whose merge then runs as the
    reference torch operations, which round like the fused merge."""
    problem = _problem(context, "mxfp8", gated=True, accumulate=False)
    problem["gate_main_grad"] = torch.zeros_like(problem["gate_kwargs"]["shared_output_gate_weight"],
                                                 dtype=torch.float32)

    def run():
        problem["gate_main_grad"].zero_()
        functional._VALIDATED_LAUNCHES.clear()
        with monkeypatch.context() as patch:
            counts = _count_calls(patch, CUSTOM_OPS + ACCUMULATING_WRAPPERS + ("_launch",))
            launches = _record_subclass_launches(patch)
            results = []
            for _ in range(2):
                output, gradients = _step(problem)
                results += [output, *gradients]
        torch.cuda.synchronize()
        return results + [problem["gate_main_grad"].clone()], counts, launches

    expected, _, _ = run()
    _use_subclass(problem, operand, monkeypatch, lambda tensor: tensor.as_subclass(_Subclass))
    actual, counts, subclass_launches = run()

    for index, (reference, result) in enumerate(zip(expected, actual, strict=True)):
        assert torch.equal(reference, result), f"result {index} differs from the direct path"
    assert subclass_launches == []
    # Two microbatches. build_schedule launches directly unless the routes themselves are the subclass; forward
    # does unless one of its operands is.
    backward_only = operand in ("forward_context", "gate_main_grad")
    expected_counts = {
        "barrier_all": 2 * (2 + (0 if backward_only else 2) + (1 if operand == "top_experts" else 0)),
        "stage_into_buffers": 2 * (1 + (0 if backward_only else 1)),
        "dispatch_mlp_swiglu_combine_bwd_mxfp8": 2,
        "bwd_epilogue": 2,
    }
    if backward_only:
        # Direct forward: its megakernel launch goes through the custom op once, for the first-use checks.
        expected_counts |= {"_launch": 2, "dispatch_mlp_swiglu_combine_fwd_mxfp8": 1}
    else:
        expected_counts |= {"dispatch_mlp_swiglu_combine_fwd_mxfp8": 2, "fwd_epilogue": 2}
    if operand == "top_experts":
        expected_counts |= {"all_gather_top_experts": 2, "schedule": 2}
    assert {name: count for name, count in counts.items() if count} == expected_counts


def _reference_gate(monkeypatch):
    """All launches through the custom ops and the gate backward as the reference torch operations: what a call with
    a subclass gate operand, or under a dispatch mode, computes for plain tensors."""
    monkeypatch.setattr(functional, "_eager_dispatch", lambda tensors: False)
    monkeypatch.setattr(functional, "_fused_gate_kernels", lambda *tensors: False)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize(
    "operand",
    ["routed_weight", "shared_weight", "router_weights", "gate_weight", "main_grad", "d_output", "x_rewrapped"],
)
def test_wrapper_subclass_operand(context, monkeypatch, operand, accumulate):
    """A __torch_dispatch__ wrapper subclass (no storage, data_ptr() 0) never reaches a kernel that functional launches
    directly, on any call. Without main_grads, forward and backward run through the custom ops and the reference gate
    operations, and give the results of plain tensors bitwise. With main_grads, which has no custom op, backward raises
    a TypeError before it launches anything."""
    if operand == "main_grad" and not accumulate:
        pytest.skip("main grads exist only with accumulation")
    problem = _problem(context, "mxfp8", gated=True, accumulate=accumulate)
    workspace = problem["workspace"]
    # The reference gate operations sum the gate gradient in another order than the fused kernel, so a call whose
    # grad_output or saved S/G is a subclass is compared with plain tensors on that path.
    reference_gate = operand in ("d_output", "x_rewrapped")

    def run(record=False):
        for main_grad in (*(problem["main_grads"] or ()), problem["gate_main_grad"]):
            if main_grad is not None:
                _unwrap(main_grad).zero_()
        results = []
        with monkeypatch.context() as patch:
            launches = _record_subclass_launches(patch) if record else []
            if reference_gate and not record:
                _reference_gate(patch)
            for _ in range(2):  # the second call is the one a warm validation cache would launch directly
                output, gradients = _step(problem)
                results += [_unwrap(output), *map(_unwrap, gradients)]
        torch.cuda.synchronize()
        assert launches == []
        return results

    if not accumulate:
        expected = run()
        if operand == "x_rewrapped":
            _use_subclass(problem, "x", monkeypatch, lambda tensor: _WrapperTensor(tensor, rewrap=True))
        else:
            _use_subclass(problem, operand, monkeypatch, _WrapperTensor)
        actual = run(record=True)
        for index, (reference, result) in enumerate(zip(expected, actual, strict=True)):
            assert type(result) is not _WrapperTensor
            assert torch.equal(reference, result), f"result {index} differs from plain tensors"
        return

    if operand == "x_rewrapped":
        _use_subclass(problem, "x", monkeypatch, lambda tensor: _WrapperTensor(tensor, rewrap=True))
    else:
        _use_subclass(problem, operand, monkeypatch, _WrapperTensor)
    p = problem
    schedule = functional.build_schedule(
        p["workspace"], p["config"], _unwrap(p["top_experts"]), num_local_experts=p["num_local_experts"]
    )
    _, forward_context = functional.forward(
        p["config"], workspace, schedule, p["x"], p["router_weights"], *p["shared"], *p["forward_routed"],
        **p["gate_kwargs"],
    )
    workspace.d_y_buffer.fill_(float("nan"))
    torch.cuda.synchronize()
    barrier_target = workspace.barrier_target.clone()
    with pytest.raises(TypeError, match="main_grads"):
        functional.backward(
            p["config"], workspace, schedule, forward_context, p["d_output"], p["x"], p["router_weights"],
            *p["shared"], *p["backward_routed"], main_grads=p["main_grads"],
            shared_output_gate_main_grad=p["gate_main_grad"], **p["gate_kwargs"],
        )
    torch.cuda.synchronize()
    # Nothing was launched: no barrier, no staging, no accumulation.
    assert torch.equal(workspace.barrier_target, barrier_target)
    assert workspace.d_y_buffer.isnan().all()
    assert not any(_unwrap(main_grad).any() for main_grad in (*p["main_grads"], p["gate_main_grad"]))


def _poison(workspace):
    """Fills the buffers that peers read through raw pointers with NaN, so a read that precedes its staging shows."""
    for buffer in (workspace.x_buffer, workspace.d_y_buffer, workspace.router_weight_buffer):
        buffer.fill_(float("nan"))


def _offset_storage(tensor, offset_bytes):
    """A 512-byte-aligned 1-D tensor that holds a copy of ``tensor`` ``offset_bytes`` past its start, and that offset
    in elements."""
    offset = offset_bytes // tensor.element_size()
    storage = torch.empty(tensor.numel() + 512 // tensor.element_size(), dtype=tensor.dtype, device=tensor.device)
    assert storage.data_ptr() % 512 == 0
    storage[offset:offset + tensor.numel()].copy_(tensor.flatten())
    return storage, offset


def _at_offset(tensor, offset_bytes):
    """A contiguous copy of ``tensor`` that starts ``offset_bytes`` past a 512-byte-aligned allocation."""
    storage, offset = _offset_storage(tensor, offset_bytes)
    view = storage[offset:offset + tensor.numel()].view(tensor.shape)
    assert view.data_ptr() % 512 == offset * tensor.element_size()
    return view


def _misaligned_inputs(p, offset):
    """x, router weights and grad_output at ``offset`` bytes past 512-byte alignment (router weights at whole FP32
    elements)."""
    return (
        _at_offset(p["x"], offset),
        _at_offset(p["router_weights"], offset - offset % 4),
        _at_offset(p["d_output"], offset),
    )


@pytest.mark.parametrize(
    "precision,num_local_tokens,hidden_dim", [("bf16", 512, 256), ("mxfp8", 512, 256), ("mxfp8", 8192, 1024)]
)
def test_eager_stages_misaligned_inputs_in_full(context, monkeypatch, precision, num_local_tokens, hidden_dim):
    """In eager mode, an x or grad_output that is not 256-byte aligned does not stand in for its workspace buffer: the
    direct path stages it by a full copy, and every result equals the aligned run's bitwise. That includes the gate
    and its weight gradient, which cuBLAS computes from the staged copy, as the shared expert does: at 8192 x 1024, a
    gate computed from x at a 2-byte offset differs in the last bit for some tokens, which changes the output and
    every gradient."""
    p = _problem(
        context, precision, gated=True, accumulate=False, single_grouped=precision == "mxfp8",
        num_local_tokens=num_local_tokens, hidden_dim=hidden_dim,
    )
    schedule = functional.build_schedule(
        p["workspace"], p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )

    def step(x, router_weights, d_output):
        output, forward_context = functional.forward(
            p["config"], p["workspace"], schedule, x, router_weights, *p["shared"], *p["forward_routed"],
            **p["gate_kwargs"],
        )
        gradients = functional.backward(
            p["config"], p["workspace"], schedule, forward_context, d_output, x, router_weights, *p["shared"],
            *p["backward_routed"], **p["gate_kwargs"],
        )
        return (forward_context.shared_output_gate, output, *gradients)

    names = ("gate", "output", "d_x", "d_router_weights", *(f"weight gradient {i}" for i in range(6)),
             "gate-weight gradient")
    expected = step(p["x"], p["router_weights"], p["d_output"])
    failures = []
    with monkeypatch.context() as patch:
        counts = _count_calls(patch, ("barrier_all",))
        for offset in (2, 16, 128):
            actual = step(*_misaligned_inputs(p, offset))
            torch.cuda.synchronize()
            differing = [
                name for name, reference, result in zip(names, expected, actual, strict=True)
                if not torch.equal(reference, result)
            ]
            if differing:
                failures.append(f"{offset}-byte offset: {differing} differ from the aligned run")
    # Every misaligned call took the direct path: no barrier custom op (each path makes the same barriers).
    assert not any(counts.values()), counts
    assert not failures, failures


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_inductor_stages_inputs_ahead_of_peers(context, precision, gated):
    """Under torch.compile with the default backend (AOTAutograd and Inductor), forward and backward stage x, router
    weights and grad_output into the workspace buffers before the barrier after which peers read them, at any
    alignment: of the compiled function's inputs, and of contiguous views that the compiled function takes at an
    offset. The buffers are poisoned before every compiled call, and the results equal eager ones bitwise."""
    from torch._inductor.utils import run_and_get_code

    # BF16 with separate gate and up weights: the fake kernel of the BF16 backward does not describe the
    # combined-FC1 weight gradient, which Inductor checks.
    p = _problem(context, precision, gated=gated, accumulate=False, single_grouped=precision == "mxfp8")
    schedule = functional.build_schedule(
        p["workspace"], p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )

    def step(x, router_weights, d_output):
        output, forward_context = functional.forward(
            p["config"], p["workspace"], schedule, x, router_weights, *p["shared"], *p["forward_routed"],
            **p["gate_kwargs"],
        )
        gradients = functional.backward(
            p["config"], p["workspace"], schedule, forward_context, d_output, x, router_weights, *p["shared"],
            *p["backward_routed"], **p["gate_kwargs"],
        )
        return (output, *gradients[:8]) + ((gradients[8],) if gated else ())

    def step_on_views(x_storage, router_weights, d_output_storage, offset):
        """Inductor realigns a misaligned input, but not a view taken inside the graph."""
        x = x_storage[offset:offset + p["x"].numel()].view(p["x"].shape)
        d_output = d_output_storage[offset:offset + p["d_output"].numel()].view(p["d_output"].shape)
        return step(x, router_weights, d_output)

    expected = step(p["x"], p["router_weights"], p["d_output"])
    torch.cuda.synchronize()
    torch._dynamo.reset()
    failures = []
    try:
        compiled = torch.compile(step, fullgraph=True)
        compiled_on_views = torch.compile(step_on_views, fullgraph=True, dynamic=False)
        for offset in (0, 2, 128):
            x_storage, elements = _offset_storage(p["x"], offset)
            d_output_storage, _ = _offset_storage(p["d_output"], offset)
            x = x_storage[elements:elements + p["x"].numel()].view(p["x"].shape)
            d_output = d_output_storage[elements:elements + p["d_output"].numel()].view(p["d_output"].shape)
            router_weights = _at_offset(p["router_weights"], offset - offset % 4)
            variants = {
                "inputs": lambda: compiled(x, router_weights, d_output),
                "views": lambda: compiled_on_views(x_storage, router_weights, d_output_storage, elements),
            }
            for kind, call in variants.items():
                dist.barrier()
                _poison(p["workspace"])
                try:
                    if offset == 0 and kind == "inputs":
                        actual, codes = run_and_get_code(call)
                    else:
                        actual = call()
                    torch.cuda.synchronize()
                except Exception as error:  # noqa: BLE001 (collect every variant's outcome)
                    failures.append(f"{kind} at a {offset}-byte offset: {type(error).__name__}: {error}"[:300])
                    continue
                differing = [
                    index for index, (reference, result) in enumerate(zip(expected, actual, strict=True))
                    if not torch.equal(reference, result)
                ]
                if differing:
                    failures.append(f"{kind} at a {offset}-byte offset: results {differing} differ from eager")
    finally:
        torch._dynamo.reset()
    assert not failures, failures
    # The generated code keeps the program order of the MoK ops: each staging write precedes its barrier.
    calls = re.findall(r"torch\.ops\.mok\.(\w+)\.default\(", "\n".join(codes))
    assert calls == [name.split(".")[1] for name in _expected_targets(precision)[3:]], calls


def _mok_calls(codes):
    """The MoK custom ops that Inductor's generated code calls, in order."""
    return re.findall(r"torch\.ops\.mok\.(\w+)\.default\(", "\n".join(codes))


@contextlib.contextmanager
def _inductor_code(check=None):
    """Collects the code that Inductor generates inside the block. Inductor hands each graph's code over before the
    graph first runs, also on a cache hit, so ``check(code)`` can stop a graph by raising. Unlike run_and_get_code, it
    does not reset Dynamo, which would also drop the Triton kernels that a lazily compiled backward graph refers to."""
    from unittest import mock

    from torch._inductor.graph import GraphLowering

    codes = []

    def save(code):
        if check is not None:
            check(code)
        codes.append(code)

    with mock.patch.object(GraphLowering, "save_output_code", save):
        yield codes


# MoKForwardContext fields in order; the routed ones are (data, scale) pairs for MXFP8.
_CONTEXT_FIELDS = ("x_routed", "gate_shared", "gate_routed", "up_shared", "up_routed", "hidden_shared", "hidden_routed")


@pytest.mark.xfail(
    strict=False,
    reason="pre-existing on head 9e89fb5: AOTAutograd's partitioner moves the backward's barriers, which do not depend "
    "on the incoming gradient, into the compiled forward graph, where they would wait forever on a copy of the barrier "
    "state; the test stops that graph before it runs (see Known limitations)",
)
@pytest.mark.parametrize(
    "precision,single_grouped,gated",
    [("bf16", False, False), ("bf16", True, True), ("mxfp8", True, False), ("mxfp8", True, True)],
)
def test_inductor_compiles_autograd_function(context, precision, single_grouped, gated):
    """MCore calls forward and backward from a torch.autograd.Function. Compiled with the default backend (one layer,
    one microbatch, one compiled region), the forward graph and the backward graph would each run their own MoK ops in
    program order, and with the workspace buffers poisoned before the forward and again before the backward, the
    output and gradients would equal eager ones bitwise. The backward's staging op takes the incoming gradient, so the
    partitioner keeps it in the backward graph; the backward's barriers do not depend on it. Each graph is checked
    before it first runs, because a graph that runs a barrier out of place, or on a copy of the barrier buffer, which
    peers never signal, would wait forever."""
    p = _problem(context, precision, gated=gated, accumulate=False, single_grouped=single_grouped)
    workspace = p["workspace"]
    schedule = functional.build_schedule(
        workspace, p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    paired = precision == "mxfp8"

    class MoE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, router_weights):
            output, forward_context = functional.forward(
                p["config"], workspace, schedule, x, router_weights, *p["shared"], *p["forward_routed"],
                **p["gate_kwargs"],
            )
            saved = []
            for name in _CONTEXT_FIELDS:
                value = getattr(forward_context, name)
                saved += list(value) if paired and name.endswith("_routed") else [value]
            if gated:
                saved += [forward_context.shared_output, forward_context.shared_output_gate]
            ctx.save_for_backward(x, router_weights, *saved)
            return output

        @staticmethod
        def backward(ctx, d_output):
            x, router_weights, *saved = ctx.saved_tensors
            values = []
            for name in _CONTEXT_FIELDS:
                if paired and name.endswith("_routed"):
                    values.append((saved.pop(0), saved.pop(0)))
                else:
                    values.append(saved.pop(0))
            gradients = functional.backward(
                p["config"], workspace, schedule, functional.MoKForwardContext(*values, *saved),
                d_output.contiguous(), x, router_weights, *p["shared"], *p["backward_routed"], **p["gate_kwargs"],
            )
            return gradients[0], gradients[1]

    def layer(x, router_weights):
        return MoE.apply(x, router_weights)

    def leaves():
        return p["x"].clone().requires_grad_(True), p["router_weights"].clone().requires_grad_(True)

    x, router_weights = leaves()
    output = layer(x, router_weights)
    output.backward(p["d_output"])
    torch.cuda.synchronize()
    expected = (output.detach(), x.grad, router_weights.grad)

    targets = [name.split(".")[1] for name in _expected_targets(precision)]
    # Inductor generates the forward graph's code first, and the backward graph's after it.
    graphs = [("forward", targets[3:8]), ("backward", targets[8:])]
    checked = []

    def check(code):
        name, expected_calls = graphs[len(checked)] if len(checked) < len(graphs) else ("extra", [])
        checked.append(name)
        calls = _mok_calls([code])
        on_copies = re.findall(r"torch\.ops\.mok\.barrier_all\.default\((buf\d+),", code)
        if calls != expected_calls or on_copies:
            raise AssertionError(
                f"the compiled {name} graph would run the MoK ops {calls}, not {expected_calls}"
                + (f", with barriers on copies {on_copies} of the barrier buffer" if on_copies else "")
            )

    torch._dynamo.reset()
    try:
        x, router_weights = leaves()
        dist.barrier()
        _poison(workspace)
        with _inductor_code(check):
            output = torch.compile(layer, fullgraph=True)(x, router_weights)
            torch.cuda.synchronize()
            dist.barrier()
            _poison(workspace)
            output.backward(p["d_output"])
            torch.cuda.synchronize()
    finally:
        torch._dynamo.reset()
    assert checked[:2] == ["forward", "backward"], checked
    for name, reference, result in zip(("output", "d_x", "d_router_weights"), expected,
                                       (output.detach(), x.grad, router_weights.grad), strict=True):
        assert torch.equal(reference, result), f"{name} differs from eager"


@pytest.mark.xfail(
    strict=False,
    reason="pre-existing on head 9e89fb5 and on the base 7a8cfb7: under memory pressure Inductor moves barriers ahead "
    "of the staging op and the megakernels, which do not depend on them (see Known limitations)",
)
@pytest.mark.parametrize("gated", [False, True])
def test_inductor_keeps_mok_order_under_memory_pressure(context, gated):
    """Inductor reorders independent operations when that lowers its estimated peak memory (reorder_for_peak_memory,
    on by default). In this graph, whose first product is used last and whose MoK inputs depend on another product,
    every MoK op would have to stay in program order. The workspace buffers are poisoned before each call, and the
    results would equal eager ones bitwise."""
    rank, _, device = context
    p = _problem(context, "mxfp8", gated=gated, accumulate=False)
    schedule = functional.build_schedule(
        p["workspace"], p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    generator = torch.Generator(device=device).manual_seed(11 + rank)
    c = torch.randn(4096, 4096, generator=generator, device=device)
    d = torch.randn(4096, 4096, generator=generator, device=device)
    a = torch.randn(64, 64, generator=generator, device=device) / 64
    b = torch.randn(64, 64, generator=generator, device=device) / 64

    def step(x_in, router_weights, d_in, a, b, c, d):
        big = torch.mm(c, d)  # first in program order, used last
        zero = 0.0 * torch.mm(a, b).sum()  # x and grad_output depend on it, and equal x_in and d_in
        x = (x_in.float() + zero).to(torch.bfloat16)
        d_output = (d_in.float() + zero).to(torch.bfloat16)
        output, forward_context = functional.forward(
            p["config"], p["workspace"], schedule, x, router_weights, *p["shared"], *p["forward_routed"],
            **p["gate_kwargs"],
        )
        gradients = functional.backward(
            p["config"], p["workspace"], schedule, forward_context, d_output, x, router_weights, *p["shared"],
            *p["backward_routed"], **p["gate_kwargs"],
        )
        return (output, *gradients[:8], *(gradients[8:] if gated else ())), big.sum(0)

    inputs = (p["x"], p["router_weights"], p["d_output"], a, b, c, d)
    expected, _ = step(*inputs)
    torch.cuda.synchronize()
    torch._dynamo.reset()
    failures = []
    try:
        compiled = torch.compile(step, fullgraph=True)
        for call in range(3):
            dist.barrier()
            _poison(p["workspace"])
            with _inductor_code() as codes:
                actual, _ = compiled(*inputs)
            torch.cuda.synchronize()
            differing = [
                index for index, (reference, result) in enumerate(zip(expected, actual, strict=True))
                if not torch.equal(reference, result)
            ]
            if differing:
                failures.append(f"call {call}: results {differing} differ from eager")
            if call == 0:
                calls = _mok_calls(codes)
                if calls != [name.split(".")[1] for name in _expected_targets("mxfp8")[3:]]:
                    failures.append(f"MoK ops out of program order: {calls}")
    finally:
        torch._dynamo.reset()
    assert not failures, failures


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("pre_dispatch", [False, True])
@pytest.mark.parametrize("tracing_mode", ["fake", "symbolic", "real"])
def test_make_fx_traces_forward_and_backward(context, monkeypatch, tracing_mode, pre_dispatch, gated):
    """Standalone make_fx traces build_schedule, forward and backward (gated or not, without main_grads) to the
    custom ops, with the gate backward as the reference torch operations. No data pointer is read, so fake tracing
    launches nothing, and the graph does not depend on the alignment of the traced inputs. A real-mode graph replays
    to the eager results bitwise."""
    from torch.fx.experimental.proxy_tensor import make_fx

    p = _problem(context, "mxfp8", gated=gated, accumulate=False)

    def step(x, top_experts, router_weights, d_output):
        schedule = functional.build_schedule(
            p["workspace"], p["config"], top_experts, num_local_experts=p["num_local_experts"]
        )
        output, forward_context = functional.forward(
            p["config"], p["workspace"], schedule, x, router_weights, *p["shared"], *p["forward_routed"],
            **p["gate_kwargs"],
        )
        gradients = functional.backward(
            p["config"], p["workspace"], schedule, forward_context, d_output, x, router_weights, *p["shared"],
            *p["backward_routed"], **p["gate_kwargs"],
        )
        return output, *gradients[:8], *(gradients[8:] if gated else ())

    inputs = (p["x"], p["top_experts"], p["router_weights"], p["d_output"])
    workspace = p["workspace"]
    buffers = (workspace.x_buffer, workspace.d_y_buffer, workspace.router_weight_buffer, workspace.barrier_target)
    torch.cuda.synchronize()
    dist.barrier()
    _poison(workspace)
    before = [buffer.clone() for buffer in buffers]
    graph = make_fx(step, tracing_mode=tracing_mode, pre_dispatch=pre_dispatch, _allow_non_fake_inputs=True)(*inputs)
    torch.cuda.synchronize()
    targets = [node.target for node in graph.graph.nodes if node.op == "call_function"]
    assert _mok_targets(targets) == _expected_targets("mxfp8")
    assert ("aten.sigmoid_backward.default" in map(str, targets)) == gated
    if tracing_mode != "real":
        for buffer, snapshot in zip(buffers, before, strict=True):
            assert torch.equal(buffer, snapshot) or (buffer.isnan() & snapshot.isnan()).all()
        return
    with monkeypatch.context() as patch:
        _reference_gate(patch)
        expected = step(*inputs)
    torch.cuda.synchronize()
    dist.barrier()
    _poison(workspace)
    actual = graph(*inputs)
    torch.cuda.synchronize()
    for index, (reference, result) in enumerate(zip(expected, actual, strict=True)):
        assert torch.equal(reference, result), f"result {index} differs from eager"


@pytest.mark.parametrize("gated", [False, True])
def test_fake_tensor_mode_runs_forward_and_backward(context, gated):
    """FakeTensorMode on its own runs build_schedule, forward and backward (gated or not) to outputs with the right
    metadata, without launching anything: the workspace buffers keep their contents."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    p = _problem(context, "mxfp8", gated=gated, accumulate=False)
    workspace = p["workspace"]
    torch.cuda.synchronize()
    dist.barrier()
    _poison(workspace)
    barrier_target = workspace.barrier_target.clone()
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = {name: mode.from_tensor(p[name]) for name in ("x", "top_experts", "router_weights", "d_output")}
        schedule = functional.build_schedule(
            workspace, p["config"], fake["top_experts"], num_local_experts=p["num_local_experts"]
        )
        output, forward_context = functional.forward(
            p["config"], workspace, schedule, fake["x"], fake["router_weights"], *p["shared"],
            *p["forward_routed"], **p["gate_kwargs"],
        )
        gradients = functional.backward(
            p["config"], workspace, schedule, forward_context, fake["d_output"], fake["x"], fake["router_weights"],
            *p["shared"], *p["backward_routed"], **p["gate_kwargs"],
        )
    torch.cuda.synchronize()
    assert all(type(tensor) is torch._subclasses.fake_tensor.FakeTensor for tensor in (output, *gradients[:8]))
    assert output.shape == p["x"].shape and output.dtype == torch.bfloat16
    assert gradients[0].shape == p["x"].shape and gradients[1].shape == p["router_weights"].shape
    assert (gradients[8] is not None) == gated
    assert torch.equal(workspace.barrier_target, barrier_target)
    for buffer in (workspace.x_buffer, workspace.d_y_buffer, workspace.router_weight_buffer):
        assert buffer.isnan().all()


@pytest.mark.parametrize("tracer", ["make_fx_fake", "make_fx_real", "make_fx_pre_dispatch", "fake_tensor_mode"])
def test_main_grads_cannot_be_traced(context, tracer):
    """backward with main_grads has no custom op, so under a tracer it raises a RuntimeError before it launches
    anything."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    p = _problem(context, "mxfp8", gated=True, accumulate=True)
    workspace = p["workspace"]
    schedule = functional.build_schedule(
        workspace, p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    _, forward_context = functional.forward(
        p["config"], workspace, schedule, p["x"], p["router_weights"], *p["shared"], *p["forward_routed"],
        **p["gate_kwargs"],
    )

    def backward(d_output):
        return functional.backward(
            p["config"], workspace, schedule, forward_context, d_output, p["x"], p["router_weights"], *p["shared"],
            *p["backward_routed"], main_grads=p["main_grads"], shared_output_gate_main_grad=p["gate_main_grad"],
            **p["gate_kwargs"],
        )

    torch.cuda.synchronize()
    barrier_target = workspace.barrier_target.clone()
    with pytest.raises(RuntimeError, match="main_grads"):
        if tracer == "fake_tensor_mode":
            with FakeTensorMode(allow_non_fake_inputs=True):
                backward(p["d_output"])
        else:
            mode, pre_dispatch = {
                "make_fx_fake": ("fake", False), "make_fx_real": ("real", False),
                "make_fx_pre_dispatch": ("real", True),
            }[tracer]
            make_fx(backward, tracing_mode=mode, pre_dispatch=pre_dispatch, _allow_non_fake_inputs=True)(p["d_output"])
    torch.cuda.synchronize()
    assert torch.equal(workspace.barrier_target, barrier_target)
    assert not any(main_grad.any() for main_grad in (*p["main_grads"], p["gate_main_grad"]))


@pytest.mark.parametrize(
    "transform,target",
    [("functionalize", "backward"), ("vmap", "backward"), ("grad", "backward"), ("vjp", "backward"),
     ("jvp", "backward"), ("jvp", "forward"), ("grad", "forward_gate_weight"), ("vmap", "backward_gate_weight")],
)
def test_func_transforms_raise_before_launching_on_wrapped_tensors(context, transform, target):
    """torch.func transforms are unsupported. Their wrapper tensors have the exact type torch.Tensor and hold no data
    of their own, and the custom ops implement no transform, so under a transform forward and backward (gated here)
    raise a RuntimeError before they launch anything: they neither launch a kernel on a null pointer nor return zero
    derivatives for MoK's part, as jvp, or grad with respect to the gate weight, would through the custom ops. The
    next plain step runs normally."""
    p = _problem(context, "mxfp8", gated=True, accumulate=False)
    expected = _step(p)
    workspace = p["workspace"]
    schedule = functional.build_schedule(
        workspace, p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    weight = p["gate_kwargs"]["shared_output_gate_weight"]
    _, forward_context = functional.forward(
        p["config"], workspace, schedule, p["x"], p["router_weights"], *p["shared"], *p["forward_routed"],
        shared_output_gate_weight=weight,
    )

    def forward(x=p["x"], w=weight):
        return functional.forward(
            p["config"], workspace, schedule, x, p["router_weights"], *p["shared"], *p["forward_routed"],
            shared_output_gate_weight=w,
        )[0]

    def backward(d_output=p["d_output"], w=weight):
        return functional.backward(
            p["config"], workspace, schedule, forward_context, d_output, p["x"], p["router_weights"],
            *p["shared"], *p["backward_routed"], shared_output_gate_weight=w,
        )[0]

    function, argument = {
        "forward": (lambda x: forward(x=x), p["x"]),
        "backward": (lambda d_output: backward(d_output=d_output), p["d_output"]),
        "forward_gate_weight": (lambda w: forward(w=w), weight),
        "backward_gate_weight": (lambda w: backward(w=w), weight),
    }[target]
    transformed = {
        "functionalize": lambda: torch.func.functionalize(function)(argument),
        "vmap": lambda: torch.func.vmap(function)(argument.unsqueeze(0)),
        "grad": lambda: torch.func.grad(lambda a: function(a).float().sum())(argument),
        "vjp": lambda: torch.func.vjp(function, argument)[1](torch.ones_like(p["x"])),
        "jvp": lambda: torch.func.jvp(function, (argument,), (torch.ones_like(argument),)),
    }[transform]
    torch.cuda.synchronize()
    dist.barrier()
    barrier_target = workspace.barrier_target.clone()
    with pytest.raises(RuntimeError, match="torch.func transforms"):
        transformed()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    # Nothing was launched: not even a barrier.
    assert torch.equal(workspace.barrier_target, barrier_target)
    actual = _step(p)
    torch.cuda.synchronize()
    for index, (reference, result) in enumerate(zip((expected[0], *expected[1]), (actual[0], *actual[1]), strict=True)):
        assert torch.equal(reference, result), f"result {index} differs after the transform"


@pytest.mark.parametrize(
    "case",
    ["forward_unused", "backward_unused", "forward_gate_weight", "backward_gate_weight",
     "backward_gate_weight_main_grads"],
)
def test_functionalize_never_launches_directly(context, case):
    """Inside torch.func.functionalize no dispatch mode is active and every tensor keeps the exact type torch.Tensor,
    but a C++ entry point launched there would allocate its outputs as functional tensors and write through them, an
    illegal memory access, even when no MoK operand is transformed (the *_unused cases transform only a dummy input)
    or when only the gate weight is. Such a call raises a RuntimeError before anything is launched (with main_grads,
    the accumulating backward's own check raises first), and the next plain step is unaffected."""
    accumulate = case.endswith("main_grads")
    gated = "gate_weight" in case
    p = _problem(context, "mxfp8", gated=gated, accumulate=accumulate)
    # A plain step first, which also fills the validated-launch cache with these tensors.
    output, gradients = _step(p)
    expected = (output, gradients[0], gradients[1])
    workspace = p["workspace"]
    schedule = functional.build_schedule(
        workspace, p["config"], p["top_experts"], num_local_experts=p["num_local_experts"]
    )
    _, forward_context = functional.forward(
        p["config"], workspace, schedule, p["x"], p["router_weights"], *p["shared"], *p["forward_routed"],
        **p["gate_kwargs"],
    )

    def forward(weight):
        kwargs = {"shared_output_gate_weight": weight} if gated else {}
        return functional.forward(
            p["config"], workspace, schedule, p["x"], p["router_weights"], *p["shared"], *p["forward_routed"],
            **kwargs,
        )[0]

    def backward(weight):
        kwargs = {"shared_output_gate_weight": weight} if gated else {}
        if accumulate:
            kwargs["shared_output_gate_main_grad"] = p["gate_main_grad"]
        return functional.backward(
            p["config"], workspace, schedule, forward_context, p["d_output"], p["x"], p["router_weights"],
            *p["shared"], *p["backward_routed"], main_grads=p["main_grads"], **kwargs,
        )[0]

    function = forward if case.startswith("forward") else backward
    if case.endswith("unused"):
        transformed = lambda dummy: function(None).float().sum() + dummy  # noqa: E731
        argument = torch.zeros((), device=p["x"].device)
    else:
        transformed, argument = function, p["gate_kwargs"]["shared_output_gate_weight"]
    torch.cuda.synchronize()
    dist.barrier()
    barrier_target = workspace.barrier_target.clone()
    with pytest.raises(RuntimeError) as error:
        torch.func.functionalize(transformed)(argument)
        torch.cuda.synchronize()
    assert "illegal memory access" not in str(error.value)
    assert ("main_grads" in str(error.value)) == accumulate
    torch.cuda.synchronize()
    # Nothing was launched: not even a barrier.
    assert torch.equal(workspace.barrier_target, barrier_target)
    if accumulate:
        for main_grad in (*p["main_grads"], p["gate_main_grad"]):
            main_grad.zero_()
    output, gradients = _step(p)
    torch.cuda.synchronize()
    for name, reference, result in zip(("output", "d_x", "d_router_weights"), expected,
                                       (output, gradients[0], gradients[1]), strict=True):
        assert torch.equal(reference, result), f"{name} differs after the transform"
