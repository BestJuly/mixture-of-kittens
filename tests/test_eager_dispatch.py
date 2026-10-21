"""Eager dispatch: functional launches MoK's C++ entry points directly and validates caller tensors once."""

import gc

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from mok import functional, ops

from .utils import generate_inputs

CUSTOM_OPS = (
    "all_gather_top_experts",
    "barrier_all",
    "schedule",
    "copy_routed_inputs",
    "copy_routed_grad_inputs",
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


def _problem(context, precision, *, gated=True, accumulate=True, single_grouped=True):
    """One rank's MoE problem with unrouted slots; single-grouped is MCore's argument layout."""
    rank, world_size, device = context
    num_local_experts = 2
    hidden_dim = intermediate_dim = 256
    topk = 2
    num_local_tokens = 512
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
                patch.setattr(functional, "_eager_dispatch", lambda tensor: False)
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


def test_dispatch_modes_see_the_custom_ops(context):
    problem = _problem(context, "bf16", gated=False, accumulate=False, single_grouped=False)
    with _RecordingMode() as mode:
        _step(problem)
    torch.cuda.synchronize()
    assert [name for name in mode.ops if name.startswith("mok.")] == [
        "mok.all_gather_top_experts.default",
        "mok.barrier_all.default",
        "mok.schedule.default",
        "mok.copy_routed_inputs.default",
        "mok.barrier_all.default",
        "mok.dispatch_mlp_swiglu_combine_fwd_bf16.default",
        "mok.barrier_all.default",
        "mok.fwd_epilogue.default",
        "mok.copy_routed_grad_inputs.default",
        "mok.barrier_all.default",
        "mok.dispatch_mlp_swiglu_combine_bwd_bf16.default",
        "mok.barrier_all.default",
        "mok.bwd_epilogue.default",
    ]
