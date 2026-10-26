# Mixture-of-Kittens (MoK)

Mixture-of-Kittens (MoK) is a fully deterministic mixture-of-experts (MoE) training megakernel built from first principles for NVL72s. MoK fuses all MoE computation and communication into a single kernel, overlapping compute and inter-GPU networking at configurable granularity, and fully eliminates CPU-GPU synchronization. It supports BF16 and MXFP8, covers both forward and backward passes, and powers production training of Composer at Cursor.

For a deep dive, read our [blog post](https://cursor.com/blog/mixture-of-kittens).

## Performance

We evaluated MoK at two levels: standalone MoE layers, with all benchmark code available in [`benchmarks`](./benchmarks), and end-to-end training on our internal production stack across multiple NVL72 racks. The following figure summarizes the standalone MoE layer benchmarks:

![MoE benchmark results](./figures/moe_benchmarks.png)

Compared with the fastest baseline, MoK is up to 2.37x faster for the MXFP8 forward, 1.78x for the MXFP8 backward, 1.92x for the BF16 forward, and 1.58x for the BF16 backward.

See our [blog post](https://cursor.com/blog/mixture-of-kittens) for the methodology and full results.

## Setup

### Requirements

- NVIDIA Blackwell SM100 or SM103 GPUs (e.g., GB200 NVL72 or GB300 NVL72)
- Python 3.12 or later
- PyTorch 2.10 or later
- Triton (`mok.functional` imports it for the shared output gate backward)
- CUDA toolkit 13.0 or later

### PyTorch and CUDA

The installed PyTorch build must target CUDA 13.0+, and its CUDA version must match the major and minor version of the system CUDA toolkit. MoK is built and tested against CUDA 13.0, but later versions should work as well.

For example, to install PyTorch 2.10 built against CUDA 13.0:

```bash
python -m pip install "torch==2.10.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

### Installation

By default, MoK builds for SM103. Install from the repository root with either:

```bash
pip install . --no-build-isolation
```

or:

```bash
python setup.py install
```

To build for SM100 instead, set the `MOK_ARCH` environment variable:

```bash
MOK_ARCH=SM100 pip install . --no-build-isolation
```

To build one wheel for both SM100 and SM103, set `MOK_ARCH=ALL`:

```bash
MOK_ARCH=ALL pip wheel . --no-build-isolation --no-deps
```

This includes `sm_100a` and `sm_103a` kernels using the BF16/MXFP8 features shared
by both targets. For direct Makefile builds, use `make ARCH=ALL`.

To verify the installation:

```bash
python -c "import mok; print(mok.__version__)"
```

The `qizhang/mok-fused-grad-accum` integration branch carries one
ThunderKittens change as a repository-local patch so that all referenced
submodule commits remain fetchable from their upstream repositories. Prepare
the submodule before building this branch:

```bash
./scripts/prepare_thunderkittens.sh
make -B ARCH=SM103
```

The preparation script is idempotent. It enables the native MN-major MXFP8 B
operand consumed by MCore single-grouped parameters.

### Development

For development, install MoK in editable mode once, then use `make` for fast rebuilds:

```bash
pip install -e . --no-build-isolation
make
```

#### Unit tests

Build configuration tests run without PyTorch, CUDA, or GPUs:

```bash
python -m unittest tests.test_build -v
```

Launch multi-GPU unit tests through `torchrun` and `pytest`:

```bash
torchrun --standalone --nproc-per-node=<num-gpus> -m pytest -s <test-path>
```

## Getting Started

MoK provides two layers of abstraction:

- **Ops layer** (`mok/ops.py`): the low-level API that lets you call our CUDA kernel implementations directly with minimal overhead. With this, you fully manage your own data (e.g., symmetric tensors) and coordinate the kernel calls yourself to implement the MoE layer's computation and communication.
- **Functional layer** (`mok/functional.py`): the higher-level API that handles more for you, such as maintaining scratch memory and coordinating kernel launches. It consists of workspace creation functions, a schedule function, and forward/backward functions.

The functional layer is our choice for production training, so we recommend it unless you have specific needs. We explain only the functional layer from here.

With the functional API, MoK is simple to use: call `schedule(...)` once to build the dispatch/combine schedule, then pass that same schedule to `forward(...)` and `backward(...)`. Before doing so, however, you need to define and manage a **config** and a **workspace**.

### Config

MoK exposes 5 hyperparameters that can affect the performance of MoE execution. Because optimal values depend heavily on the workload, you should tune and sweep them before using MoK in production.

- `fwd_num_comm_sms`: the number of communication SMs during forward. We recommend values between 4 and 52.
- `bwd_num_comm_sms`: the number of communication SMs during backward. We recommend values between 4 and 52.
- `minibatch_size`: the granularity of computation–communication overlap. This is an important parameter in MoK, and you must tune it properly to get optimal performance. We recommend values between 2048 and 16384.
- `macrobatch_size`: the token ring buffer size. Setting this to a large value (e.g., 131072) means the ring buffer is used only once. You should maximize this value to fill the available GPU memory.
- `schedule_capacity_multiplier`: defaults to 0.5. This should be set to the worst-case fraction of tokens routed to a single rank. Setting it to 1 assumes the absolute worst case (all tokens routed to one rank) but adds kernel scheduling overhead (due to expert padding, the actual worst case is slightly above 1.0). Ideally, use a higher value during the first steps of training when expert imbalance is bad, then reduce it to around 0.5 in later steps. Note that decreasing this value does not save GPU memory meaningfully, as the schedule table is at most a few megabytes.

You can set these values when creating the `MoKConfig` dataclass, which you pass to all functional-layer functions.

### Workspace

MoK relies on PyTorch symmetric memory to allocate and manage inter-GPU symmetric buffers, or identically sized memory allocations across many GPUs (i.e., all GPUs in an EP group). These buffers serve as the source/destination of token dispatch/combine, along with a few other purposes. We call the entire collection of symmetric memory, along with the other metadata and scratchpads MoK needs, the **workspace**. We provide a data structure and functions for creating and destroying workspaces.

Because symmetric buffers are expensive to allocate, we recommend creating a single workspace per model and reusing it across layers. For this, we provide the `get_workspace(...)` function, which automatically caches workspaces with identical properties (EP group, device, model shapes, etc.). If you prefer to manage a workspace's lifetime yourself, the `create_workspace(...)` function does not cache.

### MXFP8

To run MoK in MXFP8 mode, pass the activations as-is in BF16 while prequantizing the weights to MXFP8. We *could* quantize the weights inside our kernels, but prequantizing leaves better opportunities for things like FSDP, so we keep it separate and provide the `mxfp8_quantize(...)` function at the ops layer so you can prequantize the weights yourself.

### BF16 shared output gate

MoK supports an optional, bias-free BF16 output gate `W_g[1, H]`:
`Y = routed_output + sigmoid(linear(X, W_g)) * shared_output`. It is distinct
from the shared SwiGLU gate projection. Gate projection and sigmoid use
PyTorch operations, and the shared-output multiplication is fused into the
MOK forward epilogue. The gate backward runs two fused Triton kernels
(`mok/_gate_fused.py`). They keep every BF16 rounding step of the PyTorch
operations they replace; only the FP32 summation order of the gate gradient
differs. The kernels run on plain tensors in eager mode and under
`torch.compile`, which traces them. When `grad_output` or the saved shared
output or gate is a tensor subclass, or under `make_fx`, `FakeTensorMode` or
another dispatch mode, the gate backward runs those PyTorch operations
instead; a tensor-subclass gate weight sends only the merge into `d_x` to
them.

Example with BF16 routed and shared weights:

```python
weights = (w_shared_gate, w_shared_up, w_shared_down,
           w_routed_gate, w_routed_up, w_routed_down)
output, context = functional.forward(
    config, workspace, schedule, x, router_weights, *weights,
    shared_output_gate_weight=w_g,
)
gradients = functional.backward(
    config, workspace, schedule, context, d_output, x, router_weights, *weights,
    shared_output_gate_weight=w_g,
)
d_w_g = gradients[8]  # Fresh FP32 gradient, without a BF16 result intermediate.
```

`backward` now always returns **nine** entries: the original eight gradients
followed by the output-gate gradient (`None` when ungated). An optional
`shared_output_gate_main_grad` BF16 or FP32 `[1, H]` buffer receives additive
FP32 gate-gradient contributions and is returned by reference. A BF16 buffer
rounds when each addition is written back; the contribution is not first cast
to BF16. Without this buffer, the fresh gate gradient remains FP32. The gate
buffer is independent of the six MLP `main_grads` buffers. Migrate existing
eight-item unpacking accordingly.

Gated forward supports BF16 or MXFP8 routed experts. Shared MLP weights and
saved activations, and the output-gate weight, logits and sigmoid output, are
BF16. The epilogue multiplies shared output by the gate and accumulates routed
outputs in FP32 before storing BF16. The returned context retains the pre-gate
shared output and sigmoid output; preserve it and keep the weights unchanged
until backward. Shared backward consumes the BF16 gated gradient; routed
experts retain their original upstream gradient and quantization path.
Gated `recompute_forward_context` is unsupported; internal routed replay is
unchanged. Ungated BF16/MXFP8 retain their original numerical paths.

### Router-weight gradients of unrouted slots

`backward` zeroes `d_router_weights` wherever `top_experts` is `-1`. A caller
that reads `d_router_weights` only at routed slots can skip this masking with
`backward(..., zero_unrouted_router_grads=False)`. The unrouted entries are
then unspecified: zero, or a gradient that an earlier call wrote there.

### Eager launches and cached validation

For plain tensors in eager mode, the functional layer launches its kernels
directly instead of through the `mok.ops` custom ops. A call makes all of its
kernel launches through the custom ops instead when any of its tensor
arguments is a tensor subclass (activations, routes, router weights, schedule,
forward context, weights and scales, descriptor tables, main grads, or the
gate weight and its main grad), and under `torch.compile`, a dispatch mode
such as `FakeTensorMode` or `make_fx`, or pre-dispatch tracing
(`make_fx(pre_dispatch=True)`, `torch.export`); under a `torch.func`
transform, it raises instead. Only direct launches read data pointers: a
direct launch reads an `x` or `grad_output` that is 256-byte aligned in place
and copies only its routed rows into the workspace buffers, and copies a
misaligned one in full; the gate and its weight gradient then read the same
copy as the shared expert, so the results do not depend on the alignment of
the inputs. Every other call copies `x`, the router weights and `grad_output`
in full into the workspace buffers, through one call of the
`mok.ops.stage_into_buffers` custom op per direction, so a traced graph does
not depend on the alignment of its inputs and keeps these copies ahead of the
kernels whose peers read the buffers.

What each mode supports:

| Mode | `build_schedule`, `forward`, `backward` | `backward` with `main_grads` |
| --- | --- | --- |
| Eager, plain tensors (CUDA graph capture included) | Direct launches | Direct launches |
| Eager, a tensor-subclass argument | Custom ops | `TypeError` |
| `torch.compile` | Custom ops (see the known limitations below) | Not traceable: `fullgraph=True` fails, otherwise a graph break |
| `make_fx` (fake, symbolic or real; pre-dispatch or not), `FakeTensorMode` | Custom ops, traced | `RuntimeError` |
| `torch.func` transforms | Unsupported: `RuntimeError` | Unsupported: `RuntimeError` |
| `torch.jit.trace` | Unsupported: the direct launches are not recorded | Unsupported |
| Other dispatch modes, plain tensors | Custom ops | Accumulating wrappers |

`recompute_forward_context` launches its kernels through the custom ops in
every mode. In eager mode with plain tensors it stages `x` with a plain copy;
otherwise it stages through `stage_into_buffers`.

`backward` with `main_grads` accumulates the weight gradients through eager
wrappers that are not custom ops and hand every operand to the C++ kernels,
so it raises before launching anything when it cannot run them safely. Pass
`main_grads=None` to get fresh weight gradients through the custom ops.

The caller's activations and routes are checked on every call. Weights, main
grads and descriptor tables are checked when they are first used, and the
functional layer then remembers them by tensor identity. Updating their
values in place, as an optimizer step does, needs nothing else. If you change
such a tensor while keeping the same Python object, for example by assigning
a tensor of another shape or dtype to its `.data`, call
`functional.clear_workspace_cache()` before the next call. In the same way,
`get_workspace` validates a repeated set of arguments once and
`build_schedule` validates its config once per device and shape;
`clear_workspace_cache()` clears these caches as well.

#### Known limitations

These hold with and without the direct launches:

- **Order of the MoK ops under `torch.compile`.** The custom ops order each
  barrier relative to the work around it only through data dependencies, and
  a barrier depends on nothing but the barrier counter: the staging op and
  the megakernels do not depend on the barrier before them. Inductor keeps the
  program order of the MoK ops in the graphs that the tests compile, but a
  pass that reorders independent nodes, such as its peak-memory reordering
  with large unrelated work in the same graph, can move a barrier past the
  staging op or a megakernel.
- **Compiled `torch.autograd.Function`.** Calling `forward` and `backward`
  with `main_grads=None` from an autograd function that `torch.compile`
  compiles hangs, even with one layer and one microbatch: AOTAutograd's
  partitioner moves the backward's barriers, which do not depend on the
  incoming gradient, into the compiled forward graph. There they run on a
  copy of the barrier state, which peers never signal. (The backward's
  staging op takes the incoming gradient, so it stays in the backward graph.)
  With `main_grads`, as MCore's MoE layer passes them, Dynamo cannot trace
  the accumulating backward and breaks the graph there instead, and the call
  runs correctly; `fullgraph=True` fails. A compiled function that calls
  `forward` and `backward` directly, in one graph, is not affected.
- **`torch.func` transforms and `torch.jit.trace`** are unsupported. Under a
  transform (`functionalize`, `vmap`, `grad`, `vjp`, `jvp`, ...),
  `build_schedule`, `forward`, `recompute_forward_context` and `backward`
  raise a `RuntimeError` before they launch anything: the custom ops implement
  no transform, and some transforms would otherwise return zero derivatives
  for MoK's part without an error. `torch.jit.trace` does not record the
  direct launches.

### Example (MXFP8 forward and backward using the functional layer)

The following is a canonical example of implementing MoE forward and backward with MoK in MXFP8 mode:

```python
import torch.distributed as dist

from mok import functional, ops

# Inputs:
#   num_local_tokens:    int
#   hidden_size:         int
#   intermediate_size:   int
#   topk:                int
#   num_local_experts:   int
#   x:                   torch.bfloat16 [num_local_tokens, hidden_size]
#   topk_experts:        torch.int64    [num_local_tokens, topk]
#   router_weights:      torch.float32  [num_local_tokens, topk]
#   w_shared_gate:       torch.bfloat16 [intermediate_size, hidden_size]
#   w_shared_up:         torch.bfloat16 [intermediate_size, hidden_size]
#   w_shared_down:       torch.bfloat16 [hidden_size, intermediate_size]
#   w_routed_gate:       torch.bfloat16 [num_local_experts, intermediate_size, hidden_size]
#   w_routed_up:         torch.bfloat16 [num_local_experts, intermediate_size, hidden_size]
#   w_routed_down:       torch.bfloat16 [num_local_experts, hidden_size, intermediate_size]
#   d_output:            torch.bfloat16 [num_local_tokens, hidden_size]

config = functional.MoKConfig() # tune for your workload
workspace = functional.get_workspace(
    config,
    dist.group.WORLD, # replace with your expert-parallel process group
    device=x.device,
    num_local_tokens=num_local_tokens,
    hidden_size=hidden_size,
    topk=topk,
)


########################
# Weight quantization
########################

(
    w_routed_gate_fp8,
    w_routed_gate_sc,
    w_routed_gate_t_fp8,
    w_routed_gate_t_sc,
) = ops.mxfp8_quantize(w_routed_gate, True, True)
(
    w_routed_up_fp8,
    w_routed_up_sc,
    w_routed_up_t_fp8,
    w_routed_up_t_sc,
) = ops.mxfp8_quantize(w_routed_up, True, True)
(
    w_routed_down_fp8,
    w_routed_down_sc,
    w_routed_down_t_fp8,
    w_routed_down_t_sc,
) = ops.mxfp8_quantize(w_routed_down, True, True)


########################
# Forward
########################

schedule = functional.build_schedule(
    workspace,
    config,
    topk_experts,
    num_local_experts=num_local_experts,
)
output, forward_context = functional.forward(
    config,
    workspace,
    schedule,
    x,
    router_weights,
    w_shared_gate,
    w_shared_up,
    w_shared_down,
    (w_routed_gate_fp8, w_routed_gate_sc),
    (w_routed_up_fp8, w_routed_up_sc),
    (w_routed_down_fp8, w_routed_down_sc),
)

# Save `schedule` and `forward_context` for the backward


########################
# Backward
########################

(
    d_x,
    d_router_weights,
    d_w_routed_gate,
    d_w_routed_up,
    d_w_routed_down,
    d_w_shared_gate,
    d_w_shared_up,
    d_w_shared_down,
    d_w_shared_output_gate,  # None: this MXFP8 example is ungated.
) = functional.backward(
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
    (w_routed_gate_fp8, w_routed_gate_sc, w_routed_gate_t_fp8, w_routed_gate_t_sc),
    (w_routed_up_fp8, w_routed_up_sc, w_routed_up_t_fp8, w_routed_up_t_sc),
    (w_routed_down_t_fp8, w_routed_down_t_sc),
)
```

### Manual activation checkpointing

The example above saves the first macrobatch's forward intermediates for the backward pass. While we recommend this pattern, you can avoid storing them by discarding the returned context and rebuilding it immediately before backward with `functional.recompute_forward_context`.

```python
# Forward pass
output, discarded_forward_context = functional.forward(...)  # same call as above
del discarded_forward_context

# Backward pass
forward_context = functional.recompute_forward_context(
    config,
    workspace,
    schedule,
    x,
    w_shared_gate,
    w_shared_up,
    (w_routed_gate_fp8, w_routed_gate_sc),
    (w_routed_up_fp8, w_routed_up_sc),
)
gradients = functional.backward(...)  # same backward call as above
```

## Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

Mixture-of-Kittens is released under the Apache 2.0 License. See [`LICENSE`](LICENSE).

## Citation

If you use this work, please cite:

```
Stuart H. Sul, Nash Brown, Henry Wildermuth, William Lin, and Federico Cassano. "Mixture-of-Kittens: MoE Megakernel for NVL72s." Cursor Research, Aug 2026. https://github.com/cursor/mixture-of-kittens
```

Or in BibTeX:

```bibtex
@misc{sul2026mok,
  title        = {Mixture-of-Kittens: {MoE} Megakernel for {NVL72s}},
  author       = {Stuart H. Sul and Nash Brown and Henry Wildermuth and William Lin and Federico Cassano},
  organization = {Cursor Research},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/cursor/mixture-of-kittens}},
}
```
