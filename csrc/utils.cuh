#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty_like.h>
#include <optional>
#include <vector>

using namespace kittens;

namespace utils {

// Row-tiled helpers of the register epilogue fallback and the routed-row copy: a warp owns one (token,
// 1024-column segment) task and moves 16-byte vectors; a token's routes are read once per task, one slot per lane.
namespace rows {

struct config {
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
    static constexpr int VEC = 8;                // BF16 elements per 16-byte access
    static constexpr int VECS_PER_LANE = 4;      // 16-byte accesses per lane per task
    static constexpr int SEGMENT = WARP_THREADS * VEC * VECS_PER_LANE; // columns per task
};

struct task {
    int token;
    int col;   // first column of this lane's first vector; vector v covers [col + v * WARP_THREADS * VEC, +VEC)
};

static __device__ __forceinline__ bool get_task(task &t, const int num_tokens, const int hidden) {
    const int segments = (hidden + config::SEGMENT - 1) / config::SEGMENT;
    const int idx = blockIdx.x * config::NUM_WARPS + warpid();
    if (idx >= num_tokens * segments) return false;
    t.token = idx / segments;
    t.col = (idx - t.token * segments) * config::SEGMENT + warp::laneid() * config::VEC;
    return true;
}

static __device__ __forceinline__ bool has_vec(const task &t, const int v, const int hidden) {
    return t.col + v * WARP_THREADS * config::VEC < hidden;
}

static __device__ __forceinline__ void load_row(uint4 (&dst)[config::VECS_PER_LANE], const bf16 *row,
                                                const task &t, const int hidden) {
    #pragma unroll
    for (int v = 0; v < config::VECS_PER_LANE; ++v)
        if (has_vec(t, v, hidden))
            dst[v] = __ldcg(reinterpret_cast<const uint4 *>(row + t.col + v * WARP_THREADS * config::VEC));
}

static __device__ __forceinline__ void store_row(bf16 *row, const uint4 (&src)[config::VECS_PER_LANE],
                                                 const task &t, const int hidden) {
    #pragma unroll
    for (int v = 0; v < config::VECS_PER_LANE; ++v)
        if (has_vec(t, v, hidden))
            *reinterpret_cast<uint4 *>(row + t.col + v * WARP_THREADS * config::VEC) = src[v];
}

static __device__ __forceinline__ float bf16_lo(const uint32_t pair) { return __uint_as_float(pair << 16); }
static __device__ __forceinline__ float bf16_hi(const uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

// The same per-element round-to-nearest-even conversion as group::store of an rv_fl into an sv_bf
static __device__ __forceinline__ uint32_t pack_bf16(const float lo, const float hi) {
    const bf16 l = base_types::convertor<bf16, float>::convert(lo);
    const bf16 h = base_types::convertor<bf16, float>::convert(hi);
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&l)) |
           (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&h)) << 16);
}

} // namespace rows

// Combines the shared and routed expert rows of every token (forward: y_shared [* gate] + sum_k w_k * combine_k;
// backward: d_x_shared + sum_k d_x_routed_k). Only the routed slots (top_experts >= 0) are read: the original
// kernels loaded every slot but added only the routed ones. The arithmetic is unchanged and per element: an FP32
// accumulator initialized from the BF16 shared row (times the FP32 gate when gated), routed terms added in slot
// order (acc + term * weight, contracted to one FMA as before), one round-to-nearest-even BF16 conversion. Rows
// without routes (padding) are therefore still written, as the shared row (times the gate), bit-identical to the
// original kernels.
namespace combine {

struct globals {
    const bf16 *shared;       // (num_tokens, hidden)
    const bf16 *routed;       // (num_tokens * topk, hidden), row token * topk + k
    const float *weights;     // (num_tokens, topk), forward only
    const int *top_experts;   // (num_tokens, topk); -1 marks no route
    const bf16 *gate;         // (num_tokens,), gated forward only
    bf16 *output;             // (num_tokens, hidden)
    int num_tokens;
    int hidden;
    int topk;
    int ring;                 // gather kernel: row slots per warp
};

// acc = shared row (times the gate), for the 8 elements of one 16-byte vector
template <bool GATED>
static __device__ __forceinline__ void init_acc(float (&acc)[rows::config::VEC], const uint4 &shared, const float gate) {
    const uint32_t words[4] = {shared.x, shared.y, shared.z, shared.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc[2 * i] = rows::bf16_lo(words[i]);
        acc[2 * i + 1] = rows::bf16_hi(words[i]);
        if constexpr (GATED) {
            acc[2 * i] = acc[2 * i] * gate;
            acc[2 * i + 1] = acc[2 * i + 1] * gate;
        }
    }
}

// acc += routed row (times the router weight)
template <bool WEIGHTED>
static __device__ __forceinline__ void add_routed(float (&acc)[rows::config::VEC], const uint4 &routed, const float weight) {
    const uint32_t words[4] = {routed.x, routed.y, routed.z, routed.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (WEIGHTED) {
            acc[2 * i] = acc[2 * i] + rows::bf16_lo(words[i]) * weight;
            acc[2 * i + 1] = acc[2 * i + 1] + rows::bf16_hi(words[i]) * weight;
        } else {
            acc[2 * i] = acc[2 * i] + rows::bf16_lo(words[i]);
            acc[2 * i + 1] = acc[2 * i + 1] + rows::bf16_hi(words[i]);
        }
    }
}

static __device__ __forceinline__ uint4 pack_acc(const float (&acc)[rows::config::VEC]) {
    return make_uint4(rows::pack_bf16(acc[0], acc[1]), rows::pack_bf16(acc[2], acc[3]),
                      rows::pack_bf16(acc[4], acc[5]), rows::pack_bf16(acc[6], acc[7]));
}

// Main path: every warp gathers batches of token segments into its own shared-memory ring with cp.async and then
// combines them. A batch takes as many of the warp's next token segments as fit in the ring's row slots (one slot for
// the shared row, one per routed row), so mostly-padded and dense microbatches alike keep a full ring in flight per
// warp; the next batch's routes are fetched while the current batch's rows are in flight. The ring holds topk + 1
// slots (at least MIN_RING): one dense token segment, so the most warps fit per SM. Each lane copies and later reads
// only its own 16-byte columns, so a lane's cp.async.wait_group is the only synchronization the rows need; the
// per-slot router weights are exchanged through shared memory with __syncwarp.
struct gather_config {
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int SEGMENT = 512;                    // columns per token segment
    static constexpr int VECS_PER_LANE = SEGMENT / (WARP_THREADS * rows::config::VEC);
    static constexpr int MIN_RING = 9;
    static constexpr int MAX_RING = 32;                    // ring weights fit the 128-byte pad
    static constexpr int MAX_TOPK = MAX_RING - 1;
    static constexpr int ring_slots(const int topk) { return topk + 1 > MIN_RING ? topk + 1 : MIN_RING; }
    static constexpr int warp_shared_memory(const int ring) { return ring * SEGMENT * static_cast<int>(sizeof(bf16)) + 128; }
    static constexpr int dynamic_shared_memory(const int ring) { return NUM_WARPS * warp_shared_memory(ring); }
    static_assert(SEGMENT % (WARP_THREADS * rows::config::VEC) == 0, "a segment is whole 16-byte vectors per lane");
    static_assert(MAX_RING * static_cast<int>(sizeof(float)) <= 128, "ring weights must fit the padding");
};

static __device__ __forceinline__ void cp_async_16(void *smem, const void *gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "l"(gmem) : "memory");
}

// One candidate token segment per lane: its routes (bitmask over slots k < topk), router weights, gate, row slots
template <int MAX_TOPK>
struct candidate {
    int token;
    int col;
    int slots;            // 1 + routed slots; ring + 1 for no candidate
    uint32_t routed;
    float gate;
    float weight[MAX_TOPK];
};

template <typename C, bool WEIGHTED, bool GATED>
static __device__ __forceinline__ void load_candidate(candidate<C::MAX_TOPK> &c, const globals &g, const bool valid,
                                                      const int item, const int segments) {
    c.token = 0;
    c.col = 0;
    c.slots = g.ring + 1; // never fits
    c.routed = 0;
    c.gate = 1.0f;
    if (!valid) return;
    c.token = item / segments;
    c.col = (item - c.token * segments) * C::SEGMENT;
    const size_t route_row = static_cast<size_t>(c.token) * g.topk;
    #pragma unroll
    for (int k = 0; k < C::MAX_TOPK; ++k) {
        if (k < g.topk) {
            c.routed |= (__ldg(g.top_experts + route_row + k) >= 0 ? 1u : 0u) << k;
            if constexpr (WEIGHTED) c.weight[k] = __ldg(g.weights + route_row + k);
        }
    }
    if constexpr (GATED) c.gate = __bfloat162float(g.gate[c.token]);
    c.slots = 1 + __popc(c.routed);
}

template <bool WEIGHTED, bool GATED, typename C = gather_config>
static __global__ void __launch_bounds__(C::NUM_THREADS)
gather_combine_kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[];
    const int lane = warp::laneid();
    const int warp_idx = warpid();
    uint8_t *warp_smem = reinterpret_cast<uint8_t *>(&__shm[0]) + warp_idx * C::warp_shared_memory(g.ring);
    bf16 *ring = reinterpret_cast<bf16 *>(warp_smem);                                                    // (ring, SEGMENT)
    float *ring_weights = reinterpret_cast<float *>(warp_smem + g.ring * C::SEGMENT * sizeof(bf16)); // (ring,)

    const int segments = (g.hidden + C::SEGMENT - 1) / C::SEGMENT;
    const int num_items = g.num_tokens * segments;          // item = token * segments + segment
    const int total_warps = gridDim.x * C::NUM_WARPS;
    const int first_item = blockIdx.x * C::NUM_WARPS + warp_idx; // this warp's items: first_item + i * total_warps
    const int warp_items = first_item < num_items ? (num_items - first_item + total_warps - 1) / total_warps : 0;

    // Lane c of a batch holds candidate next + c (the warp's item index)
    candidate<C::MAX_TOPK> cand;
    int next = 0;
    load_candidate<C, WEIGHTED, GATED>(cand, g, lane < warp_items, first_item + lane * total_warps, segments);
    while (next < warp_items) {
        int last_slot = cand.slots; // inclusive prefix; fits is monotone since every candidate needs >= 1 slot
        #pragma unroll
        for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
            const int n = __shfl_up_sync(0xffffffff, last_slot, offset);
            if (lane >= offset) last_slot += n;
        }
        const bool fits = last_slot <= g.ring;
        const int batch = __popc(__ballot_sync(0xffffffff, fits)); // >= 1: the first candidate always fits
        const int first_slot = last_slot - cand.slots;

        // Copy the batch's rows, each lane its own 16-byte columns
        for (int i = 0; i < batch; ++i) {
            const int item_token = __shfl_sync(0xffffffff, cand.token, i);
            const int item_col = __shfl_sync(0xffffffff, cand.col, i);
            uint32_t item_routed = __shfl_sync(0xffffffff, cand.routed, i);
            int slot = __shfl_sync(0xffffffff, first_slot, i);
            const bf16 *src = g.shared + static_cast<size_t>(item_token) * g.hidden + item_col;
            const size_t route_row = static_cast<size_t>(item_token) * g.topk;
            while (true) {
                #pragma unroll
                for (int v = 0; v < C::VECS_PER_LANE; ++v) {
                    const int offset = v * WARP_THREADS * rows::config::VEC + lane * rows::config::VEC;
                    if (item_col + offset < g.hidden)
                        cp_async_16(ring + slot * C::SEGMENT + offset, src + offset);
                }
                if (item_routed == 0) break;
                const int k = __ffs(item_routed) - 1;
                item_routed &= item_routed - 1;
                src = g.routed + (route_row + k) * g.hidden + item_col;
                ++slot;
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if constexpr (WEIGHTED) {
            if (fits) {
                int slot = first_slot + 1;
                #pragma unroll
                for (int k = 0; k < C::MAX_TOPK; ++k)
                    if ((cand.routed >> k) & 1u) ring_weights[slot++] = cand.weight[k];
            }
        }
        const int token = cand.token, col = cand.col, slots = cand.slots;
        const float gate = cand.gate;

        // The next batch's routes load while this batch's rows are in flight
        next += batch;
        load_candidate<C, WEIGHTED, GATED>(cand, g, next + lane < warp_items,
                                           first_item + (next + lane) * total_warps, segments);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncwarp();

        for (int i = 0; i < batch; ++i) {
            const int item_token = __shfl_sync(0xffffffff, token, i);
            const int item_col = __shfl_sync(0xffffffff, col, i);
            const int item_first = __shfl_sync(0xffffffff, first_slot, i);
            const int item_slots = __shfl_sync(0xffffffff, slots, i);
            const float item_gate = GATED ? __shfl_sync(0xffffffff, gate, i) : 1.0f;
            bf16 *out = g.output + static_cast<size_t>(item_token) * g.hidden + item_col;
            #pragma unroll
            for (int v = 0; v < C::VECS_PER_LANE; ++v) {
                const int offset = v * WARP_THREADS * rows::config::VEC + lane * rows::config::VEC;
                if (item_col + offset < g.hidden) {
                    const bf16 *column = ring + item_first * C::SEGMENT + offset;
                    float acc[rows::config::VEC];
                    init_acc<GATED>(acc, *reinterpret_cast<const uint4 *>(column), item_gate);
                    for (int r = 1; r < item_slots; ++r) {
                        const float w = WEIGHTED ? ring_weights[item_first + r] : 1.0f;
                        add_routed<WEIGHTED>(acc, *reinterpret_cast<const uint4 *>(column + r * C::SEGMENT), w);
                    }
                    *reinterpret_cast<uint4 *>(out + offset) = pack_acc(acc);
                }
            }
        }
        __syncwarp(); // the ring weights are rewritten by the next batch
    }
}

// Fallback for topk > gather_config::MAX_TOPK: one warp per (token, 1024-column segment), rows through registers.
template <bool WEIGHTED, bool GATED>
static __global__ void __launch_bounds__(rows::config::NUM_THREADS) combine_rows_kernel(const __grid_constant__ globals g) {
    constexpr int VECS = rows::config::VECS_PER_LANE;
    rows::task t;
    if (!rows::get_task(t, g.num_tokens, g.hidden)) return;
    const int lane = warp::laneid();
    const size_t route_row = static_cast<size_t>(t.token) * g.topk;

    // The first slot group's routes are requested before the shared row is converted (independent loads)
    int expert = lane < g.topk ? __ldg(g.top_experts + route_row + lane) : -1;
    float weight = 0.0f;
    if constexpr (WEIGHTED) weight = lane < g.topk ? __ldg(g.weights + route_row + lane) : 0.0f;

    float acc[VECS][rows::config::VEC];
    {
        uint4 shared[VECS];
        rows::load_row(shared, g.shared + static_cast<size_t>(t.token) * g.hidden, t, g.hidden);
        const float gate = GATED ? __bfloat162float(g.gate[t.token]) : 1.0f;
        #pragma unroll
        for (int v = 0; v < VECS; ++v)
            init_acc<GATED>(acc[v], shared[v], gate);
    }

    for (int first_slot = 0; first_slot < g.topk; first_slot += WARP_THREADS) {
        if (first_slot > 0) { // topk > 32: next slot group
            const int slot = first_slot + lane;
            expert = slot < g.topk ? __ldg(g.top_experts + route_row + slot) : -1;
            if constexpr (WEIGHTED) weight = slot < g.topk ? __ldg(g.weights + route_row + slot) : 0.0f;
        }
        uint32_t routed = __ballot_sync(0xffffffff, expert >= 0);

        // Routed slots in increasing k; the next slot's rows are in flight while the current one is accumulated
        uint4 cur[VECS], next[VECS];
        int k = routed ? __ffs(routed) - 1 : -1;
        if (k >= 0) {
            routed &= routed - 1;
            rows::load_row(cur, g.routed + (route_row + first_slot + k) * g.hidden, t, g.hidden);
        }
        while (k >= 0) {
            const int next_k = routed ? __ffs(routed) - 1 : -1;
            if (next_k >= 0) {
                routed &= routed - 1;
                rows::load_row(next, g.routed + (route_row + first_slot + next_k) * g.hidden, t, g.hidden);
            }
            const float w = WEIGHTED ? __shfl_sync(0xffffffff, weight, k) : 1.0f;
            #pragma unroll
            for (int v = 0; v < VECS; ++v)
                add_routed<WEIGHTED>(acc[v], cur[v], w);
            #pragma unroll
            for (int v = 0; v < VECS; ++v) cur[v] = next[v];
            k = next_k;
        }
    }

    uint4 out[VECS];
    #pragma unroll
    for (int v = 0; v < VECS; ++v)
        out[v] = pack_acc(acc[v]);
    rows::store_row(g.output + static_cast<size_t>(t.token) * g.hidden, out, t, g.hidden);
}

static __host__ void check_activation(const at::Tensor &t, const char *name, const int64_t rows, const int64_t cols,
                                      const at::Device &device) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == at::kBFloat16 && t.is_contiguous() &&
                t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
                "MoK: ", name, " must be contiguous CUDA BF16 [", rows, ", ", cols, "] on the input device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "MoK: ", name, " must be 16-byte aligned");
}

static __host__ void check_routes(const at::Tensor &t, const char *name, const int64_t rows, const int64_t cols,
                                  const at::ScalarType dtype, const at::Device &device) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype && t.is_contiguous() &&
                t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
                "MoK: ", name, " must be contiguous CUDA [", rows, ", ", cols, "] ", dtype, " on the input device");
}

// Launches the cp.async gather kernel as one resident wave of persistent warps; false if it cannot run this shape.
template <bool WEIGHTED, bool GATED, typename C = gather_config>
static __host__ bool launch_gather(globals g, cudaStream_t stream) {
    const cudaDeviceProp *props = at::cuda::getCurrentDeviceProperties();
    const int64_t items = static_cast<int64_t>(g.num_tokens) * ((g.hidden + C::SEGMENT - 1) / C::SEGMENT);
    g.ring = C::ring_slots(g.topk);
    const int smem = C::dynamic_shared_memory(g.ring);
    if (g.topk > C::MAX_TOPK || items >= (int64_t{1} << 30) || smem > static_cast<int>(props->sharedMemPerBlockOptin))
        return false;
    static thread_local int configured_device = -1;
    static thread_local int blocks_per_sm[C::MAX_RING + 1];
    const int device = at::cuda::current_device();
    auto kernel = gather_combine_kernel<WEIGHTED, GATED, C>;
    if (configured_device != device) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            std::min(C::dynamic_shared_memory(C::MAX_RING),
                                                     static_cast<int>(props->sharedMemPerBlockOptin))));
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
        for (int &b : blocks_per_sm) b = 0;
        configured_device = device;
    }
    if (blocks_per_sm[g.ring] == 0) {
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm[g.ring], kernel, C::NUM_THREADS, smem));
        blocks_per_sm[g.ring] = std::max(blocks_per_sm[g.ring], 1);
    }
    const int64_t warps = std::max<int64_t>(1, std::min<int64_t>(
        static_cast<int64_t>(blocks_per_sm[g.ring]) * props->multiProcessorCount * C::NUM_WARPS, items));
    const int blocks = static_cast<int>((warps + C::NUM_WARPS - 1) / C::NUM_WARPS);
    kernel<<<blocks, C::NUM_THREADS, smem, stream>>>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

template <bool WEIGHTED, bool GATED>
static __host__ void launch(const globals &g) {
    if (g.num_tokens == 0 || g.hidden == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (launch_gather<WEIGHTED, GATED>(g, stream)) return;
    const int segments = (g.hidden + rows::config::SEGMENT - 1) / rows::config::SEGMENT;
    const int64_t tasks = static_cast<int64_t>(g.num_tokens) * segments;
    const int blocks = static_cast<int>((tasks + rows::config::NUM_WARPS - 1) / rows::config::NUM_WARPS);
    combine_rows_kernel<WEIGHTED, GATED><<<blocks, rows::config::NUM_THREADS, 0, stream>>>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace combine

namespace fwd_epilogue {

static __host__ at::Tensor fwd_epilogue_entrypoint(
    const at::Tensor &y_shared,
    const at::Tensor &combine_buffer,
    const at::Tensor &topk_weights,
    const at::Tensor &top_experts,
    const std::optional<at::Tensor> &shared_output_gate = std::nullopt
) {
    TORCH_CHECK(y_shared.dim() == 2 && topk_weights.dim() == 2, "MoK: fwd_epilogue expects 2-D inputs");
    const int64_t num_tokens = y_shared.size(0);
    const int64_t hidden = y_shared.size(1);
    const int64_t topk = topk_weights.size(1);
    TORCH_CHECK(hidden % rows::config::VEC == 0, "MoK: hidden size must be a multiple of 8");
    combine::check_activation(y_shared, "y_shared", num_tokens, hidden, y_shared.device());
    combine::check_activation(combine_buffer, "combine_buffer", num_tokens * topk, hidden, y_shared.device());
    combine::check_routes(topk_weights, "topk_weights", num_tokens, topk, at::kFloat, y_shared.device());
    combine::check_routes(top_experts, "top_experts", num_tokens, topk, at::kInt, y_shared.device());
    if (shared_output_gate.has_value()) {
        TORCH_CHECK(shared_output_gate->is_cuda() &&
                    shared_output_gate->scalar_type() == at::kBFloat16 &&
                    shared_output_gate->is_contiguous() &&
                    shared_output_gate->device() == y_shared.device() &&
                    shared_output_gate->dim() == 2 &&
                    shared_output_gate->size(0) == y_shared.size(0) &&
                    shared_output_gate->size(1) == 1,
                    "MoK: shared_output_gate must be contiguous CUDA BF16 [T, 1] on y_shared's device");
    }
    at::Tensor output = at::empty_like(y_shared);
    combine::globals g {
        .shared = reinterpret_cast<const bf16 *>(y_shared.data_ptr<at::BFloat16>()),
        .routed = reinterpret_cast<const bf16 *>(combine_buffer.data_ptr<at::BFloat16>()),
        .weights = topk_weights.data_ptr<float>(),
        .top_experts = top_experts.data_ptr<int>(),
        .gate = nullptr,
        .output = reinterpret_cast<bf16 *>(output.data_ptr<at::BFloat16>()),
        .num_tokens = static_cast<int>(num_tokens),
        .hidden = static_cast<int>(hidden),
        .topk = static_cast<int>(topk),
    };
    if (shared_output_gate.has_value()) {
        g.gate = reinterpret_cast<const bf16 *>(shared_output_gate->data_ptr<at::BFloat16>());
        combine::launch<true, true>(g);
    } else {
        combine::launch<true, false>(g);
    }
    return output;
}

} // namespace fwd_epilogue

namespace bwd_epilogue {

static __host__ at::Tensor bwd_epilogue_entrypoint(
    const at::Tensor &d_x_shared,
    const at::Tensor &d_x_routed_buffer,
    const at::Tensor &top_experts
) {
    TORCH_CHECK(d_x_shared.dim() == 2 && d_x_routed_buffer.dim() == 2 && d_x_shared.size(0) > 0,
                "MoK: bwd_epilogue expects 2-D inputs");
    const int64_t num_tokens = d_x_shared.size(0);
    const int64_t hidden = d_x_shared.size(1);
    const int64_t topk = d_x_routed_buffer.size(0) / num_tokens;
    TORCH_CHECK(hidden % rows::config::VEC == 0, "MoK: hidden size must be a multiple of 8");
    combine::check_activation(d_x_shared, "d_x_shared", num_tokens, hidden, d_x_shared.device());
    combine::check_activation(d_x_routed_buffer, "d_x_routed_buffer", num_tokens * topk, hidden, d_x_shared.device());
    combine::check_routes(top_experts, "top_experts", num_tokens, topk, at::kInt, d_x_shared.device());
    at::Tensor d_x = at::empty_like(d_x_shared);
    combine::globals g {
        .shared = reinterpret_cast<const bf16 *>(d_x_shared.data_ptr<at::BFloat16>()),
        .routed = reinterpret_cast<const bf16 *>(d_x_routed_buffer.data_ptr<at::BFloat16>()),
        .weights = nullptr,
        .top_experts = top_experts.data_ptr<int>(),
        .gate = nullptr,
        .output = reinterpret_cast<bf16 *>(d_x.data_ptr<at::BFloat16>()),
        .num_tokens = static_cast<int>(num_tokens),
        .hidden = static_cast<int>(hidden),
        .topk = static_cast<int>(topk),
    };
    combine::launch<false, false>(g);
    return d_x;
}

} // namespace bwd_epilogue

// Stages a rank's inputs in its symmetric buffers: copies the rows of each source that have at least one routed
// slot (top_experts >= 0) into the matching buffer and leaves the other rows untouched (peers read only routed
// rows, through the buffer pointers; the shared expert reads the caller's tensors), and copies the full router
// weights.
namespace copy_routed_rows {

static constexpr int MAX_PAIRS = 2;

struct globals {
    const int *top_experts;            // (num_tokens, topk)
    const bf16 *src[MAX_PAIRS];        // (num_tokens, hidden)
    bf16 *dst[MAX_PAIRS];              // (num_tokens, hidden)
    const float *weights;              // (num_tokens, topk) or nullptr
    float *weights_dst;                // (num_tokens, topk) or nullptr
    int num_pairs;
    int num_tokens;
    int hidden;
    int topk;
};

static __global__ void __launch_bounds__(rows::config::NUM_THREADS) copy_routed_rows_kernel(const __grid_constant__ globals g) {
    rows::task t;
    if (!rows::get_task(t, g.num_tokens, g.hidden)) return;
    const int lane = warp::laneid();
    const size_t route_row = static_cast<size_t>(t.token) * g.topk;
    bool routed = false;
    for (int first_slot = 0; first_slot < g.topk && !routed; first_slot += WARP_THREADS) {
        const int slot = first_slot + lane;
        routed = __any_sync(0xffffffff, slot < g.topk && __ldg(g.top_experts + route_row + slot) >= 0);
    }
    if (g.weights != nullptr && t.col < WARP_THREADS * rows::config::VEC) // the token's first segment
        for (int slot = lane; slot < g.topk; slot += WARP_THREADS)
            g.weights_dst[route_row + slot] = __ldg(g.weights + route_row + slot);
    if (!routed) return;
    uint4 vecs[MAX_PAIRS][rows::config::VECS_PER_LANE];
    #pragma unroll
    for (int p = 0; p < MAX_PAIRS; ++p)
        if (p < g.num_pairs)
            rows::load_row(vecs[p], g.src[p] + static_cast<size_t>(t.token) * g.hidden, t, g.hidden);
    #pragma unroll
    for (int p = 0; p < MAX_PAIRS; ++p)
        if (p < g.num_pairs)
            rows::store_row(g.dst[p] + static_cast<size_t>(t.token) * g.hidden, vecs[p], t, g.hidden);
}

static __host__ void copy_routed_rows_entrypoint(
    const at::Tensor &top_experts,
    const std::vector<at::Tensor> &sources,
    const std::vector<at::Tensor> &destinations,
    const std::optional<at::Tensor> &weights = std::nullopt,
    const std::optional<at::Tensor> &weights_destination = std::nullopt
) {
    TORCH_CHECK(top_experts.dim() == 2, "MoK: top_experts must be 2-D");
    TORCH_CHECK(!sources.empty() && sources.size() <= MAX_PAIRS && sources.size() == destinations.size(),
                "MoK: copy_routed_rows takes one or two source/destination pairs");
    TORCH_CHECK(weights.has_value() == weights_destination.has_value(),
                "MoK: router weights and their destination must be given together");
    const int64_t num_tokens = top_experts.size(0);
    const int64_t topk = top_experts.size(1);
    const int64_t hidden = sources[0].dim() == 2 ? sources[0].size(1) : -1;
    TORCH_CHECK(hidden > 0 && hidden % rows::config::VEC == 0, "MoK: hidden size must be a positive multiple of 8");
    const at::Device device = top_experts.device();
    combine::check_routes(top_experts, "top_experts", num_tokens, topk, at::kInt, device);
    globals g {};
    g.top_experts = top_experts.data_ptr<int>();
    g.num_pairs = static_cast<int>(sources.size());
    for (size_t p = 0; p < sources.size(); ++p) {
        combine::check_activation(sources[p], "source", num_tokens, hidden, device);
        combine::check_activation(destinations[p], "destination", num_tokens, hidden, device);
        g.src[p] = reinterpret_cast<const bf16 *>(sources[p].data_ptr<at::BFloat16>());
        g.dst[p] = reinterpret_cast<bf16 *>(destinations[p].data_ptr<at::BFloat16>());
    }
    if (weights.has_value()) {
        combine::check_routes(*weights, "router_weights", num_tokens, topk, at::kFloat, device);
        combine::check_routes(*weights_destination, "router_weight_buffer", num_tokens, topk, at::kFloat, device);
        g.weights = weights->data_ptr<float>();
        g.weights_dst = weights_destination->data_ptr<float>();
    }
    g.num_tokens = static_cast<int>(num_tokens);
    g.hidden = static_cast<int>(hidden);
    g.topk = static_cast<int>(topk);
    const int segments = (g.hidden + rows::config::SEGMENT - 1) / rows::config::SEGMENT;
    const int64_t tasks = num_tokens * segments;
    if (tasks == 0) return;
    const int blocks = static_cast<int>((tasks + rows::config::NUM_WARPS - 1) / rows::config::NUM_WARPS);
    copy_routed_rows_kernel<<<blocks, rows::config::NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace copy_routed_rows

namespace all_gather_top_experts {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    int *local_ptr;
    int *multicast_ptr;
    int rank;
    int numel;
    int chunk_bytes;

    __host__ inline dim3 grid() const {
        return dim3(numel * sizeof(int) / chunk_bytes);
    }

    __host__ inline int dynamic_shared_memory() const {
        return chunk_bytes + 1024;
    }
};

__device__ __forceinline__ void all_gather_top_experts_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int *shared = &al.allocate<int>();
    __shared__ semaphore arrived;
    const int offset = blockIdx.x * G.chunk_bytes / sizeof(int);

    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, G.chunk_bytes);
    tma::load_async(shared, G.local_ptr + offset, G.chunk_bytes, arrived);
    wait(arrived, 0);
    tma::store_async(G.multicast_ptr + G.rank * G.numel + offset, shared, G.chunk_bytes);
    tma::store_async_wait();
}

static __host__ void all_gather_top_experts_entrypoint(
    const at::Tensor &top_experts,
    const at::Tensor &all_gather_top_experts_buffer,
    int64_t all_gather_top_experts_buffer_multicast_ptr,
    int rank,
    int chunk_bytes
) {
    (void)all_gather_top_experts_buffer;
    const int numel = top_experts.numel();

    globals G {
        .local_ptr = top_experts.data_ptr<int>(),
        .multicast_ptr = reinterpret_cast<int *>(all_gather_top_experts_buffer_multicast_ptr),
        .rank = rank,
        .numel = numel,
        .chunk_bytes = chunk_bytes,
    };
    kittens::py::launch_kernel<config, globals, all_gather_top_experts_kernel>(G);
}

} // namespace all_gather_top_experts

namespace barrier_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    uint32_t *multicast_ptr;
    uint32_t *local_ptr;
    uint32_t *target_ptr;
    uint32_t ep_size;
};

__device__ __forceinline__ void barrier_all_kernel(const globals &G) {
    const uint32_t target = atomicAdd(G.target_ptr, G.ep_size) + G.ep_size;

    asm volatile("{multimem.red.release.sys.global.add.u32 [%0], 1;}" :: "l"(G.multicast_ptr) : "memory");
    // The counter is updated through its multicast alias and polled through its unicast alias
    asm volatile("{fence.proxy.alias;}" ::: "memory");

    uint32_t value;
    do {
        asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}" : "=r"(value) : "l"(G.local_ptr) : "memory");
    } while (value < target);
    asm volatile("{fence.acquire.sys;}" ::: "memory");
}

static __host__ void barrier_all_entrypoint(
    const at::Tensor &barrier_buffer,
    const std::vector<int64_t> &barrier_buffer_ptrs,
    int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &target
) {
    const int64_t barrier_buffer_local_ptr = reinterpret_cast<int64_t>(barrier_buffer.data_ptr<int>());
    const int64_t target_ptr = reinterpret_cast<int64_t>(target.data_ptr<int>());

    globals G {
        .multicast_ptr = reinterpret_cast<uint32_t *>(barrier_buffer_multicast_ptr),
        .local_ptr = reinterpret_cast<uint32_t *>(barrier_buffer_local_ptr),
        .target_ptr = reinterpret_cast<uint32_t *>(target_ptr),
        .ep_size = static_cast<uint32_t>(barrier_buffer_ptrs.size()),
    };
    kittens::py::launch_kernel<config, globals, barrier_all_kernel>(G);
}

} // namespace barrier_all

namespace zero_empty_routed_wgrads {

struct globals {
    uint16_t *d_w_routed_gate;
    uint16_t *d_w_routed_up;
    uint16_t *d_w_routed_down;
    const int *tokens_per_expert;
    const int64_t elements_per_expert;
};

static __global__ void zero_empty_routed_wgrads_kernel(const __grid_constant__ globals g) {
    const int expert_idx = blockIdx.y;
    if (g.tokens_per_expert[expert_idx] != 0) return;
    const int64_t expert_offset = expert_idx * g.elements_per_expert;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < g.elements_per_expert; idx += gridDim.x * blockDim.x) {
        g.d_w_routed_gate[expert_offset + idx] = 0;
        g.d_w_routed_up[expert_offset + idx] = 0;
        g.d_w_routed_down[expert_offset + idx] = 0;
    }
}

} // namespace zero_empty_routed_wgrads

} // namespace utils
