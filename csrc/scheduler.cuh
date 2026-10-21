#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>

using namespace kittens;

namespace scheduler {

// Schedule order (unchanged from the original single-pass kernel, which ran one 1024-thread block per
// (local expert, peer) and rescanned the peer's whole route slice twice per block):
//   route x of peer p's (num_local_tokens * topk) slice belongs to reference thread a = x % 1024 at stride step
//   m = x / 1024; the routes of one (peer, local expert) are ranked j = 0, 1, ... in key order (a, m), and rank j
//   lands at row
//     expert_base[e] + sum_r min(count[r][e], j) + #{r < p : count[r][e] > j}
//   (round-robin interleave of the peers inside the expert's 256-padded segment).
// Every route is now visited twice in total instead of 2 * num_local_experts times:
//   count_kernel   grid (chunks, peers): local-expert histogram of one chunk (a range of reference threads with all
//                  their strides); also fills schedule_peer_rank with -1 and resets the done counter
//   scan_kernel    grid (expert groups, peers): exclusive prefix over chunks per (peer, expert) and per-peer totals;
//                  the last block to finish derives padded sizes, expert bases, num_tokens and traps on overflow
//   scatter_kernel grid (chunks, peers): ranks the chunk's routes in key order (warp match + per-warp counts) and
//                  writes each route's slot
// The result is bit-identical to the original kernels: schedule_peer_rank over the whole capacity,
// schedule_peer_token_idx on every routed slot (padding slots stay uninitialized as before), num_tokens and
// tokens_per_expert.

struct config {
    static constexpr int EXPERT_PADDING = 256;    // row alignment for contiguous grouped GEMM
    static constexpr int ORDER_THREADS = 1024;    // reference thread count that defines the route order
    static constexpr int CHUNK_ROUTES = 1024;     // target routes per chunk
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
    static constexpr int STAGE_ROUTES = 4096;     // routes staged in shared memory at a time by scatter_kernel
};

struct globals {
    const int *topk;                // (world_size, num_local_tokens * topk)
    int *schedule_peer_rank;        // (schedule_capacity,) -1 for padding slots
    int *schedule_peer_token_idx;   // (schedule_capacity,) original_token_idx * topk + k on routed slots
    int *num_tokens;                // (1,) total padded token count
    int *tokens_per_expert;         // (num_local_experts,) padded per-expert token counts
    int *chunk_counts;              // (world_size, num_chunks, num_local_experts) histogram, then exclusive prefix
    int *peer_counts;               // (world_size, num_local_experts) routes from each peer to each local expert
    int *expert_base;               // (num_local_experts,) first row of each expert's padded segment
    int *expert_min_count;          // (num_local_experts,) min over peers of peer_counts
    int *done;                      // (1,) scan_kernel block counter

    int routes_per_peer;            // num_local_tokens * topk
    int strides;                    // ceil(routes_per_peer / ORDER_THREADS)
    int chunk_threads;              // reference threads per chunk
    int num_chunks;                 // chunks per peer
    int stage_routes;               // min(STAGE_ROUTES, chunk_threads * strides)
    int world_size;
    int num_local_experts;
    int first_expert;
    int schedule_capacity;
};

static __device__ __forceinline__ int local_expert_of(const globals &G, const int *peer_topk, const int route) {
    if (route >= G.routes_per_peer) return -1;
    const int local_expert = __ldg(peer_topk + route) - G.first_expert;
    return static_cast<unsigned>(local_expert) < static_cast<unsigned>(G.num_local_experts) ? local_expert : -1;
}

// Stage 1: per-chunk local-expert histogram; -1 fill of schedule_peer_rank
static __global__ void __launch_bounds__(config::NUM_THREADS) count_kernel(const __grid_constant__ globals G) {
    extern __shared__ int counts[]; // (num_local_experts,)
    const int chunk = blockIdx.x;
    const int peer_rank = blockIdx.y;
    for (int i = threadIdx.x; i < G.num_local_experts; i += blockDim.x)
        counts[i] = 0;

    // This block's share of the -1 fill (16-byte stores; schedule_peer_rank comes from at::empty)
    {
        const int num_blocks = gridDim.x * gridDim.y;
        const int block = peer_rank * gridDim.x + chunk;
        const int num_vecs = G.schedule_capacity / 4;
        const int vecs_per_block = (num_vecs + num_blocks - 1) / num_blocks;
        const int vec_end = min(num_vecs, (block + 1) * vecs_per_block);
        int4 *vecs = reinterpret_cast<int4 *>(G.schedule_peer_rank);
        for (int i = block * vecs_per_block + threadIdx.x; i < vec_end; i += blockDim.x)
            vecs[i] = make_int4(-1, -1, -1, -1);
        if (block == num_blocks - 1)
            for (int i = num_vecs * 4 + threadIdx.x; i < G.schedule_capacity; i += blockDim.x)
                G.schedule_peer_rank[i] = -1;
        if (block == 0 && threadIdx.x == 0)
            *G.done = 0;
    }
    __syncthreads();

    const int first_thread = chunk * G.chunk_threads;
    const int chunk_threads = min(G.chunk_threads, config::ORDER_THREADS - first_thread);
    const int chunk_routes = chunk_threads * G.strides;
    const int *peer_topk = G.topk + static_cast<size_t>(peer_rank) * G.routes_per_peer;
    #pragma unroll 4
    for (int i = threadIdx.x; i < chunk_routes; i += blockDim.x) {
        const int stride = i / chunk_threads; // consecutive i -> consecutive routes (coalesced)
        const int route = first_thread + (i - stride * chunk_threads) + config::ORDER_THREADS * stride;
        const int local_expert = local_expert_of(G, peer_topk, route);
        if (local_expert >= 0) atomicAdd(&counts[local_expert], 1);
    }
    __syncthreads();

    int *chunk_counts = G.chunk_counts + (static_cast<size_t>(peer_rank) * G.num_chunks + chunk) * G.num_local_experts;
    for (int i = threadIdx.x; i < G.num_local_experts; i += blockDim.x)
        chunk_counts[i] = counts[i];
}

// Stage 2: exclusive prefix over chunks per (peer, local expert); the last block derives the expert segments
static __global__ void __launch_bounds__(config::NUM_THREADS) scan_kernel(const __grid_constant__ globals G) {
    extern __shared__ int counts_smem[]; // last block only: (world_size, num_local_experts) counts, then padded sizes
    __shared__ int warp_sums[config::NUM_WARPS];
    __shared__ bool is_last_block;
    const int lane = warp::laneid();
    const int warp_idx = warpid();
    const int peer_rank = blockIdx.y;
    const int local_expert = blockIdx.x * config::NUM_WARPS + warp_idx;

    if (local_expert < G.num_local_experts) {
        // num_chunks <= ORDER_THREADS: load the whole column first (one memory round trip), then scan it
        constexpr int MAX_TILES = config::ORDER_THREADS / WARP_THREADS;
        int *column = G.chunk_counts + static_cast<size_t>(peer_rank) * G.num_chunks * G.num_local_experts + local_expert;
        int counts[MAX_TILES];
        #pragma unroll
        for (int tile = 0; tile < MAX_TILES; ++tile) {
            const int chunk = tile * WARP_THREADS + lane;
            counts[tile] = chunk < G.num_chunks ? column[static_cast<size_t>(chunk) * G.num_local_experts] : 0;
        }
        int carry = 0;
        #pragma unroll
        for (int tile = 0; tile < MAX_TILES; ++tile) {
            if (tile * WARP_THREADS < G.num_chunks) {
                const int chunk = tile * WARP_THREADS + lane;
                int inclusive = counts[tile];
                #pragma unroll
                for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
                    const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
                    if (lane >= offset) inclusive += n;
                }
                if (chunk < G.num_chunks)
                    column[static_cast<size_t>(chunk) * G.num_local_experts] = carry + inclusive - counts[tile];
                carry += __shfl_sync(0xffffffff, inclusive, WARP_THREADS - 1);
            }
        }
        if (lane == 0) G.peer_counts[peer_rank * G.num_local_experts + local_expert] = carry;
    }

    // Last-block election: every block's peer_counts are visible to the block that takes the final ticket
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int ticket = atomicAdd(G.done, 1);
        __threadfence();
        is_last_block = ticket == static_cast<int>(gridDim.x * gridDim.y) - 1;
    }
    __syncthreads();
    if (!is_last_block) return;

    // All peers' counts in one memory round trip (written by other blocks: read through L2)
    int *padded = counts_smem + G.world_size * G.num_local_experts;
    for (int i = threadIdx.x; i < G.world_size * G.num_local_experts; i += blockDim.x)
        counts_smem[i] = __ldcg(G.peer_counts + i);
    __syncthreads();
    for (int e = threadIdx.x; e < G.num_local_experts; e += blockDim.x) {
        int total = 0;
        int min_count = 0x7fffffff;
        for (int r = 0; r < G.world_size; ++r) {
            const int count = counts_smem[r * G.num_local_experts + e];
            total += count;
            min_count = min(min_count, count);
        }
        const int padded_count = (total + config::EXPERT_PADDING - 1) / config::EXPERT_PADDING * config::EXPERT_PADDING;
        padded[e] = padded_count;
        G.tokens_per_expert[e] = padded_count;
        G.expert_min_count[e] = min_count;
    }
    __syncthreads();

    // Exclusive prefix of the padded sizes over local experts (each thread owns a contiguous run)
    const int per_thread = (G.num_local_experts + blockDim.x - 1) / blockDim.x;
    const int first = threadIdx.x * per_thread;
    const int last = min(G.num_local_experts, first + per_thread);
    int run = 0;
    for (int e = first; e < last; ++e) run += padded[e];
    int inclusive = run;
    #pragma unroll
    for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
        const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (lane >= offset) inclusive += n;
    }
    if (lane == WARP_THREADS - 1) warp_sums[warp_idx] = inclusive;
    __syncthreads();
    int base = inclusive - run;
    for (int w = 0; w < warp_idx; ++w) base += warp_sums[w];
    for (int e = first; e < last; ++e) {
        G.expert_base[e] = base;
        base += padded[e];
    }
    if (threadIdx.x == blockDim.x - 1) {
        int total = 0;
        for (int w = 0; w < config::NUM_WARPS; ++w) total += warp_sums[w];
        *G.num_tokens = total;
        if (total > G.schedule_capacity) asm volatile("{trap;}");
    }
}

// Stage 3: rank each chunk's routes in key order and write their schedule slots
static __global__ void __launch_bounds__(config::NUM_THREADS) scatter_kernel(const __grid_constant__ globals G) {
    extern __shared__ int smem[];
    int *peer_counts = smem;                                                       // (world_size, num_local_experts)
    int *expert_base = peer_counts + G.world_size * G.num_local_experts;           // (num_local_experts,)
    int *expert_min_count = expert_base + G.num_local_experts;                     // (num_local_experts,)
    int *running = expert_min_count + G.num_local_experts;                         // (num_local_experts,)
    int *warp_counts = running + G.num_local_experts;                              // (NUM_WARPS, num_local_experts)
    int *staged = warp_counts + config::NUM_WARPS * G.num_local_experts;          // (stage_routes,)

    const int chunk = blockIdx.x;
    const int peer_rank = blockIdx.y;
    const int lane = warp::laneid();
    const int warp_idx = warpid();
    const int num_local_experts = G.num_local_experts;
    const int world_size = G.world_size;

    const int *chunk_offsets = G.chunk_counts + (static_cast<size_t>(peer_rank) * G.num_chunks + chunk) * num_local_experts;
    for (int i = threadIdx.x; i < world_size * num_local_experts; i += blockDim.x)
        peer_counts[i] = G.peer_counts[i];
    for (int i = threadIdx.x; i < num_local_experts; i += blockDim.x) {
        expert_base[i] = G.expert_base[i];
        expert_min_count[i] = G.expert_min_count[i];
        running[i] = chunk_offsets[i];
    }
    for (int i = threadIdx.x; i < config::NUM_WARPS * num_local_experts; i += blockDim.x)
        warp_counts[i] = 0;

    const int first_thread = chunk * G.chunk_threads;
    const int chunk_threads = min(G.chunk_threads, config::ORDER_THREADS - first_thread);
    const int chunk_routes = chunk_threads * G.strides;
    const int *peer_topk = G.topk + static_cast<size_t>(peer_rank) * G.routes_per_peer;
    const unsigned lower_lanes = (1u << lane) - 1u;

    // Chunks spanning several reference threads hold at most CHUNK_ROUTES routes and are staged whole, in route
    // order (coalesced); single-thread chunks with more strides than STAGE_ROUTES are staged in stride windows.
    // The first window's loads overlap the table loads above; each window's last tile ends with a barrier.
    for (int window = 0; window < chunk_routes; window += G.stage_routes) {
        const int window_routes = min(G.stage_routes, chunk_routes - window);
        #pragma unroll 4
        for (int i = threadIdx.x; i < window_routes; i += blockDim.x) {
            const int position = window + i; // stride-major staging order within the chunk
            const int stride = position / chunk_threads;
            const int route = first_thread + (position - stride * chunk_threads) + config::ORDER_THREADS * stride;
            staged[i] = local_expert_of(G, peer_topk, route);
        }
        __syncthreads(); // staged routes and the shared tables are visible

        for (int tile = 0; tile < window_routes; tile += config::NUM_THREADS) {
            // Key order inside the chunk: reference thread major, stride minor
            const int key = window + tile + threadIdx.x;
            int local_expert = -1;
            int route = 0;
            if (tile + threadIdx.x < window_routes) {
                const int thread_in_chunk = key / G.strides;
                const int stride = key - thread_in_chunk * G.strides;
                route = first_thread + thread_in_chunk + config::ORDER_THREADS * stride;
                local_expert = staged[stride * chunk_threads + thread_in_chunk - window];
            }
            const unsigned group = __match_any_sync(0xffffffff, local_expert);
            const int rank_in_warp = __popc(group & lower_lanes);
            if (local_expert >= 0 && rank_in_warp == 0)
                warp_counts[warp_idx * num_local_experts + local_expert] = __popc(group);
            __syncthreads();

            if (local_expert >= 0) {
                int j = running[local_expert] + rank_in_warp;
                for (int w = 0; w < warp_idx; ++w)
                    j += warp_counts[w * num_local_experts + local_expert];
                int dst = expert_base[local_expert];
                if (j < expert_min_count[local_expert]) {
                    dst += world_size * j + peer_rank; // every peer still has a route at rank j
                } else {
                    for (int r = 0; r < world_size; ++r) {
                        const int count = peer_counts[r * num_local_experts + local_expert];
                        dst += min(count, j);
                        dst += (r < peer_rank && count > j) ? 1 : 0;
                    }
                }
                G.schedule_peer_rank[dst] = peer_rank;
                G.schedule_peer_token_idx[dst] = route; // original_token_idx * topk + k
            }
            __syncthreads();

            for (int e = threadIdx.x; e < num_local_experts; e += blockDim.x) {
                int total = 0;
                #pragma unroll
                for (int w = 0; w < config::NUM_WARPS; ++w) {
                    total += warp_counts[w * num_local_experts + e];
                    warp_counts[w * num_local_experts + e] = 0;
                }
                running[e] += total;
            }
            __syncthreads();
        }
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule_entrypoint(
    const at::Tensor &topk_all,
    const int num_local_experts,
    const int schedule_capacity,
    const int rank
) {
    TORCH_CHECK(topk_all.is_cuda() && topk_all.scalar_type() == at::kInt && topk_all.is_contiguous() &&
                topk_all.dim() == 3, "MoK: topk_all must be contiguous CUDA int32 (world_size, num_local_tokens, topk)");
    TORCH_CHECK(num_local_experts > 0 && schedule_capacity > 0, "MoK: invalid schedule arguments");
    const int world_size = static_cast<int>(topk_all.size(0));
    const int routes_per_peer = static_cast<int>(topk_all.size(1) * topk_all.size(2));
    TORCH_CHECK(world_size > 0 && routes_per_peer > 0, "MoK: topk_all must not be empty");

    globals G;
    G.routes_per_peer = routes_per_peer;
    G.strides = (routes_per_peer + config::ORDER_THREADS - 1) / config::ORDER_THREADS;
    G.chunk_threads = std::max(1, config::CHUNK_ROUTES / G.strides);
    G.num_chunks = (config::ORDER_THREADS + G.chunk_threads - 1) / G.chunk_threads;
    G.stage_routes = std::min(config::STAGE_ROUTES, G.chunk_threads * G.strides);
    G.world_size = world_size;
    G.num_local_experts = num_local_experts;
    G.first_expert = rank * num_local_experts;
    G.schedule_capacity = schedule_capacity;

    auto int_options = topk_all.options().dtype(at::kInt);
    at::Tensor schedule_peer_rank = at::empty({schedule_capacity}, int_options);
    at::Tensor schedule_peer_token_idx = at::empty({schedule_capacity}, int_options);
    at::Tensor num_tokens = at::empty({1}, int_options);
    at::Tensor tokens_per_expert = at::empty({num_local_experts}, int_options);
    const int64_t chunk_counts_size = static_cast<int64_t>(world_size) * G.num_chunks * num_local_experts;
    const int64_t peer_counts_size = static_cast<int64_t>(world_size) * num_local_experts;
    at::Tensor scratch = at::empty({chunk_counts_size + peer_counts_size + 2 * num_local_experts + 1}, int_options);

    G.topk = topk_all.data_ptr<int>();
    G.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    G.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    G.num_tokens = num_tokens.data_ptr<int>();
    G.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    G.chunk_counts = scratch.data_ptr<int>();
    G.peer_counts = G.chunk_counts + chunk_counts_size;
    G.expert_base = G.peer_counts + peer_counts_size;
    G.expert_min_count = G.expert_base + num_local_experts;
    G.done = G.expert_min_count + num_local_experts;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(G.schedule_peer_rank) % 16 == 0,
                "MoK: schedule_peer_rank allocation must be 16-byte aligned");

    const int scatter_smem = static_cast<int>(sizeof(int)) *
        ((world_size + 3 + config::NUM_WARPS) * num_local_experts + G.stage_routes);
    const int count_smem = static_cast<int>(sizeof(int)) * num_local_experts;
    const int scan_smem = static_cast<int>(sizeof(int)) * (world_size + 1) * num_local_experts;
    const int max_smem = static_cast<int>(at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin);
    TORCH_CHECK(scatter_smem <= max_smem && scan_smem <= max_smem,
                "MoK: too many local experts or EP ranks for the scheduler");
    if (scatter_smem > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(scatter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, scatter_smem));
    if (scan_smem > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, scan_smem));
    if (count_smem > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, count_smem));

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 chunk_grid(G.num_chunks, world_size);
    count_kernel<<<chunk_grid, config::NUM_THREADS, count_smem, stream>>>(G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 scan_grid((num_local_experts + config::NUM_WARPS - 1) / config::NUM_WARPS, world_size);
    scan_kernel<<<scan_grid, config::NUM_THREADS, scan_smem, stream>>>(G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_kernel<<<chunk_grid, config::NUM_THREADS, scatter_smem, stream>>>(G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert};
}

} // namespace scheduler
