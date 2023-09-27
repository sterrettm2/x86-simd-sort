#ifndef XSS_NETWORK_QSORT
#define XSS_NETWORK_QSORT

#include "avx512-common-qsort.h"
#include "xss-optimal-networks.hpp"

template <typename vtype,
          int64_t numVecs,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_FINLINE void bitonic_clean_n_vec(reg_t *regs)
{
X86_SIMD_SORT_UNROLL_LOOP(512)
    for (int num = numVecs / 2; num >= 2; num /= 2) {
X86_SIMD_SORT_UNROLL_LOOP(512)
        for (int j = 0; j < numVecs; j += num) {
X86_SIMD_SORT_UNROLL_LOOP(512)
            for (int i = 0; i < num / 2; i++) {
                COEX<vtype>(regs[i + j], regs[i + j + num / 2]);
            }
        }
    }
}

template <typename vtype,
          int64_t numVecs,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_FINLINE void bitonic_sort_n_vec(reg_t *regs)
{
    if constexpr (numVecs == 1){
        return;
    }else if constexpr (numVecs == 2){
        COEX<vtype>(regs[0], regs[1]);
    }else if constexpr (numVecs == 4){
        optimal_sort_4<vtype>(regs);
    }else if constexpr (numVecs == 8){
        optimal_sort_8<vtype>(regs);
    }else if constexpr (numVecs == 16){
        optimal_sort_16<vtype>(regs);
    }else if constexpr (numVecs == 32){
        optimal_sort_32<vtype>(regs);
    }else{
        // TODO should we remove this branch? I believe it is never used in the current code
        bitonic_sort_n_vec<vtype, numVecs / 2>(regs);
        bitonic_sort_n_vec<vtype, numVecs / 2>(regs + numVecs / 2);
        
        X86_SIMD_SORT_UNROLL_LOOP(64)
        for (int i = 0; i < numVecs / 2; i++){
            COEX<vtype>(regs[i], regs[numVecs - 1 - i]);
        }
        
        bitonic_clean_n_vec<vtype, numVecs>(regs);
    }
}

template <typename vtype, int64_t numVecs, int64_t scale, bool first = true>
X86_SIMD_SORT_FINLINE void internal_merge_n_vec(typename vtype::reg_t * reg){
    using reg_t = typename vtype::reg_t;
    using swizzle = typename vtype::swizzle_ops;
    if constexpr (scale <= 1){
        return;
    }else{
        if constexpr (first){
            // Use reverse then merge
            X86_SIMD_SORT_UNROLL_LOOP(64)
            for (int i = 0; i < numVecs; i++){
                reg_t &v = reg[i];
                reg_t rev = swizzle::template reverse_n<vtype, scale>(v);
                COEX<vtype>(rev, v);
                v = swizzle::template merge_n<vtype, scale>(v, rev);
            }
        }else{
            // Use swap then merge
            X86_SIMD_SORT_UNROLL_LOOP(64)
            for (int i = 0; i < numVecs; i++){
                reg_t &v = reg[i];
                reg_t swap = swizzle::template swap_n<vtype, scale>(v);
                COEX<vtype>(swap, v);
                v = swizzle::template merge_n<vtype, scale>(v, swap);
            }
        }
        internal_merge_n_vec<vtype, numVecs, scale / 2, false>(reg);
    }
}

template <typename vtype,
          int64_t numVecs,
          int64_t scale,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_FINLINE void merge_substep_n_vec(reg_t *regs)
{
    using swizzle = typename vtype::swizzle_ops;
    if constexpr (numVecs <= 1) return;
    
    // Reverse upper half of vectors
    X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = numVecs / 2; i < numVecs; i++){
        regs[i] = swizzle::template reverse_n<vtype, scale>(regs[i]);
    }
    // Do compare exchanges
    X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = 0; i < numVecs / 2; i++){
        COEX<vtype>(regs[i], regs[numVecs - 1 - i]);
    }
    
    merge_substep_n_vec<vtype, numVecs / 2, scale>(regs);
    merge_substep_n_vec<vtype, numVecs / 2, scale>(regs + numVecs / 2);
    
}

template <typename vtype,
          int64_t numVecs,
          int64_t scale,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_FINLINE void merge_step_n_vec(reg_t *regs)
{
    // Do cross vector merges
    merge_substep_n_vec<vtype, numVecs, scale>(regs);
    
    // Do internal vector merges
    internal_merge_n_vec<vtype, numVecs, scale>(regs);
}

template <typename vtype,
          int64_t numVecs,
          int64_t numPer = 2,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_FINLINE void merge_n_vec(reg_t *regs)
{
    if constexpr (numPer > vtype::numlanes)
        return;
    else {
        merge_step_n_vec<vtype, numVecs, numPer>(regs);
        merge_n_vec<vtype, numVecs, numPer * 2>(regs);
    }
}

template <typename vtype, int numVecs, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void sort_n_vec(typename vtype::type_t *arr, int32_t N)
{
    static_assert(numVecs > 0, "numVecs should be > 0");
    if constexpr (numVecs > 1){
        if (N * 2 <= numVecs * vtype::numlanes) {
            sort_n_vec<vtype, numVecs / 2>(arr, N);
            return;
        }
    }

    reg_t vecs[numVecs];
    
    // Generate masks for loading and storing
    typename vtype::opmask_t ioMasks[numVecs - numVecs / 2];
X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = numVecs / 2, j = 0; i < numVecs; i++, j++) {
        int64_t num_to_read
                = std::min((int64_t)std::max(0, N - i * vtype::numlanes),
                           (int64_t)vtype::numlanes);
        ioMasks[j] = ((0x1ull << num_to_read) - 0x1ull);
    }

// Unmasked part of the load
X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = 0; i < numVecs / 2; i++) {
        vecs[i] = vtype::loadu(arr + i * vtype::numlanes);
    }
// Masked part of the load
X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = numVecs / 2, j = 0; i < numVecs; i++, j++) {
        vecs[i] = vtype::mask_loadu(
                vtype::zmm_max(), ioMasks[j], arr + i * vtype::numlanes);
    }

// Run the initial sorting network
    bitonic_sort_n_vec<vtype, numVecs>(vecs);
    
// Merge vectors together
    merge_n_vec<vtype, numVecs>(vecs);

// Unmasked part of the store
X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = 0; i < numVecs / 2; i++) {
        vtype::storeu(arr + i * vtype::numlanes, vecs[i]);
    }
// Masked part of the store
X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int i = numVecs / 2, j = 0; i < numVecs; i++, j++) {
        vtype::mask_storeu(arr + i * vtype::numlanes, ioMasks[j], vecs[i]);
    }
}

template <typename vtype, int64_t maxN>
X86_SIMD_SORT_INLINE void sort_n(typename vtype::type_t *arr, int N)
{
    constexpr int numVecs = maxN / vtype::numlanes;
    constexpr bool isMultiple = (maxN == (vtype::numlanes * numVecs));
    constexpr bool powerOfTwo = (numVecs != 0 && !(numVecs & (numVecs - 1)));
    static_assert(powerOfTwo == true && isMultiple == true,
                  "maxN must be vtype::numlanes times a power of 2");

    sort_n_vec<vtype, numVecs>(arr, N);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot(type_t *arr,
                                      const int64_t left,
                                      const int64_t right);

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_blocks(type_t * arr, uint64_t left, uint64_t right){
    
    if (right - left <= 1024){
        return get_pivot<vtype>(arr, left, right);
    }
    
    using reg_t = typename vtype::reg_t;
    constexpr int numVecs = 5;
    
    uint64_t width = (right - vtype::numlanes) - left;
    uint64_t delta = width / numVecs;
    
    reg_t vecs[numVecs];
    // Load data
    for (int i = 0; i < numVecs; i++){
        vecs[i] = vtype::loadu(arr + left + delta * i);
    }
    
    // Implement sorting network (from https://bertdobbelaere.github.io/sorting_networks.html)
    COEX<vtype>(vecs[0], vecs[3]);
    COEX<vtype>(vecs[1], vecs[4]);
    
    COEX<vtype>(vecs[0], vecs[2]);
    COEX<vtype>(vecs[1], vecs[3]);
    
    COEX<vtype>(vecs[0], vecs[1]);
    COEX<vtype>(vecs[2], vecs[4]);
    
    COEX<vtype>(vecs[1], vecs[2]);
    COEX<vtype>(vecs[3], vecs[4]);
    
    COEX<vtype>(vecs[2], vecs[3]);
    
    // Calculate median of the middle vector
    reg_t &vec = vecs[numVecs / 2];
    vec = vtype::sort_vec(vec);
    
    type_t data[vtype::numlanes];
    vtype::storeu(data, vec);
    return data[vtype::numlanes / 2];
}

#endif