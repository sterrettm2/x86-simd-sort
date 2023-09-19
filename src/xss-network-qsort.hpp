#ifndef XSS_NETWORK_QSORT
#define XSS_NETWORK_QSORT

#include "avx512-common-qsort.h"
//#include "xss-network-luts.hpp"

// TODO remove these

#include <iostream>
template <typename vtype, typename reg_t = typename vtype::reg_t>
void printVecs(reg_t * vecs, int numVecs){
    using type_t = typename vtype::type_t;
    for (int i = 0; i < numVecs; i++){
        type_t data[vtype::numlanes];
        vtype::storeu(data, vecs[i]);
        for (int j = 0; j < vtype::numlanes; j++){
            std::cout << data[j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n\n";
}


/*constexpr int ctimeBitIndex(const int64_t x){
    int count = 0;
    while (x != 0){
        x /= 2;
        count++;
    }
    return count;
}*/

template <typename vtype, int64_t scale>
X86_SIMD_SORT_INLINE typename vtype::reg_t swap_n(typename vtype::reg_t reg);

template <typename vtype, int64_t scale>
X86_SIMD_SORT_INLINE typename vtype::reg_t reverse_n(typename vtype::reg_t reg);

template <typename vtype, int64_t numVecs, int64_t scale, bool first = true>
X86_SIMD_SORT_INLINE void internal_sort_n_vec(typename vtype::reg_t * reg);

template <typename vtype,
          int64_t numVecs,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void bitonic_clean_n_vec(reg_t *regs)
{
X86_SIMD_SORT_UNROLL_LOOP(64)
    for (int num = numVecs / 2; num >= 2; num /= 2) {
X86_SIMD_SORT_UNROLL_LOOP(64)
        for (int j = 0; j < numVecs; j += num) {
X86_SIMD_SORT_UNROLL_LOOP(64)
            for (int i = 0; i < num / 2; i++) {
                COEX<vtype>(regs[i + j], regs[i + j + num / 2]);
            }
        }
    }
}

template <typename vtype,
          int64_t numVecs,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void bitonic_sort_n_vec(reg_t *regs)
{
    if constexpr (numVecs == 1){
        return;
    }else{
        bitonic_sort_n_vec<vtype, numVecs / 2>(regs);
        bitonic_sort_n_vec<vtype, numVecs / 2>(regs + numVecs / 2);
        
        for (int i = 0; i < numVecs / 2; i++){
            COEX<vtype>(regs[i], regs[numVecs - 1 - i]);
        }
        
        bitonic_clean_n_vec<vtype, numVecs>(regs);
    }
}

template <typename vtype,
          int64_t numVecs,
          int64_t scale,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void merge_substep_n_vec(reg_t *regs)
{
    if constexpr (numVecs <= 1) return;
    
    // Reverse upper half of vectors
    for (int i = numVecs / 2; i < numVecs; i++){
        regs[i] = reverse_n<vtype, scale>(regs[i]);
    }
    // Do compare exchanges
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
X86_SIMD_SORT_INLINE void merge_step_n_vec(reg_t *regs)
{
    merge_substep_n_vec<vtype, numVecs, scale>(regs);
    
    // Do some work on each individual register
    internal_sort_n_vec<vtype, numVecs, scale>(regs);
    
    //printVecs<vtype>(regs, numVecs);
    
}

template <typename vtype,
          int64_t numVecs,
          int64_t numPer = 2,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void merge_n_vec(reg_t *regs)
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
    if (numVecs > 1 && N * 2 <= numVecs * vtype::numlanes) {
        sort_n_vec<vtype, numVecs / 2>(arr, N);
        return;
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
//std::cout << "AT BITONIC SORT\n";
//printVecs<vtype>(vecs, numVecs);
    bitonic_sort_n_vec<vtype, numVecs>(vecs);
    //printVecs<vtype>(vecs, numVecs);
    
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

template <typename vtype, int64_t scale>
X86_SIMD_SORT_INLINE typename vtype::reg_t reverse_n(typename vtype::reg_t reg){
    using reg_t = typename vtype::reg_t;
    using type_t = typename vtype::type_t;
    
    __m512i v = vtype::cast_to(reg);
    
    //constexpr int i = ctimeBitIndex(sizeof(type_t));
    
    if constexpr (scale == vtype::numlanes){
        return vtype::reverse(reg);
    }else if constexpr (scale == 2){
        return swap_n<vtype, 2>(reg);
    }else{
        if constexpr (sizeof(type_t) == 2){
            if constexpr (scale == 4){
            }else if constexpr (scale == 8){
            }else if constexpr (scale == 16){
                
            }else{
                static_assert(scale == 4, "should not be reached");
            }
        }else if constexpr (sizeof(type_t) == 4){
            if constexpr (scale == 4){
                __m512i mask = _mm512_set_epi32(12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3);
                v = _mm512_permutexvar_epi32(mask, v);
            }else if constexpr (scale == 8){
                __m512i mask = _mm512_set_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);
                v = _mm512_permutexvar_epi32(mask, v);
            }else{
                static_assert(scale == 4, "should not be reached");
            }
        }else if constexpr (sizeof(type_t) == 8){
            // scale == 4, all other cases already handled
            static_assert(scale == 4, "scale != 4, should not be possible");
            constexpr uint64_t mask = 0b00011011;
            v = _mm512_permutex_epi64(v, mask);
        }
    } 
    
    return vtype::cast_from(v);
}

template <typename vtype, int64_t scale>
X86_SIMD_SORT_INLINE typename vtype::reg_t swap_n(typename vtype::reg_t reg){
    using reg_t = typename vtype::reg_t;
    using type_t = typename vtype::type_t;
    
    constexpr int bytesPer = scale * sizeof(type_t);
    
    __m512i v = vtype::cast_to(reg);
    if constexpr (bytesPer == 4){
        // TODO figure this one out
    }else if constexpr (bytesPer == 8){
        v = _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0b10110001);
    }else if constexpr (bytesPer == 16){
        v = _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0b01001110);
    }else if constexpr (bytesPer == 32){
        v = _mm512_shuffle_i64x2(v, v, 0b10110001);
    }else if constexpr (bytesPer == 64){
        v = _mm512_shuffle_i64x2(v, v, 0b01001110);
    }else{
        static_assert(bytesPer == 4, "should not be reached");
    }
    
    return vtype::cast_from(v);
}

template <typename vtype, int64_t scale>
X86_SIMD_SORT_INLINE typename vtype::reg_t merge_n(typename vtype::reg_t reg, typename vtype::reg_t other){
    using reg_t = typename vtype::reg_t;
    using type_t = typename vtype::type_t;
    
    __m512i v1 = vtype::cast_to(reg);
    __m512i v2 = vtype::cast_to(other);
    
    if constexpr (sizeof(type_t) == 8){
        if constexpr (scale == 2){
            v1 = _mm512_mask_blend_epi64(0b01010101, v1, v2);
        }else if constexpr (scale == 4){
            v1 = _mm512_mask_blend_epi64(0b00110011, v1, v2);
        }else if constexpr (scale == 8){
            v1 = _mm512_mask_blend_epi64(0b00001111, v1, v2);
        }else{
            static_assert(scale == 4, "should not be reached");
        }
    }else if constexpr (sizeof(type_t) == 4){
        if constexpr (scale == 2){
            v1 = _mm512_mask_blend_epi32(0b0101010101010101, v1, v2);
        }else if constexpr (scale == 4){
            v1 = _mm512_mask_blend_epi32(0b0011001100110011, v1, v2);
        }else if constexpr (scale == 8){
            v1 = _mm512_mask_blend_epi32(0b0000111100001111, v1, v2);
        }else if constexpr (scale == 16){
            v1 = _mm512_mask_blend_epi32(0b0000000011111111, v1, v2);
        }else{
            static_assert(scale == 4, "should not be reached");
        }
    }
    
    
    return vtype::cast_from(v1);
}

template <typename vtype, int64_t numVecs, int64_t scale, bool first>
X86_SIMD_SORT_INLINE void internal_sort_n_vec(typename vtype::reg_t * reg){
    using reg_t = typename vtype::reg_t;
    if constexpr (scale <= 1){
        return;
    }else{
        if constexpr (first){
            // Use reverse then merge
            for (int i = 0; i < numVecs; i++){
                reg_t &v = reg[i];
                reg_t rev = reverse_n<vtype, scale>(v);
                COEX<vtype>(rev, v);
                v = merge_n<vtype, scale>(v, rev);
            }
        }else{
            // Use swap then merge
            for (int i = 0; i < numVecs; i++){
                reg_t &v = reg[i];
                reg_t swap = swap_n<vtype, scale>(v);
                COEX<vtype>(swap, v);
                v = merge_n<vtype, scale>(v, swap);
            }
        }
        internal_sort_n_vec<vtype, numVecs, scale / 2, false>(reg);
    }
}

#endif