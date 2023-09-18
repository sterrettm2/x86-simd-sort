#ifndef XSS_NETWORK_QSORT
#define XSS_NETWORK_QSORT

#include "avx512-common-qsort.h"
//#include "xss-network-luts.hpp"


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
X86_SIMD_SORT_INLINE void merge_step_n_vec(reg_t *regs)
{
    // Reverse upper half of vectors
    for (int i = numVecs / 2; i < numVecs; i++){
        regs[i] = reverse_n<vtype, scale>(regs[i]);
    }
    // Do compare exchanges
    for (int i = 0; i < numVecs / 2; i++){
        COEX<vtype>(regs[i], regs[numVecs - 1 - i]);
    }
    
    
    // Now cleanup individual vectors
    //... do things!
    //maybe use the name clean for this part?
    
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
                
            }else if constexpr (scale == 8){
                
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
#endif