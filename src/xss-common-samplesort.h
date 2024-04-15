#ifndef XSS_COMMON_SAMPLESORT
#define XSS_COMMON_SAMPLESORT

#include "xss-common-qsort.h"

#include <omp.h>
#include <iostream>

struct samplesort_partition{
    int index = 0;
    int length = 0;
    
    samplesort_partition(){}
    samplesort_partition(int _index, int _length) : index(_index), length(_length) {}
    
    static bool index_comparator(samplesort_partition const& lhs, samplesort_partition const& rhs){
        return lhs.index < rhs.index;
    }
    
    static bool length_comparator(samplesort_partition const& lhs, samplesort_partition const& rhs){
        return lhs.length < rhs.length;
    }
};

template <typename vtype, typename comparator, typename type_t>
X86_SIMD_SORT_INLINE void samplesort_(type_t *arr,
                                   arrsize_t left,
                                   arrsize_t right,
                                   int threadCount = -1)
{   
    type_t * base = arr + left;
    arrsize_t arrsize = right + 1 - left;
    
    if (threadCount == -1){
        threadCount = 4; // TODO somehow get a real value for this instead of an arbitrary one
    }
    
    int indexCount = threadCount;
    
    std::vector<samplesort_partition> partitions;
    partitions.reserve(indexCount);
    partitions.emplace_back(0, arrsize);
    
    // First repeatedly partition the largest remaining partition, until we have made enough partitions to move ahead
    for (int i = 0; i < indexCount - 1; i++){
        
        int largestIndex = std::distance(partitions.begin(), std::max_element(partitions.begin(), partitions.end(), &samplesort_partition::length_comparator));

        arrsize_t start = partitions[largestIndex].index;
        arrsize_t length = partitions[largestIndex].length;
        
        arrsize_t left = start;
        arrsize_t right = start + length - 1;
        
        auto pivot_result = get_pivot_smart<vtype, comparator, type_t>(base, left, right);
        type_t pivot = pivot_result.pivot;

        if (pivot_result.result == pivot_result_t::Sorted) {
            // TODO: What to do in these kind of cases?
        }

        type_t smallest = vtype::type_max();
        type_t biggest = vtype::type_min();

        arrsize_t pivot_index = partition_unrolled<vtype, comparator, vtype::partition_unroll_factor>
            (base, left, right + 1, pivot, &smallest, &biggest);
        
        if (pivot_result.result == pivot_result_t::Only2Values) {
            // TODO: What to do in these kind of cases?
        }
        
        arrsize_t leftSize = pivot_index - left;
        arrsize_t rightSize = right - pivot_index + 1;
        
        // Update the legnths and indices
        partitions[largestIndex].length = leftSize;
        
        partitions.emplace_back(pivot_index, rightSize);
    }
    
    // Call subsorting algorithm
    omp_set_num_threads(threadCount);
    
    #pragma omp parallel for
    for (int i = 0; i < threadCount; i++){
        if (partitions[i].length <= 1) continue;
        
        arrsize_t left = partitions[i].index;
        arrsize_t right = partitions[i].index + partitions[i].length - 1;
        
        qsort_<vtype, comparator, type_t>(base, left, right, 50);
    }
}

// Samplesort routines:
template <typename vtype, typename T, bool descending = false>
X86_SIMD_SORT_INLINE void xss_samplesort(T *arr, arrsize_t arrsize, bool hasnan, int threadCount)
{
    using comparator =
            typename std::conditional<descending,
                                      Comparator<vtype, true>,
                                      Comparator<vtype, false>>::type;

    if (arrsize > 1) {
        arrsize_t nan_count = 0;
        if constexpr (std::is_floating_point_v<T>) {
            if (UNLIKELY(hasnan)) {
                nan_count = replace_nan_with_inf<vtype>(arr, arrsize);
            }
        }

        UNUSED(hasnan);
        samplesort_<vtype, comparator, T>(arr, 0, arrsize - 1, threadCount);

        replace_inf_with_nan(arr, arrsize, nan_count, descending);
    }
}

template <typename T>
X86_SIMD_SORT_INLINE void avx512_samplesort(T *arr, arrsize_t arrsize, bool hasnan, bool descending, int threadCount)
{
    using vtype = zmm_vector<T>;
    if (descending)
        xss_samplesort<vtype, T, true>(arr, arrsize, hasnan, threadCount);
    else
        xss_samplesort<vtype, T, false>(arr, arrsize, hasnan, threadCount);
}

#endif // XSS_COMMON_SAMPLESORT