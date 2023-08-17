/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_16BIT_COMMON
#define AVX512_16BIT_COMMON

#include "avx512-common-qsort.h"
#include "xss-network-qsort.hpp"

/*
 * Constants used in sorting 32 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ZMM register: 31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
static const uint16_t network[6][32]
        = {{7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13, 12, 11, 10, 9,  8,
            23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24},
           {15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16},
           {4,  5,  6,  7,  0,  1,  2,  3,  12, 13, 14, 15, 8,  9,  10, 11,
            20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27},
           {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
           {8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,
            24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23},
           {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15}};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t sort_zmm_16bit(zmm_t zmm)
{
    // Level 1
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 2
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(0, 1, 2, 3)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 3
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(1), zmm), 0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 4
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(2), zmm), 0xFF00FF00);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(3), zmm), 0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 5
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(4), zmm), 0xFFFF0000);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(5), zmm), 0xFF00FF00);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(3), zmm), 0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t bitonic_merge_zmm_16bit(zmm_t zmm)
{
    // 1) half_cleaner[32]: compare 1-17, 2-18, 3-19 etc ..
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(6), zmm), 0xFFFF0000);
    // 2) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(5), zmm), 0xFF00FF00);
    // 3) half_cleaner[8]
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(3), zmm), 0xF0F0F0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    // 3) half_cleaner[2]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    return zmm;
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_16bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 32
    int64_t size = (right - left) / 32;
    type_t vec_arr[32] = {arr[left],
                          arr[left + size],
                          arr[left + 2 * size],
                          arr[left + 3 * size],
                          arr[left + 4 * size],
                          arr[left + 5 * size],
                          arr[left + 6 * size],
                          arr[left + 7 * size],
                          arr[left + 8 * size],
                          arr[left + 9 * size],
                          arr[left + 10 * size],
                          arr[left + 11 * size],
                          arr[left + 12 * size],
                          arr[left + 13 * size],
                          arr[left + 14 * size],
                          arr[left + 15 * size],
                          arr[left + 16 * size],
                          arr[left + 17 * size],
                          arr[left + 18 * size],
                          arr[left + 19 * size],
                          arr[left + 20 * size],
                          arr[left + 21 * size],
                          arr[left + 22 * size],
                          arr[left + 23 * size],
                          arr[left + 24 * size],
                          arr[left + 25 * size],
                          arr[left + 26 * size],
                          arr[left + 27 * size],
                          arr[left + 28 * size],
                          arr[left + 29 * size],
                          arr[left + 30 * size],
                          arr[left + 31 * size]};
    typename vtype::zmm_t rand_vec = vtype::loadu(vec_arr);
    typename vtype::zmm_t sort = sort_zmm_16bit<vtype>(rand_vec);
    return ((type_t *)&sort)[16];
}

#endif // AVX512_16BIT_COMMON
