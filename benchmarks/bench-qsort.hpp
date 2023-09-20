#define VQSORT_ONLY_STATIC 1
#include "hwy/contrib/sort/vqsort-inl.h"
#include "bench-qsort-common.h"

template <typename T, class... Args>
static void stdsort(benchmark::State &state, Args &&...args)
{
    auto args_tuple = std::make_tuple(std::move(args)...);
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") { arr = get_uniform_rand_array<T>(ARRSIZE); }
    else if (arrtype == "sorted") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
    }
    else if (arrtype == "constant") {
        T temp = get_uniform_rand_array<T>(1)[0];
        for (size_t ii = 0; ii < ARRSIZE; ++ii) {
            arr.push_back(temp);
        }
    }
    else if (arrtype == "reverse") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
        std::reverse(arr.begin(), arr.end());
    }
    arr_bkp = arr;

    /* call avx512 quicksort */
    for (auto _ : state) {
        std::sort(arr.begin(), arr.end());
        state.PauseTiming();
        arr = arr_bkp;
        state.ResumeTiming();
    }
}

template <typename T, class... Args>
static void avx512qsort(benchmark::State &state, Args &&...args)
{
    auto args_tuple = std::make_tuple(std::move(args)...);
    if (!__builtin_cpu_supports("avx512bw")) {
        state.SkipWithMessage("Requires AVX512 BW ISA");
    }
    if ((sizeof(T) == 2) && (!__builtin_cpu_supports("avx512vbmi2"))) {
        state.SkipWithMessage("Requires AVX512 VBMI2");
    }
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") { arr = get_uniform_rand_array<T>(ARRSIZE); }
    else if (arrtype == "sorted") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
    }
    else if (arrtype == "constant") {
        T temp = get_uniform_rand_array<T>(1)[0];
        for (size_t ii = 0; ii < ARRSIZE; ++ii) {
            arr.push_back(temp);
        }
    }
    else if (arrtype == "reverse") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
        std::reverse(arr.begin(), arr.end());
    }
    arr_bkp = arr;

    /* call avx512 quicksort */
    for (auto _ : state) {
        avx512_qsort<T>(arr.data(), ARRSIZE);
        state.PauseTiming();
        arr = arr_bkp;
        state.ResumeTiming();
    }
}

template <typename T, class... Args>
static void vqsort(benchmark::State &state, Args &&...args)
{
    auto args_tuple = std::make_tuple(std::move(args)...);
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") { arr = get_uniform_rand_array<T>(ARRSIZE); }
    else if (arrtype == "sorted") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
    }
    else if (arrtype == "constant") {
        T temp = get_uniform_rand_array<T>(1)[0];
        for (size_t ii = 0; ii < ARRSIZE; ++ii) {
            arr.push_back(temp);
        }
    }
    else if (arrtype == "reverse") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
        std::reverse(arr.begin(), arr.end());
    }
    arr_bkp = arr;

    /* call avx512 quicksort */
    for (auto _ : state) {
        hwy::HWY_NAMESPACE::VQSortStatic(arr.data(), ARRSIZE, hwy::SortAscending());
        state.PauseTiming();
        arr = arr_bkp;
        state.ResumeTiming();
    }
}

#define BENCH_BOTH_QSORT(type) \
    BENCH(avx512qsort, type) \
    BENCH(stdsort, type) \
    BENCH(vqsort, type)

BENCH_BOTH_QSORT(uint64_t)
BENCH_BOTH_QSORT(int64_t)
BENCH_BOTH_QSORT(uint32_t)
BENCH_BOTH_QSORT(int32_t)
//BENCH_BOTH_QSORT(uint16_t)
//BENCH_BOTH_QSORT(int16_t)
BENCH_BOTH_QSORT(float)
BENCH_BOTH_QSORT(double)
