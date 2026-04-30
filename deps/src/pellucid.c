#include <immintrin.h>
#include <omp.h>
#include <stddef.h>

static void insert_top_k(
    size_t *const indices,
    float *const values,
    size_t *const count,
    const size_t top_k,
    const size_t index,
    const float value
) {
    if (*count < top_k) {
        size_t i = *count;
        while ((i > 0) && (value > values[i - 1])) {
            values[i] = values[i - 1];
            indices[i] = indices[i - 1];
            --i;
        }
        values[i] = value;
        indices[i] = index;
        ++*count;
    } else if (value > values[top_k - 1]) {
        size_t i = top_k - 1;
        while ((i > 0) && (value > values[i - 1])) {
            values[i] = values[i - 1];
            indices[i] = indices[i - 1];
            --i;
        }
        values[i] = value;
        indices[i] = index;
    }
}

void pellucid_lm_head_top_k_bf16(
    size_t *const restrict top_indices,
    float *const restrict top_values,
    const size_t top_k, // must satisfy 0 < top_k <= vocabulary_size
    const __bf16 *const restrict lm_head_weight,
    const __bf16 *const restrict hidden_state,
    const size_t vocabulary_size,
    const size_t hidden_size // must be divisible by 128
) {
    const size_t max_threads = (size_t)omp_get_max_threads();
    size_t all_indices[max_threads * top_k];
    float all_values[max_threads * top_k];
    size_t all_counts[max_threads];
    for (size_t thread_id = 0; thread_id < max_threads; ++thread_id) {
        all_counts[thread_id] = 0;
    }
#pragma omp parallel
    {
        const size_t num_threads = (size_t)omp_get_num_threads();
        const size_t thread_id = (size_t)omp_get_thread_num();
        size_t *const local_indices = all_indices + thread_id * top_k;
        float *const local_values = all_values + thread_id * top_k;
        size_t local_count = 0;
        for (size_t i = (thread_id * vocabulary_size) / num_threads;
             i < ((thread_id + 1) * vocabulary_size) / num_threads; ++i) {
            const __bf16 *const row = lm_head_weight + hidden_size * i;
            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();
            for (size_t j = 0; j < hidden_size; j += 0x80) {
                const __bf16 *const a = row + j;
                const __bf16 *const b = hidden_state + j;
                const __m512bh a0 = (__m512bh)_mm512_loadu_ps(a + 0x00);
                const __m512bh a1 = (__m512bh)_mm512_loadu_ps(a + 0x20);
                const __m512bh a2 = (__m512bh)_mm512_loadu_ps(a + 0x40);
                const __m512bh a3 = (__m512bh)_mm512_loadu_ps(a + 0x60);
                const __m512bh b0 = (__m512bh)_mm512_loadu_ps(b + 0x00);
                const __m512bh b1 = (__m512bh)_mm512_loadu_ps(b + 0x20);
                const __m512bh b2 = (__m512bh)_mm512_loadu_ps(b + 0x40);
                const __m512bh b3 = (__m512bh)_mm512_loadu_ps(b + 0x60);
                c0 = _mm512_dpbf16_ps(c0, a0, b0);
                c1 = _mm512_dpbf16_ps(c1, a1, b1);
                c2 = _mm512_dpbf16_ps(c2, a2, b2);
                c3 = _mm512_dpbf16_ps(c3, a3, b3);
            }
            const float value = _mm512_reduce_add_ps(
                _mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3))
            );
            insert_top_k(
                local_indices, local_values, &local_count, top_k, i + 1, value
            );
        }
        all_counts[thread_id] = local_count;
    }
    size_t global_count = 0;
    for (size_t thread_id = 0; thread_id < max_threads; ++thread_id) {
        const size_t *const local_indices = all_indices + thread_id * top_k;
        const float *const local_values = all_values + thread_id * top_k;
        for (size_t i = 0; i < all_counts[thread_id]; ++i) {
            insert_top_k(
                top_indices, top_values, &global_count, top_k, local_indices[i],
                local_values[i]
            );
        }
    }
}
