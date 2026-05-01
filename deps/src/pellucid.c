#include <immintrin.h>
#include <omp.h>
#include <stddef.h>

void pellucid_matvec_bf16(
    __bf16 *const restrict c,
    const __bf16 *const restrict a,
    const __bf16 *const restrict b,
    const size_t m,
    const size_t n // must be divisible by 128
) {
#pragma omp parallel
    {
        const size_t num_threads = (size_t)omp_get_num_threads();
        const size_t thread_id = (size_t)omp_get_thread_num();
        for (size_t i = (thread_id * m) / num_threads;
             i < ((thread_id + 1) * m) / num_threads; ++i) {
            const __bf16 *const u = a + n * i;
            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();
            for (size_t j = 0; j < n; j += 0x80) {
                const __m512bh a0 = (__m512bh)_mm512_loadu_ps(u + j + 0x00);
                const __m512bh a1 = (__m512bh)_mm512_loadu_ps(u + j + 0x20);
                const __m512bh a2 = (__m512bh)_mm512_loadu_ps(u + j + 0x40);
                const __m512bh a3 = (__m512bh)_mm512_loadu_ps(u + j + 0x60);
                const __m512bh b0 = (__m512bh)_mm512_loadu_ps(b + j + 0x00);
                const __m512bh b1 = (__m512bh)_mm512_loadu_ps(b + j + 0x20);
                const __m512bh b2 = (__m512bh)_mm512_loadu_ps(b + j + 0x40);
                const __m512bh b3 = (__m512bh)_mm512_loadu_ps(b + j + 0x60);
                c0 = _mm512_dpbf16_ps(c0, a0, b0);
                c1 = _mm512_dpbf16_ps(c1, a1, b1);
                c2 = _mm512_dpbf16_ps(c2, a2, b2);
                c3 = _mm512_dpbf16_ps(c3, a3, b3);
            }
            c[i] = _mm_cvtness_sbh(_mm512_reduce_add_ps(
                _mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3))
            ));
        }
    }
}

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

void pellucid_matmul_bf16(
    __bf16 *restrict c,
    const __bf16 *restrict a,
    const __bf16 *restrict b,
    const size_t m, // must be divisible by 8
    const size_t n, // must be divisible by 2
    const size_t k  // must be divisible by 32
) {
#pragma omp parallel
    {
        const size_t num_threads = (size_t)omp_get_num_threads();
        const size_t thread_id = (size_t)omp_get_thread_num();
        for (size_t i = (thread_id * (m / 8)) / num_threads;
             i < ((thread_id + 1) * (m / 8)) / num_threads; ++i) {
            const __bf16 *const u0 = a + k * (8 * i + 0);
            const __bf16 *const u1 = a + k * (8 * i + 1);
            const __bf16 *const u2 = a + k * (8 * i + 2);
            const __bf16 *const u3 = a + k * (8 * i + 3);
            const __bf16 *const u4 = a + k * (8 * i + 4);
            const __bf16 *const u5 = a + k * (8 * i + 5);
            const __bf16 *const u6 = a + k * (8 * i + 6);
            const __bf16 *const u7 = a + k * (8 * i + 7);
            for (size_t j = 0; j < n; j += 2) {
                const __bf16 *const v0 = b + k * (j + 0);
                const __bf16 *const v1 = b + k * (j + 1);
                __bf16 *const w0 = c + m * (j + 0) + 8 * i;
                __bf16 *const w1 = c + m * (j + 1) + 8 * i;
                __m512 c00 = _mm512_setzero_ps();
                __m512 c01 = _mm512_setzero_ps();
                __m512 c10 = _mm512_setzero_ps();
                __m512 c11 = _mm512_setzero_ps();
                __m512 c20 = _mm512_setzero_ps();
                __m512 c21 = _mm512_setzero_ps();
                __m512 c30 = _mm512_setzero_ps();
                __m512 c31 = _mm512_setzero_ps();
                __m512 c40 = _mm512_setzero_ps();
                __m512 c41 = _mm512_setzero_ps();
                __m512 c50 = _mm512_setzero_ps();
                __m512 c51 = _mm512_setzero_ps();
                __m512 c60 = _mm512_setzero_ps();
                __m512 c61 = _mm512_setzero_ps();
                __m512 c70 = _mm512_setzero_ps();
                __m512 c71 = _mm512_setzero_ps();
                for (size_t p = 0; p < k; p += 0x20) {
                    const __m512bh a0 = (__m512bh)_mm512_loadu_ps(u0 + p);
                    const __m512bh a1 = (__m512bh)_mm512_loadu_ps(u1 + p);
                    const __m512bh a2 = (__m512bh)_mm512_loadu_ps(u2 + p);
                    const __m512bh a3 = (__m512bh)_mm512_loadu_ps(u3 + p);
                    const __m512bh a4 = (__m512bh)_mm512_loadu_ps(u4 + p);
                    const __m512bh a5 = (__m512bh)_mm512_loadu_ps(u5 + p);
                    const __m512bh a6 = (__m512bh)_mm512_loadu_ps(u6 + p);
                    const __m512bh a7 = (__m512bh)_mm512_loadu_ps(u7 + p);
                    const __m512bh b0 = (__m512bh)_mm512_loadu_ps(v0 + p);
                    const __m512bh b1 = (__m512bh)_mm512_loadu_ps(v1 + p);
                    c00 = _mm512_dpbf16_ps(c00, a0, b0);
                    c01 = _mm512_dpbf16_ps(c01, a0, b1);
                    c10 = _mm512_dpbf16_ps(c10, a1, b0);
                    c11 = _mm512_dpbf16_ps(c11, a1, b1);
                    c20 = _mm512_dpbf16_ps(c20, a2, b0);
                    c21 = _mm512_dpbf16_ps(c21, a2, b1);
                    c30 = _mm512_dpbf16_ps(c30, a3, b0);
                    c31 = _mm512_dpbf16_ps(c31, a3, b1);
                    c40 = _mm512_dpbf16_ps(c40, a4, b0);
                    c41 = _mm512_dpbf16_ps(c41, a4, b1);
                    c50 = _mm512_dpbf16_ps(c50, a5, b0);
                    c51 = _mm512_dpbf16_ps(c51, a5, b1);
                    c60 = _mm512_dpbf16_ps(c60, a6, b0);
                    c61 = _mm512_dpbf16_ps(c61, a6, b1);
                    c70 = _mm512_dpbf16_ps(c70, a7, b0);
                    c71 = _mm512_dpbf16_ps(c71, a7, b1);
                }
                w0[0] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c00));
                w1[0] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c01));
                w0[1] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c10));
                w1[1] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c11));
                w0[2] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c20));
                w1[2] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c21));
                w0[3] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c30));
                w1[3] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c31));
                w0[4] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c40));
                w1[4] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c41));
                w0[5] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c50));
                w1[5] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c51));
                w0[6] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c60));
                w1[6] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c61));
                w0[7] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c70));
                w1[7] = _mm_cvtness_sbh(_mm512_reduce_add_ps(c71));
            }
        }
    }
}
