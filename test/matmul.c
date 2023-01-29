const char* dgemm_desc = "Simple blocked dgemm.";
#include <math.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif
/*
Cache sizes:
L1: 32KB
L2: 512KB
L3: 32MB
*/

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 41
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

const int KC = 256, MC = 96, MR = 4, NR = 4;
#define A(i,j) = A[i*lda+j]
#define B(i,j) = A[i*lda+j]
#define C(i,j) = A[i*lda+j]

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_giv(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int k = 0; k < lda; k += BLOCK_SIZE) {
    for (int j = 0; j < lda; j += BLOCK_SIZE) {

    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
            // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

/*-----------------------------------------------------------------*/
void pack_b(int lda, int KC_act, double* B, double* packed_b) {
    for (int pan_ptr = 0; pan_ptr < lda; pan_ptr+=NR) {
        int nr_act = min(NR, lda-pan_ptr);
        for (int i = 0; i < KC_act; i++) {
            for (int j = 0; j < nr_act; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
            for (int j = nr_act; j < NR; j++) {
                *packed_b++ = 0.0;
            }
        }
    }
}

void pack_a(int lda, int mc_act, int kc_act, double* A, double* packed_a) {
    for (int pan_ptr = 0; pan_ptr < mc_act; pan_ptr+=MR) {
        int mr_act = min(MR, mc_act - pan_ptr);
        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < mr_act; j++) {
                *packed_a++ = A[pan_ptr + lda*i + j];
            }
            for (int j = mr_act; j < MR; j++) {
                *packed_a++ = 0.0;
            }
        }
    }
}

void pad_c(int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* C, double* micro_tile_c) {
    int idx = 0;
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            micro_tile_c[idx++] = C[i+j*lda];
        }
        for (int i = mr_act; i < MR; i++) {
            micro_tile_c[idx++] = 0.0;
        }
    }
    for (int j = nr_act; j < NR; j++) {
        for (int i = 0; i < MR; i++) {
            micro_tile_c[idx++] = 0.0;
        }
    }
}

void micro_kernel(int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_col0 = _mm256_loadu_pd(micro_tile_c);
    __m256d c_col1 = _mm256_loadu_pd(micro_tile_c+4);
    __m256d c_col2 = _mm256_loadu_pd(micro_tile_c+8);
    __m256d c_col3 = _mm256_loadu_pd(micro_tile_c+12);

    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    for (int i = 0; i < kc_act; i+=1) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        __m256d a_col0 = _mm256_loadu_pd(pan_a);

        __m256d b_0 = _mm256_broadcast_sd(pan_b);
        __m256d b_1 = _mm256_broadcast_sd(pan_b+1);
        __m256d b_2 = _mm256_broadcast_sd(pan_b+2);
        __m256d b_3 = _mm256_broadcast_sd(pan_b+3);

        c_col0 = _mm256_fmadd_pd(a_col0, b_0, c_col0);
        c_col1 = _mm256_fmadd_pd(a_col0, b_1, c_col1);
        c_col2 = _mm256_fmadd_pd(a_col0, b_2, c_col2);
        c_col3 = _mm256_fmadd_pd(a_col0, b_3, c_col3);

        pan_b += NR;
        pan_a += MR;
    }

    //Store first into microtile
    _mm256_storeu_pd(micro_tile_c, c_col0);
    _mm256_storeu_pd(micro_tile_c+4, c_col1);
    _mm256_storeu_pd(micro_tile_c+8, c_col2);
    _mm256_storeu_pd(micro_tile_c+12, c_col3);

    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] = micro_tile_c[i+j*NR];
        }
    }
}

void block_panel_multiply(int lda, int kc_act, int mc_act, int nr_act, double* packed_a, double* packed_b, double* C, double* micro_tile_c) {
    for (int a_pan_ptr = 0; a_pan_ptr < MC; a_pan_ptr += MR) {
        int mr_act = min(MR, MC-a_pan_ptr);
        pad_c(lda, kc_act, mc_act, nr_act, mr_act, C + a_pan_ptr, micro_tile_c);
        micro_kernel(lda, kc_act, mc_act, nr_act, mr_act, packed_a + a_pan_ptr*kc_act, packed_b, C + a_pan_ptr, micro_tile_c);
    }
}

void block_multi_panel_multiply(int lda, int kc_act, int mc_act, double* packed_a, double* packed_b, double* C, double* micro_tile_c) {
    for (int b_pan_ptr = 0; b_pan_ptr < lda; b_pan_ptr+=NR) {
        int nr_act = min(NR, lda-b_pan_ptr);
        block_panel_multiply(lda, kc_act, mc_act, nr_act, packed_a, packed_b + b_pan_ptr*KC, C + b_pan_ptr*lda, micro_tile_c);
    }
}

void outer_proudct(int lda, int kc_act, double* A, double* packed_b, double* C, double* packed_a, double* micro_tile_c) {
    for (int block_ptr = 0; block_ptr < lda; block_ptr+=MC) {
        // pack A
        int mc_act = min(MC, lda - block_ptr);
        pack_a(lda, mc_act, kc_act, A + block_ptr, packed_a);
        block_multi_panel_multiply(lda, kc_act, mc_act, packed_a, packed_b, C + block_ptr,micro_tile_c);
    }
}

void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    
    double* packed_b = (double*) _mm_malloc(lda*KC *sizeof(double), 64);
    double* packed_a = (double*) _mm_malloc(MC*KC *sizeof(double), 64);
    double* micro_tile_c = (double*) _mm_malloc(MR*NR * sizeof(double), 64);
    for (int j = 0; j < lda; j += KC) {
        int KC_act = min(KC, lda-j);
        pack_b(lda, KC_act, B + j, packed_b);
        outer_proudct(lda, KC_act, A + lda * j, packed_b, C, packed_a, micro_tile_c);
    }
    _mm_free(packed_b);
    _mm_free(packed_a);
    _mm_free(micro_tile_c);
}

/*-----------------------------------------------------------------*/
void transpose(int r, int c, double *dst, double *src) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            dst[j + i*c] = src[i+j*r];                      
        }
    } 
}

void matmul_no_block(const lda, double *mat1, double *mat2, double* result) {
    // Task 1.6 TODO
    int blocksize = 32;

    int size = lda*lda;
  
    double *mat1_t = malloc(size * sizeof(double));
    transpose(lda, lda, mat1_t, mat1);

    // double *mat2_t = malloc(size * sizeof(double));
    // transpose(lda, lda, mat2_t, mat2);

    int r = lda;
    int c = lda;
    // double *temp = malloc(size * sizeof(double));
    // memcpy(temp, result, size*sizeof(double));

    int unroll;
    if (size <= 1) {
        __m256d sum;
        double sums[4] = {0, 0, 0, 0};
        double sum_tail;
        // int muli, mulj;
        for (int i = 0; i < lda; i++) {
            for (int j = 0; j < lda; j++) {
                sum = _mm256_set1_pd(0);
                sum_tail = 0;
                int k;
                for (k = 0; k < lda/4*4; k+=4) {
                    __m256d m1 = _mm256_loadu_pd(mat1 + lda * i + k);
                    __m256d m2 = _mm256_loadu_pd(mat2 + lda * j + k);
                    sum = _mm256_fmadd_pd (m1, m2, sum); 
                }
                for (; k < lda;  k++) {
                    sum_tail += mat1[lda * i + k] * mat2[lda * j + k];
                }
                _mm256_storeu_pd (sums, sum);
                result[j*lda + i] += sum_tail + sums[0] + sums[1] + sums[2] + sums[3];
            }
        }
    } else {
        // unroll = 16;
        // __m256d sum1, sum2, sum3, sum4;
        // __m256d m1_1, m1_2, m1_3, m1_4;
        // __m256d m2_1;
        // /// jki
        // for (int j = 0; j < lda; j++) {
        //     for (int k = 0; k < lda; k++) {
        //         int i = 0;
        //         int j_lda = j*lda;
        //         int k_lda = k*lda;

        //         double bval = *(mat2 + lda*j + k);
        //         for (; i < lda/unroll*unroll; i+=unroll) {
        //             int j_lda_i = j_lda + i;
        //             int k_lda_i = k_lda + i;

        //             sum1 = _mm256_loadu_pd(result + j_lda_i);
        //             sum2 = _mm256_loadu_pd(result + j_lda_i + 4);
        //             sum3 = _mm256_loadu_pd(result + j_lda_i + 8);
        //             sum4 = _mm256_loadu_pd(result + j_lda_i + 12);

        //             m1_1 = _mm256_loadu_pd(mat1 + k_lda_i);
        //             m1_2 = _mm256_loadu_pd(mat1 + k_lda_i +4);
        //             m1_3 = _mm256_loadu_pd(mat1 + k_lda_i +8);
        //             m1_4 = _mm256_loadu_pd(mat1 + k_lda_i +12);

        //             m2_1 = _mm256_broadcast_sd(&bval);

        //             sum1 = _mm256_fmadd_pd(m1_1, m2_1, sum1); 
        //             sum2 = _mm256_fmadd_pd(m1_2, m2_1, sum2); 
        //             sum3 = _mm256_fmadd_pd(m1_3, m2_1, sum3); 
        //             sum4 = _mm256_fmadd_pd(m1_4, m2_1, sum4);

        //             _mm256_storeu_pd (result + j_lda_i, sum1);
        //             _mm256_storeu_pd (result + j_lda_i + 4, sum2);
        //             _mm256_storeu_pd (result + j_lda_i + 8, sum3);
        //             _mm256_storeu_pd (result + j_lda_i + 12, sum4);
        //         }                   
        //         for (; i < lda; i++) {
        //             result[j_lda + i] += mat1[k_lda + i] * bval;
        //         }
        //     }
        // }
        unroll = 20;
        double sums[unroll];
        double sum_tail;
        int mulj; int muli;
        __m256d sum1, sum2, sum3, sum4, sum5;
        __m256d m1_1, m1_2, m1_3, m1_4, m1_5;
        __m256d m2_1, m2_2, m2_3, m2_4, m2_5;
        __m256d sum6, m1_6, m2_6;
        for (int i = 0; i < lda; i++) {
            muli = lda * i;
            for (int j = 0; j < lda; j++) {
                sum1 = _mm256_set1_pd(0);
                sum2 = _mm256_set1_pd(0);
                sum3 = _mm256_set1_pd(0);
                sum4 = _mm256_set1_pd(0);
                sum5 = _mm256_set1_pd(0);
                sum6 = _mm256_set1_pd(0);

                sum_tail = 0;
                mulj = lda * j;
                int k = 0;
                for (; k < lda/unroll*unroll; k+=unroll) {
                    int mulik = muli+k;
                    int muljk = mulj +k;
                    m1_1 = _mm256_loadu_pd(mat1_t + mulik);
                    m1_2 = _mm256_loadu_pd(mat1_t + mulik +4);
                    m1_3 = _mm256_loadu_pd(mat1_t + mulik +8);
                    m1_4 = _mm256_loadu_pd(mat1_t + mulik +12);
                    m1_5 = _mm256_loadu_pd(mat1_t + mulik +16);
                    m1_6 = _mm256_loadu_pd(mat1_t + mulik +20);

                    m2_1 = _mm256_loadu_pd(mat2 + muljk );
                    m2_2 = _mm256_loadu_pd(mat2 + muljk +4);
                    m2_3 = _mm256_loadu_pd(mat2 + muljk +8);
                    m2_4 = _mm256_loadu_pd(mat2 + muljk +12);
                    m2_5 = _mm256_loadu_pd(mat2 + muljk +16);
                    // m2_6 = _mm256_loadu_pd(mat2 + muljk +20);

                    sum1 = _mm256_fmadd_pd(m1_1, m2_1, sum1); 
                    sum2 = _mm256_fmadd_pd(m1_2, m2_2, sum2); 
                    sum3 = _mm256_fmadd_pd(m1_3, m2_3, sum3); 
                    sum4 = _mm256_fmadd_pd(m1_4, m2_4, sum4); 
                    sum5 = _mm256_fmadd_pd(m1_5, m2_5, sum5); 
                    // sum6 = _mm256_fmadd_pd(m1_6, m2_6, sum6); 
                }
                // unroll = 4;
                // for (; k < lda/4*4; k+=4) {
                //     m1_1 = _mm256_loadu_pd(mat1_t + muli + k);
                //     m1_2 = _mm256_loadu_pd(mat2 + mulj + k);
                //     sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m1_2)); 
                // }                    
                for (; k < lda; k++) {
                    sum_tail += mat1_t[lda*i + k] * mat2[lda*j + k];
                }
                sum1 = _mm256_add_pd(sum1, sum2);
                sum3 = _mm256_add_pd(sum3, sum4);
                sum5 = _mm256_add_pd(sum5, sum1);
                // sum1 = _mm256_add_pd(sum1, sum3);

                _mm256_storeu_pd(sums, sum5);
                _mm256_storeu_pd(sums+4, sum3);

                // _mm256_storeu_pd (sums, sum1);
                // _mm256_storeu_pd (sums+4, sum2);
                // _mm256_storeu_pd (sums+8, sum3);
                // _mm256_storeu_pd (sums+12, sum4);
                // _mm256_storeu_pd (sums+16, sum5);
                // _mm256_storeu_pd (sums+20, sum5);

                // temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3];
                result[j*lda + i] += sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                // temp[i*c + j] = sums[8] + sums[9] + sums[10] + sums[11];

                // result[j*lda + i] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
            }
        }
    }

    
    
    // free(temp);
    // free(mat1_t);
    // free(mat2_t);
}


/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void naive(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < n; ++i) {
        // For each column j of B
        for (int j = 0; j < n; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * n];
            for (int k = 0; k < n; k++) {
                cij += A[i + k * n] * B[k + j * n];
            }
            C[i + j * n] = cij;
        }
    }
}

void print_matrix_tran(int lda, double *mat) {
    printf("\n");
    for (int i = 0; i < lda*lda; i++) {
        printf("%f ", mat[i]);
        if ((i+1) % lda == 0) {
            printf("\n");
        }
    }
}

int main(int argc, char *argv[]) {
    int lda = 10;
    double* A = (double*) malloc(lda*lda*sizeof(double));
    double* B = (double*) malloc(lda*lda*sizeof(double));
    double* C = (double*) malloc(lda*lda*sizeof(double));
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < lda; j++) {
            A[j+i*lda] = j+i*lda;
            B[j+i*lda] = j+i*lda;
            C[j+i*lda] = j+i*lda;
        }
    }

    naive(lda, A, B, C);
    print_matrix_tran(lda, C);
}

