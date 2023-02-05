const char* dgemm_desc = "Simple blocked dgemm.";
#include <math.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif
#include <stdio.h>
#include <string.h>
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

const int KC = 256, MR = 8, NR = 5;
int MC;
/*-----------------------------------------------------------------*/
void pack_b_default(const int lda, int kc_act, double* B, double* packed_b) {
    for (int pan_ptr = 0; pan_ptr < lda; pan_ptr+=NR) {
        int nr_act = min(NR, lda-pan_ptr);
        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < nr_act; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
            for (int j = nr_act; j < NR; j++) {
                *packed_b++ = 0.0;
            }
        }        
    }
}
void pack_b_unrolled(const int lda, int kc_act, double* B, double* packed_b) {
    int unroll = 4;
    int nr_act;
    for (int pan_ptr = 0; pan_ptr < lda/unroll/NR*unroll*NR; pan_ptr+=unroll*NR) {
        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < NR; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
        }
        pan_ptr += NR;

        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < NR; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
        }
        pan_ptr += NR;

        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < NR; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
        }
        pan_ptr += NR;

        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < NR; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
        }
        pan_ptr -= (unroll-1) * NR;                    
    }
    for (int pan_ptr = lda/unroll/NR*unroll*NR; pan_ptr < lda; pan_ptr+=NR) {
        int nr_act = min(NR, lda-pan_ptr);
        for (int i = 0; i < kc_act; i++) {
            for (int j = 0; j < nr_act; j++) {
                *packed_b++ = B[pan_ptr*lda + j*lda + i];
            }
            for (int j = nr_act; j < NR; j++) {
                *packed_b++ = 0.0;
            }
        }        
    }
}

void pack_a_default(const int lda, int mc_act, int kc_act, double* A, double* packed_a) {
    // double* copy = packed_a;
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
    // packed_a = copy;
    // printf("%s", "print packed a \n");
    // for (int pan_ptr = 0; pan_ptr < mc_act; pan_ptr+=MR) {
    //     for (int i = 0; i < MR; i++) {
    //         for (int j = 0; j < kc_act; j++) {
    //             printf("%f ", packed_a[pan_ptr*kc_act + i+j*MR]);
    //         }
    //         printf("%s", "\n");
    //     }
    // }
    // printf("%s", "\n");
}
void pack_a_8xKC(const int lda, int mc_act, int kc_act, double* A, double* packed_a) {
    // double* copy = packed_a;

    for (int pan_ptr = 0; pan_ptr < mc_act; pan_ptr+=MR) {
        int mr_act = min(MR, mc_act - pan_ptr);
        if (mr_act == MR) {
            int unroll = 2;
            int i = 0;
            for (; i < kc_act/unroll*unroll; i+=unroll) {
                __m256d a_0_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                i++;
                __m256d a_0_col1 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col1 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                // i++;
                // __m256d a_0_col2 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                // __m256d a_1_col2 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                // i++;
                // __m256d a_0_col3 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                // __m256d a_1_col3 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                i-=unroll-1;

                _mm256_store_pd(packed_a, a_0_col0);
                _mm256_store_pd(packed_a+4, a_1_col0);

                _mm256_store_pd(packed_a+MR, a_0_col1);
                _mm256_store_pd(packed_a+MR+4, a_1_col1);

                // _mm256_store_pd(packed_a+MR*2, a_0_col2);
                // _mm256_store_pd(packed_a+MR*2+4, a_1_col2);

                // _mm256_store_pd(packed_a+MR*3, a_0_col3);
                // _mm256_store_pd(packed_a+MR*3+4, a_1_col3);

                packed_a += MR*unroll;
            }

            for (; i < kc_act; i++) {
                __m256d a_0_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                _mm256_store_pd(packed_a, a_0_col0);
                _mm256_store_pd(packed_a+4, a_1_col0);
                packed_a += MR;
            }
        } else {
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

}
void pack_a_12xKC(const int lda, int mc_act, int kc_act, double* A, double* packed_a) {
    // double* copy = packed_a;

    for (int pan_ptr = 0; pan_ptr < mc_act; pan_ptr+=MR) {
        int mr_act = min(MR, mc_act - pan_ptr);
        if (mr_act == MR) {
            int unroll = 3;
            int i = 0;
            for (; i < kc_act/unroll*unroll; i+=unroll) {
                __m256d a_0_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                __m256d a_2_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i+8);
                i++;
                __m256d a_0_col1 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col1 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                __m256d a_2_col1 = _mm256_loadu_pd(A+pan_ptr + lda*i+8); 
                i++;
                __m256d a_0_col2 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col2 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                __m256d a_2_col2 = _mm256_loadu_pd(A+pan_ptr + lda*i+8); 
                // i++;
                // __m256d a_0_col3 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                // __m256d a_1_col3 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                // __m256d a_2_col3 = _mm256_loadu_pd(A+pan_ptr + lda*i+8); 
                i-=unroll-1;

                _mm256_store_pd(packed_a, a_0_col0);
                _mm256_store_pd(packed_a+4, a_1_col0);
                _mm256_store_pd(packed_a+8, a_2_col0);

                _mm256_store_pd(packed_a+MR, a_0_col1);
                _mm256_store_pd(packed_a+MR+4, a_1_col1);
                _mm256_store_pd(packed_a+MR+8, a_2_col1);  

                _mm256_store_pd(packed_a+MR*2, a_0_col2);
                _mm256_store_pd(packed_a+MR*2+4, a_1_col2);
                _mm256_store_pd(packed_a+MR*2+8, a_2_col2);  

                // _mm256_store_pd(packed_a+MR*3, a_0_col3);
                // _mm256_store_pd(packed_a+MR*3+4, a_1_col3);
                // _mm256_store_pd(packed_a+MR*3+8, a_2_col3);  

                packed_a += MR*unroll;
            }

            for (; i < kc_act; i++) {
                __m256d a_0_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i);
                __m256d a_1_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i+4);
                __m256d a_2_col0 = _mm256_loadu_pd(A+pan_ptr + lda*i+8);
                _mm256_store_pd(packed_a, a_0_col0);
                _mm256_store_pd(packed_a+4, a_1_col0);
                _mm256_store_pd(packed_a+8, a_2_col0);
                packed_a += MR;
            }
        } else {
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
    // packed_a = copy;
    // printf("%s", "print packed a \n");
    // for (int pan_ptr = 0; pan_ptr < mc_act; pan_ptr+=MR) {
    //     for (int i = 0; i < MR; i++) {
    //         for (int j = 0; j < kc_act; j++) {
    //             printf("%f ", packed_a[pan_ptr*kc_act + i+j*MR]);
    //         }
    //         printf("%s", "\n");
    //     }
    // }
    // printf("%s", "\n");
}

void pad_c_default(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* C, double* micro_tile_c) {
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

    //print tile
    // printf("%s", "padded c: \n");
    // for (int i = 0; i < MR; i++) {
    //     for (int j = 0; j < NR; j++) {
    //         printf("%f ", micro_tile_c[i+j*MR]);
    //     }
    //     printf("\n");
    // }
}

void pad_c_12x4(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* C, double* micro_tile_c) {
    int idx = 0;
    if (mr_act == MR && nr_act == NR) {
        //vectorize full tiles
        __m256d c_0_col0 = _mm256_loadu_pd(C);
        __m256d c_1_col0 = _mm256_loadu_pd(C+4);
        __m256d c_2_col0 = _mm256_loadu_pd(C+8);

        __m256d c_0_col1 = _mm256_loadu_pd(C+lda*1);
        __m256d c_1_col1 = _mm256_loadu_pd(C+lda*1+4);
        __m256d c_2_col1 = _mm256_loadu_pd(C+lda*1+8);

        __m256d c_0_col2 = _mm256_loadu_pd(C+lda*2);
        __m256d c_1_col2 = _mm256_loadu_pd(C+lda*2+4);
        __m256d c_2_col2 = _mm256_loadu_pd(C+lda*2+8);

        __m256d c_0_col3 = _mm256_loadu_pd(C+lda*3);
        __m256d c_1_col3 = _mm256_loadu_pd(C+lda*3+4);
        __m256d c_2_col3 = _mm256_loadu_pd(C+lda*3+8);
        
        _mm256_store_pd(micro_tile_c, c_0_col0);
        _mm256_store_pd(micro_tile_c+4, c_1_col0);
        _mm256_store_pd(micro_tile_c+8, c_2_col0);

        _mm256_store_pd(micro_tile_c+12, c_0_col1);
        _mm256_store_pd(micro_tile_c+12+4, c_1_col1);
        _mm256_store_pd(micro_tile_c+12+8, c_2_col1);

        _mm256_store_pd(micro_tile_c+24, c_0_col2);
        _mm256_store_pd(micro_tile_c+24+4, c_1_col2);
        _mm256_store_pd(micro_tile_c+24+8, c_2_col2);    

        _mm256_store_pd(micro_tile_c+36, c_0_col3);
        _mm256_store_pd(micro_tile_c+36+4, c_1_col3);
        _mm256_store_pd(micro_tile_c+36+8, c_2_col3);
    } else if (mr_act == MR && nr_act != NR) {
        for (int j = 0; j < nr_act; j++) {
            for (int i = 0; i < mr_act; i++) {
                micro_tile_c[idx++] = C[i+j*lda];
            }
        }        
        for (int j = nr_act; j < NR; j++) {
            for (int i = 0; i < MR; i++) {
                micro_tile_c[idx++] = 0.0;
            }
        }
    } else if (mr_act != MR && nr_act == NR) {
        for (int j = 0; j < nr_act; j++) {
            for (int i = 0; i < mr_act; i++) {
                micro_tile_c[idx++] = C[i+j*lda];
            }
            for (int i = mr_act; i < MR; i++) {
                micro_tile_c[idx++] = 0.0;
            }
        }
    } else {
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
    //print tile
    // printf("%s", "padded c: \n");
    // for (int i = 0; i < MR; i++) {
    //     for (int j = 0; j < NR; j++) {
    //         printf("%f ", micro_tile_c[i+j*MR]);
    //     }
    //     printf("\n");
    // }
}

void micro_kernel_4x4(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_col0 = _mm256_load_pd(micro_tile_c);
    __m256d c_col1 = _mm256_load_pd(micro_tile_c+4);
    __m256d c_col2 = _mm256_load_pd(micro_tile_c+8);
    __m256d c_col3 = _mm256_load_pd(micro_tile_c+12);

    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    for (int i = 0; i < kc_act; i+=1) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        __m256d a_col0 = _mm256_load_pd(pan_a);

        // printf("b values: %f %f %f %f \n", *(pan_b), *(pan_b+1), *(pan_b+2), *(pan_b+3));
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
    _mm256_store_pd(micro_tile_c, c_col0);
    _mm256_store_pd(micro_tile_c+4, c_col1);
    _mm256_store_pd(micro_tile_c+8, c_col2);
    _mm256_store_pd(micro_tile_c+12, c_col3);

    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] = micro_tile_c[i+j*MR];
        }
    }
}
void micro_kernel_8x4(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_load_pd(micro_tile_c);
    __m256d c_1_col0 = _mm256_load_pd(micro_tile_c+4);
    __m256d c_0_col1 = _mm256_load_pd(micro_tile_c+8);
    __m256d c_1_col1 = _mm256_load_pd(micro_tile_c+12);

    __m256d c_0_col2 = _mm256_load_pd(micro_tile_c+16);
    __m256d c_1_col2 = _mm256_load_pd(micro_tile_c+4+16);
    __m256d c_0_col3 = _mm256_load_pd(micro_tile_c+8+16);
    __m256d c_1_col3 = _mm256_load_pd(micro_tile_c+12+16);
    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    for (int i = 0; i < kc_act; i+=1) {
        //Compute a 12x1 x 1x12 outer-product by loading from A and broadcasting the value of B
        __m256d a_0_col0 = _mm256_load_pd(pan_a);
        __m256d a_1_col0 = _mm256_load_pd(pan_a+4);

        __m256d b_0 = _mm256_broadcast_sd(pan_b);
        __m256d b_1 = _mm256_broadcast_sd(pan_b+1);
        __m256d b_2 = _mm256_broadcast_sd(pan_b+2);
        __m256d b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        pan_b += NR;
        pan_a += MR;
    }

    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);
    _mm256_store_pd(micro_tile_c+8, c_0_col1);
    _mm256_store_pd(micro_tile_c+12, c_1_col1);

    _mm256_store_pd(micro_tile_c+16, c_0_col2);
    _mm256_store_pd(micro_tile_c+4+16, c_1_col2);
    _mm256_store_pd(micro_tile_c+8+16, c_0_col3);
    _mm256_store_pd(micro_tile_c+12+16, c_1_col3);
    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] = micro_tile_c[i+j*MR];
        }
    }
}
void micro_kernel_12x4(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_load_pd(micro_tile_c);
    __m256d c_1_col0 = _mm256_load_pd(micro_tile_c+4);
    __m256d c_2_col0 = _mm256_load_pd(micro_tile_c+8);

    __m256d c_0_col1 = _mm256_load_pd(micro_tile_c+MR);
    __m256d c_1_col1 = _mm256_load_pd(micro_tile_c+MR+4);
    __m256d c_2_col1 = _mm256_load_pd(micro_tile_c+MR+8);

    __m256d c_0_col2 = _mm256_load_pd(micro_tile_c+MR*2);
    __m256d c_1_col2 = _mm256_load_pd(micro_tile_c+MR*2+4);
    __m256d c_2_col2 = _mm256_load_pd(micro_tile_c+MR*2+8);

    __m256d c_0_col3 = _mm256_load_pd(micro_tile_c+MR*3);
    __m256d c_1_col3 = _mm256_load_pd(micro_tile_c+MR*3+4);
    __m256d c_2_col3 = _mm256_load_pd(micro_tile_c+MR*3+8);
    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    int unroll = 2;
    __m256d a_0_col0, a_1_col0, a_2_col0;
    __m256d b_0, b_1, b_2, b_3;
    for (int i = 0; i < kc_act/unroll*unroll; i+=unroll) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);
        a_2_col0 = _mm256_load_pd(pan_a+8);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_2_col0 = _mm256_fmadd_pd(a_2_col0, b_0, c_2_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_2_col1 = _mm256_fmadd_pd(a_2_col0, b_1, c_2_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_2_col2 = _mm256_fmadd_pd(a_2_col0, b_2, c_2_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);
        c_2_col3 = _mm256_fmadd_pd(a_2_col0, b_3, c_2_col3);

        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
                a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);
        a_2_col0 = _mm256_load_pd(pan_a+8);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_2_col0 = _mm256_fmadd_pd(a_2_col0, b_0, c_2_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_2_col1 = _mm256_fmadd_pd(a_2_col0, b_1, c_2_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_2_col2 = _mm256_fmadd_pd(a_2_col0, b_2, c_2_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);
        c_2_col3 = _mm256_fmadd_pd(a_2_col0, b_3, c_2_col3);

        pan_b += NR;
        pan_a += MR;
    }
    for (int i = kc_act/unroll*unroll; i < kc_act; i++) {
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);
        a_2_col0 = _mm256_load_pd(pan_a+8);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_2_col0 = _mm256_fmadd_pd(a_2_col0, b_0, c_2_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_2_col1 = _mm256_fmadd_pd(a_2_col0, b_1, c_2_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_2_col2 = _mm256_fmadd_pd(a_2_col0, b_2, c_2_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);
        c_2_col3 = _mm256_fmadd_pd(a_2_col0, b_3, c_2_col3);

        pan_b += NR;
        pan_a += MR;
    }
    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);
    _mm256_store_pd(micro_tile_c+8, c_2_col0);

    _mm256_store_pd(micro_tile_c+12, c_0_col1);
    _mm256_store_pd(micro_tile_c+12+4, c_1_col1);
    _mm256_store_pd(micro_tile_c+12+8, c_2_col1);

    _mm256_store_pd(micro_tile_c+24, c_0_col2);
    _mm256_store_pd(micro_tile_c+24+4, c_1_col2);
    _mm256_store_pd(micro_tile_c+24+8, c_2_col2);    

    _mm256_store_pd(micro_tile_c+36, c_0_col3);
    _mm256_store_pd(micro_tile_c+36+4, c_1_col3);
    _mm256_store_pd(micro_tile_c+36+8, c_2_col3);
    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] = micro_tile_c[i+j*MR];
        }
    }
}

void micro_kernel_4x4_no_packing(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_col0 = _mm256_set1_pd(0);
    __m256d c_col1 = _mm256_set1_pd(0);
    __m256d c_col2 = _mm256_set1_pd(0);
    __m256d c_col3 = _mm256_set1_pd(0);

    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    for (int i = 0; i < kc_act; i+=1) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        __m256d a_col0 = _mm256_load_pd(pan_a);

        // printf("b values: %f %f %f %f \n", *(pan_b), *(pan_b+1), *(pan_b+2), *(pan_b+3));
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
    _mm256_store_pd(micro_tile_c, c_col0);
    _mm256_store_pd(micro_tile_c+4, c_col1);
    _mm256_store_pd(micro_tile_c+8, c_col2);
    _mm256_store_pd(micro_tile_c+12, c_col3);

    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] += micro_tile_c[i+j*MR];
        }
    }
}
void micro_kernel_8x4_no_packing(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_set1_pd(0);
    __m256d c_1_col0 = _mm256_set1_pd(0);
    __m256d c_0_col1 = _mm256_set1_pd(0);
    __m256d c_1_col1 = _mm256_set1_pd(0);

    __m256d c_0_col2 = _mm256_set1_pd(0);
    __m256d c_1_col2 = _mm256_set1_pd(0);
    __m256d c_0_col3 = _mm256_set1_pd(0);
    __m256d c_1_col3 = _mm256_set1_pd(0);
    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    for (int i = 0; i < kc_act; i+=1) {
        //Compute a 12x1 x 1x12 outer-product by loading from A and broadcasting the value of B
        __m256d a_0_col0 = _mm256_load_pd(pan_a);
        __m256d a_1_col0 = _mm256_load_pd(pan_a+4);

        __m256d b_0 = _mm256_broadcast_sd(pan_b);
        __m256d b_1 = _mm256_broadcast_sd(pan_b+1);
        __m256d b_2 = _mm256_broadcast_sd(pan_b+2);
        __m256d b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        pan_b += NR;
        pan_a += MR;
    }

    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);
    _mm256_store_pd(micro_tile_c+8, c_0_col1);
    _mm256_store_pd(micro_tile_c+12, c_1_col1);

    _mm256_store_pd(micro_tile_c+16, c_0_col2);
    _mm256_store_pd(micro_tile_c+4+16, c_1_col2);
    _mm256_store_pd(micro_tile_c+8+16, c_0_col3);
    _mm256_store_pd(micro_tile_c+12+16, c_1_col3);
    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] += micro_tile_c[i+j*MR];
        }
    }
}
void micro_kernel_8x5_no_packing(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_set1_pd(0);
    __m256d c_1_col0 = _mm256_set1_pd(0);
    __m256d c_0_col1 = _mm256_set1_pd(0);
    __m256d c_1_col1 = _mm256_set1_pd(0);

    __m256d c_0_col2 = _mm256_set1_pd(0);
    __m256d c_1_col2 = _mm256_set1_pd(0);
    __m256d c_0_col3 = _mm256_set1_pd(0);
    __m256d c_1_col3 = _mm256_set1_pd(0);

    __m256d c_0_col4 = _mm256_set1_pd(0);
    __m256d c_1_col4 = _mm256_set1_pd(0);
    // __m256d c_0_col5 = _mm256_set1_pd(0);
    // __m256d c_1_col5 = _mm256_set1_pd(0);
    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    int unroll = 1;
    __m256d a_0_col0, a_1_col0;
    __m256d b_0, b_1, b_2, b_3, b_4;
    for (int i = 0; i < kc_act/unroll*unroll; i+=unroll) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);
        // b_5 = _mm256_broadcast_sd(pan_b+5);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        
        // c_0_col5 = _mm256_fmadd_pd(a_0_col0, b_5, c_0_col5);
        // c_1_col5= _mm256_fmadd_pd(a_1_col0, b_5, c_1_col5);    
        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
    }

    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);

    _mm256_store_pd(micro_tile_c+MR, c_0_col1);
    _mm256_store_pd(micro_tile_c+MR+4, c_1_col1);

    _mm256_store_pd(micro_tile_c+MR*2, c_0_col2);
    _mm256_store_pd(micro_tile_c+MR*2+4, c_1_col2);

    _mm256_store_pd(micro_tile_c+MR*3, c_0_col3);
    _mm256_store_pd(micro_tile_c+MR*3+4, c_1_col3);

    _mm256_store_pd(micro_tile_c+MR*4, c_0_col4);
    _mm256_store_pd(micro_tile_c+MR*4+4, c_1_col4);

    // _mm256_store_pd(micro_tile_c+MR*5, c_0_col5);
    // _mm256_store_pd(micro_tile_c+MR*5+4, c_1_col5);

    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] += micro_tile_c[i+j*MR];
        }
    }
}
void micro_kernel_8x5_no_packing_unrolled(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_set1_pd(0);
    __m256d c_1_col0 = _mm256_set1_pd(0);
    __m256d c_0_col1 = _mm256_set1_pd(0);
    __m256d c_1_col1 = _mm256_set1_pd(0);

    __m256d c_0_col2 = _mm256_set1_pd(0);
    __m256d c_1_col2 = _mm256_set1_pd(0);
    __m256d c_0_col3 = _mm256_set1_pd(0);
    __m256d c_1_col3 = _mm256_set1_pd(0);

    __m256d c_0_col4 = _mm256_set1_pd(0);
    __m256d c_1_col4 = _mm256_set1_pd(0);

    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    int unroll = 4;
    __m256d a_0_col0, a_1_col0;
    __m256d b_0, b_1, b_2, b_3, b_4;
    for (int i = 0; i < kc_act/unroll*unroll; i+=unroll) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        
        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        
        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        
        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        pan_b += NR;
        pan_a += MR;
        /*------------------------------------------------*/
    }
    for (int i = kc_act/unroll*unroll; i < kc_act; i++) {
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        
        pan_b += NR;
        pan_a += MR;
    }

    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);

    _mm256_store_pd(micro_tile_c+MR, c_0_col1);
    _mm256_store_pd(micro_tile_c+MR+4, c_1_col1);

    _mm256_store_pd(micro_tile_c+MR*2, c_0_col2);
    _mm256_store_pd(micro_tile_c+MR*2+4, c_1_col2);

    _mm256_store_pd(micro_tile_c+MR*3, c_0_col3);
    _mm256_store_pd(micro_tile_c+MR*3+4, c_1_col3);

    _mm256_store_pd(micro_tile_c+MR*4, c_0_col4);
    _mm256_store_pd(micro_tile_c+MR*4+4, c_1_col4);


    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] += micro_tile_c[i+j*MR];
        }
    }
}


void micro_kernel_8x6_no_packing(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_set1_pd(0);
    __m256d c_1_col0 = _mm256_set1_pd(0);
    __m256d c_0_col1 = _mm256_set1_pd(0);
    __m256d c_1_col1 = _mm256_set1_pd(0);

    __m256d c_0_col2 = _mm256_set1_pd(0);
    __m256d c_1_col2 = _mm256_set1_pd(0);
    __m256d c_0_col3 = _mm256_set1_pd(0);
    __m256d c_1_col3 = _mm256_set1_pd(0);

    __m256d c_0_col4 = _mm256_set1_pd(0);
    __m256d c_1_col4 = _mm256_set1_pd(0);
    __m256d c_0_col5 = _mm256_set1_pd(0);
    __m256d c_1_col5 = _mm256_set1_pd(0);
    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    int unroll = 1;
    __m256d a_0_col0, a_1_col0;
    __m256d b_0, b_1, b_2, b_3, b_4, b_5;
    for (int i = 0; i < kc_act/unroll*unroll; i+=unroll) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);
        b_4 = _mm256_broadcast_sd(pan_b+4);
        b_5 = _mm256_broadcast_sd(pan_b+5);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);

        c_0_col4 = _mm256_fmadd_pd(a_0_col0, b_4, c_0_col4);
        c_1_col4 = _mm256_fmadd_pd(a_1_col0, b_4, c_1_col4);
        
        c_0_col5 = _mm256_fmadd_pd(a_0_col0, b_5, c_0_col5);
        c_1_col5= _mm256_fmadd_pd(a_1_col0, b_5, c_1_col5);    
        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
    }

    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);

    _mm256_store_pd(micro_tile_c+MR, c_0_col1);
    _mm256_store_pd(micro_tile_c+MR+4, c_1_col1);

    _mm256_store_pd(micro_tile_c+MR*2, c_0_col2);
    _mm256_store_pd(micro_tile_c+MR*2+4, c_1_col2);

    _mm256_store_pd(micro_tile_c+MR*3, c_0_col3);
    _mm256_store_pd(micro_tile_c+MR*3+4, c_1_col3);

    _mm256_store_pd(micro_tile_c+MR*4, c_0_col4);
    _mm256_store_pd(micro_tile_c+MR*4+4, c_1_col4);

    _mm256_store_pd(micro_tile_c+MR*5, c_0_col5);
    _mm256_store_pd(micro_tile_c+MR*5+4, c_1_col5);

    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] += micro_tile_c[i+j*MR];
        }
    }
}
void micro_kernel_12x4_no_packing(const int lda, int kc_act, int mc_act, int nr_act, int mr_act, double* pan_a, double* pan_b, double* C, double* micro_tile_c) {
    //4x4 micro-kernel
    // can not pack c and b in NR direction

    //Load from C (padded to be MRxNR into micro-tile)
    __m256d c_0_col0 = _mm256_set1_pd(0);
    __m256d c_1_col0 = _mm256_set1_pd(0);
    __m256d c_2_col0 = _mm256_set1_pd(0);

    __m256d c_0_col1 = _mm256_set1_pd(0);
    __m256d c_1_col1 = _mm256_set1_pd(0);
    __m256d c_2_col1 = _mm256_set1_pd(0);

    __m256d c_0_col2 = _mm256_set1_pd(0);
    __m256d c_1_col2 = _mm256_set1_pd(0);
    __m256d c_2_col2 = _mm256_set1_pd(0);

    __m256d c_0_col3 = _mm256_set1_pd(0);
    __m256d c_1_col3 = _mm256_set1_pd(0);
    __m256d c_2_col3 = _mm256_set1_pd(0);
    // Panels of A,B padded so they are MRxkc_act kc_actxNR respectively
    int unroll = 2;
    __m256d a_0_col0, a_1_col0, a_2_col0;
    __m256d b_0, b_1, b_2, b_3;
    for (int i = 0; i < kc_act/unroll*unroll; i+=unroll) {
        //Compute a 4x1 x 1x4 outer-product by loading from A and broadcasting the value of B
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);
        a_2_col0 = _mm256_load_pd(pan_a+8);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_2_col0 = _mm256_fmadd_pd(a_2_col0, b_0, c_2_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_2_col1 = _mm256_fmadd_pd(a_2_col0, b_1, c_2_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_2_col2 = _mm256_fmadd_pd(a_2_col0, b_2, c_2_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);
        c_2_col3 = _mm256_fmadd_pd(a_2_col0, b_3, c_2_col3);

        pan_b += NR;
        pan_a += MR;
        /*-------------------------------------------------------------*/
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);
        a_2_col0 = _mm256_load_pd(pan_a+8);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_2_col0 = _mm256_fmadd_pd(a_2_col0, b_0, c_2_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_2_col1 = _mm256_fmadd_pd(a_2_col0, b_1, c_2_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_2_col2 = _mm256_fmadd_pd(a_2_col0, b_2, c_2_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);
        c_2_col3 = _mm256_fmadd_pd(a_2_col0, b_3, c_2_col3);

        pan_b += NR;
        pan_a += MR;
    }
    for (int i = kc_act/unroll*unroll; i < kc_act; i++) {
        a_0_col0 = _mm256_load_pd(pan_a);
        a_1_col0 = _mm256_load_pd(pan_a+4);
        a_2_col0 = _mm256_load_pd(pan_a+8);

        b_0 = _mm256_broadcast_sd(pan_b);
        b_1 = _mm256_broadcast_sd(pan_b+1);
        b_2 = _mm256_broadcast_sd(pan_b+2);
        b_3 = _mm256_broadcast_sd(pan_b+3);

        c_0_col0 = _mm256_fmadd_pd(a_0_col0, b_0, c_0_col0);
        c_1_col0 = _mm256_fmadd_pd(a_1_col0, b_0, c_1_col0);
        c_2_col0 = _mm256_fmadd_pd(a_2_col0, b_0, c_2_col0);

        c_0_col1 = _mm256_fmadd_pd(a_0_col0, b_1, c_0_col1);
        c_1_col1 = _mm256_fmadd_pd(a_1_col0, b_1, c_1_col1);
        c_2_col1 = _mm256_fmadd_pd(a_2_col0, b_1, c_2_col1);

        c_0_col2 = _mm256_fmadd_pd(a_0_col0, b_2, c_0_col2);
        c_1_col2 = _mm256_fmadd_pd(a_1_col0, b_2, c_1_col2);
        c_2_col2 = _mm256_fmadd_pd(a_2_col0, b_2, c_2_col2);

        c_0_col3 = _mm256_fmadd_pd(a_0_col0, b_3, c_0_col3);
        c_1_col3 = _mm256_fmadd_pd(a_1_col0, b_3, c_1_col3);
        c_2_col3 = _mm256_fmadd_pd(a_2_col0, b_3, c_2_col3);

        pan_b += NR;
        pan_a += MR;
    }
    //Store first into microtile
    _mm256_store_pd(micro_tile_c, c_0_col0);
    _mm256_store_pd(micro_tile_c+4, c_1_col0);
    _mm256_store_pd(micro_tile_c+8, c_2_col0);

    _mm256_store_pd(micro_tile_c+12, c_0_col1);
    _mm256_store_pd(micro_tile_c+12+4, c_1_col1);
    _mm256_store_pd(micro_tile_c+12+8, c_2_col1);

    _mm256_store_pd(micro_tile_c+24, c_0_col2);
    _mm256_store_pd(micro_tile_c+24+4, c_1_col2);
    _mm256_store_pd(micro_tile_c+24+8, c_2_col2);    

    _mm256_store_pd(micro_tile_c+36, c_0_col3);
    _mm256_store_pd(micro_tile_c+36+4, c_1_col3);
    _mm256_store_pd(micro_tile_c+36+8, c_2_col3);
    //Unpack microtile into C, keeping actual size of C in mind
    for (int j = 0; j < nr_act; j++) {
        for (int i = 0; i < mr_act; i++) {
            C[i+j*lda] += micro_tile_c[i+j*MR];
        }
    }
}

void block_panel_multiply(const int lda, int kc_act, int mc_act, int nr_act, double* packed_a, double* packed_b, double* C, double* micro_tile_c) {
    for (int a_pan_ptr = 0; a_pan_ptr < mc_act; a_pan_ptr += MR) {
        int mr_act = min(MR, mc_act-a_pan_ptr);
        // memset(micro_tile_c, 0, sizeof(*micro_tile_c));
        micro_kernel_8x5_no_packing_unrolled(lda, kc_act, mc_act, nr_act, mr_act, packed_a + a_pan_ptr*kc_act, packed_b, C + a_pan_ptr, micro_tile_c);
    }
}

void block_multi_panel_multiply(const int lda, int kc_act, int mc_act, double* packed_a, double* packed_b, double* C, double* micro_tile_c) {
    for (int b_pan_ptr = 0; b_pan_ptr < lda; b_pan_ptr+=NR) {
        int nr_act = min(NR, lda-b_pan_ptr);
        block_panel_multiply(lda, kc_act, mc_act, nr_act, packed_a, packed_b + b_pan_ptr*kc_act, C + b_pan_ptr*lda, micro_tile_c);
    }
}

void outer_proudct(const int lda, int kc_act, double* A, double* packed_b, double* C, double* packed_a, double* micro_tile_c) {
    for (int block_ptr = 0; block_ptr < lda; block_ptr+=MC) {
        // pack A
        int mc_act = min(MC, lda - block_ptr);
        pack_a_8xKC(lda, mc_act, kc_act, A + block_ptr, packed_a);
        
        block_multi_panel_multiply(lda, kc_act, mc_act, packed_a, packed_b, C + block_ptr,micro_tile_c);
    }
}

void square_dgemm(const int lda, double* A, double* B, double* C) {    
    MC = 144 - 16 * ((lda > 150) && (lda < 200));

    double* packed_b = (double*) _mm_malloc((lda+NR)*KC *sizeof(double), 64);
    double* micro_tile_c = (double*) _mm_malloc(MR*NR * sizeof(double), 64);
    double* packed_a = (double*) _mm_malloc(MC*KC *sizeof(double), 64);

    for (int j = 0; j < lda; j += KC) {
        int KC_act = min(KC, lda-j);
        pack_b_unrolled(lda, KC_act, B + j, packed_b);
        outer_proudct(lda, KC_act, A + lda * j, packed_b, C, packed_a, micro_tile_c);
    }

    _mm_free(packed_b);
    _mm_free(packed_a);
    _mm_free(micro_tile_c);
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
// To test correctness
// gcc -g -mavx -mfma -fsanitize=address matrix.c -o mat

// int main(int argc, char *argv[]) {
//     int lda = 651;
//     double* A = (double*) malloc(lda*lda*sizeof(double));
//     double* B = (double*) malloc(lda*lda*sizeof(double));
//     double* C = (double*) malloc(lda*lda*sizeof(double));
//     double* D = (double*) malloc(lda*lda*sizeof(double));
//     for (int i = 0; i < lda; i++) {
//         for (int j = 0; j < lda; j++) {
//             A[j+i*lda] =  B[j+i*lda] =C[j+i*lda] = D[j+i*lda] = j+i*lda;
//         }
//     }
//     naive(lda, A, B, D);
//     square_dgemm(lda, A, B, C);
//     int correct = 1;
//     for (int i = 0; i <lda*lda; i++){
//         // printf("%f ", D[i]-C[i]);
//         correct = correct & (D[i]-C[i] == 0.0);
//     }
//     printf("\n correct?: %d \n", correct);
//     // print_matrix_tran(lda, C);
//     // print_matrix_tran(lda, D);
//     free(A);
//     free(B);
//     free(C);
//     free(D);
// }