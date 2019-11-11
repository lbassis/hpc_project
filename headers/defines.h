#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#include "util.h"
#include "ddot.h"


#define TEST_MAT(NAME, N, M, INIT, ...) void test_ ## NAME ## _(void){                        \
                                          double *a = alloc_mat(N, M), *b = alloc_mat(N, M);  \
                                          INIT_MAT(N, M, INIT)                                \
                                          CHECK(__VA_ARGS__)                                  \
                                          free(a);                                            \
                                          free(b);                                            \
                                        }

#define TEST_VEC(NAME, N, INIT, ...) void test_ ## NAME ## _(void){                           \
                                        double *a = alloc_vec(N), *b = alloc_vec(N);          \
                                        INIT_VEC(N, INIT)                                     \
                                        CHECK(__VA_ARGS__)                                    \
                                        free(a);                                              \
                                        free(b);                                              \
                                      }



#define INIT_MAT(N, M, ...) {int j = 0;                   \
                             for(j = 0; j < M; j++)       \
                                INIT_VEC(N, __VA_ARGS__)  \
                            }

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }


#define CHECK(...) if(fabs(my_ddot(__VA_ARGS__) - cblas_ddot(__VA_ARGS__)) > ESP) { \
                      printf("[FAILED] %s\n", __func__);                            \
                      nb_fail++;                                                    \
                    }

#define CALL(NAME) test_ ## NAME ## _();

#define ESP 1.11e-16
