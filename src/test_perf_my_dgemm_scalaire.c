#include <stdio.h>
#include <stdlib.h>

#include "my_lib.h"
#include <mkl.h>

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }

#ifndef NB_LOOP
#define NB_LOOP 10000//1000
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 100
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 1000
#endif


int main() {

  printf("version, n, Mflops, ms\n");
  test_version("mkl", &cblas_dgemm);
  test_version("ikj", &my_dgemm_scalaire);
  test_version("kij", &my_dgemm_scalaire_kij);
  test_version("ijk", &my_dgemm_scalaire_ijk);
  test_version("jik", &my_dgemm_scalaire_jik);
}

void test_version(char *id, void (*gemm)(int, int, int, int, int, int, double, double*, int, double*, int, double, double*, int)){

  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */
  long nb_loop = NB_LOOP;

  double performance;
  perf_t start,stop;

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat((unsigned long)MAX_SIZE, (unsigned long)MAX_SIZE);
  double *b = alloc_mat((unsigned long)MAX_SIZE, (unsigned long)MAX_SIZE);
  double *c = alloc_mat((unsigned long)MAX_SIZE, (unsigned long)MAX_SIZE);

  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, c);

  double result = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n += n / 4){
    long flop = 3*n*n;
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      (*gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., a, n, b, n, 0, c, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, flop * n);
    printf("%s, %d, %lf, ", id, n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
  }

  free(a);
  free(b);

  return 0;
}
