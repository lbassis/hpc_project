#include <stdio.h>
#include <stdlib.h>
#include "my_lib.h"
#include "perf.h"
#include "util.h"
#include <mkl.h>

#ifndef NB_LOOP
#define NB_LOOP 100
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 128
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 2048
#endif

void test_version(char *id, void (*gemm)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, double, double*, int, double*, int, double, double*, int)){

  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */
  long nb_loop = NB_LOOP;

  double performance;
  perf_t start,stop;

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *b = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *c = alloc_mat(MAX_SIZE, MAX_SIZE);

  LAPACKE_dlarnv_work(1, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(1, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_dlarnv_work(1, ISEED, MAX_SIZE*MAX_SIZE, c);

  double result = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n *= 2){
    long flop = 3*n*n;
    perf(&start);

    for(l = 0; l < nb_loop; l++){
      gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., a, n, b, n, 0, c, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, flop * nb_loop);
    printf("%s, %d, %lf, ", id, n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");

    nb_loop = (int)nb_loop/1.5;
  }

  free(a);
  free(b);

  return 0;
}

int main() {
  printf("version, n, Mflops, us\n");
  test_version("mkl", &cblas_dgemm);
  test_version("ikj", &my_dgemm_scalaire);
  test_version("kij", &my_dgemm_scalaire_kij);
  test_version("ijk", &my_dgemm_scalaire_ijk);
  test_version("jik", &my_dgemm_scalaire_jik);
}
