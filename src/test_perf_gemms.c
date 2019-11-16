#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "ddot.h"
#include "perf.h"
#include "mkl.h"

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }

#ifndef NB_LOOP
#define NB_LOOP 10000//1000
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 10
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 100
#endif


int main() {
  test_version("ikj", &my_dgemm_scalaire);
  /* test_version("kij", &my_dgemm_scalaire_kij); */
  /* test_version("ijk", &my_dgemm_scalaire_ijk); */
  /* test_version("jik", &my_dgemm_scalaire_jik); */
}
  

void test_version(char *id, double (*gemm)()){

  long nb_loop = NB_LOOP;

  double performance;
  perf_t start,stop;
  printf("version, n, Mflops, ms\n");

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *b = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *c = alloc_mat(MAX_SIZE, MAX_SIZE);

  LAPACKE_zlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_zlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_zlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, c);

  double result = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n += n / 4){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      result = gemm(n, a, b, c);
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
