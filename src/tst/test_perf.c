#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "perf.h"
#include "defines.h"

#include "my_lib.h"

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }

#ifndef NB_LOOP
#define NB_LOOP 100//1000
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 10000
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 10000000
#endif



int main(void){
  long nb_loop = NB_LOOP;
  /*if(argc == 1){
    nb_loop = atoi(argv[0]);
  }*/

  double performance;
  perf_t start,stop;
  printf("n, Mflops, ms, Mflops_mkl, ms_mkl\n");
  long flop = 2;

  // Executions a vide, flush potentiel, ...

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_vec(MAX_SIZE), *b = alloc_vec(MAX_SIZE);
  INIT_VEC(MAX_SIZE, a[i] = i + 1; b[i] = i + 1)
  double result = 0;
  double result_mkl = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n += n / 8){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      result = my_ddot(n, a, 1, b, 1);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    printf("%d, ", n);
    performance = perf_mflops(&stop, flop * n * NB_LOOP);
    printf("%lf, ", performance);

    perf_print_time(&stop, nb_loop);

    perf(&start);
    for(l = 0; l < nb_loop; l++){
      result_mkl = cblas_ddot(n, a, 1, b, 1);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    printf("%d, ", n);
    performance = perf_mflops(&stop, flop * n * NB_LOOP);
    printf("%lf, ", performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
    // Verification
    assert(result - result_mkl < ESP);



  }

  free(a);
  free(b);

  return 0;
}
