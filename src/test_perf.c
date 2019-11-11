#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "ddot.h"
#include "perf.h"
//#include "cblas.h"

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
#define MAX_SIZE 1000000
#endif



int main(int argc, char* argv[]){
  long nb_loop = NB_LOOP;
  /*if(argc == 1){
    nb_loop = atoi(argv[0]);
  }*/

  double performance;
  perf_t start,stop;
  printf("n, Mflops, ms\n");
  long flop = 4;

  // Executions a vide, flush potentiel, ...

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_vec(MAX_SIZE), *b = alloc_vec(MAX_SIZE);
  INIT_VEC(MAX_SIZE, a[i] = i + 1; b[i] = i + 1)
  double result = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n += n / 4){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      result = my_ddot(n, a, 1, b, 1);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    printf("%d, ", n);
    performance = perf_mflops(&stop, flop * n);
    printf("%lf, ", performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
    // Verification
    /*if(fabs(result - cblas_ddot(n, a, 1, b, 1)) > ESP){
      printf("FAIL\n");
    }*/



  }

  free(a);
  free(b);

  return 0;
}
