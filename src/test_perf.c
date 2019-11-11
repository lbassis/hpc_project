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
#define NB_LOOP 1000
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 100
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 1000000
#endif

#ifndef NB_ITER
#define NB_ITER 100
#endif


int main(int argc, char* argv[]){
  /*
  if(argc < 1){
    printf("Please enter the function to test\n");
    return 1;
  }
  int find = 0;
  for(find = 0; find < NB_FUNC; find++){
    if(strcpy((char*)(functions[find]), argv[1]) == 0){
      break;
    }
  }
  if(find == NB_FUNC){
    printf("Function not found\n");
    return 1;
  }
*/
  double performance;
  perf_t start,stop;

  long flop = 4;

  // Executions a vide, flush potentiel, ...

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_vec(MAX_SIZE), *b = alloc_vec(MAX_SIZE), *c = alloc_vec(MAX_SIZE);
  INIT_VEC(MAX_SIZE, a[i] = 1.0 / (i + 1); b[i] = i + 1; c[i] = i)
  double result = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n += MAX_SIZE / NB_ITER){
    perf(&start);
    for(l = 0; l < NB_LOOP; l++){
      result = my_ddot(n, a, 1, b, 1);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    printf("%d, ", n);

    perf_print_time(&stop, NB_LOOP);
    // Verification
    /*if(fabs(result - cblas_ddot(n, a, 1, b, 1)) > ESP){
      printf("FAIL\n");
    }*/

    // Performance
    #ifdef FLOP
    performance = perf_mflops(&stop, flop * n * NB_LOOP);
    printf("%lf\n", performance);
    #endif
  }

  free(a);
  free(b);

  return 0;
}
