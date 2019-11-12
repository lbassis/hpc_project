#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "ddot.h"
#include "perf.h"
//#include "cblas.h"

#define INIT_MAT(N, M, ...) {int j = 0;                   \
                             for(j = 0; j < M; j++)       \
                                INIT_VEC(N, __VA_ARGS__)  \
                            }

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }

#ifndef NB_LOOP
#define NB_LOOP 100//1000
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 100
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 1000
#endif



int main(int argc, char* argv[]){
  long nb_loop = NB_LOOP;
  /*if(argc == 1){
    nb_loop = atoi(argv[0]);
  }*/

  double performance;
  perf_t start,stop;
  printf("n, ikj, t_ikj, kij, t_kij, ijk, t_ijk, jik, t_jik\n");
  long flop = 4;

  // Executions a vide, flush potentiel, ...
  void* tab[4] = {&my_dgemm_scalaire, &my_dgemm_scalaire_kij, &my_dgemm_scalaire_ijk, &my_dgemm_scalaire_jik};

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_vec(MAX_SIZE * MAX_SIZE), *b = alloc_vec(MAX_SIZE * MAX_SIZE), *c = alloc_vec(MAX_SIZE * MAX_SIZE);
  INIT_VEC(MAX_SIZE * MAX_SIZE, a[i] = i + 1; b[i] = i + 1)
  long nb[2] = {MIN_SIZE, MAX_SIZE};
  for(n = 0;  n < 2 ; n++){
    int func = 0;
    printf("%ld, ", nb[n]);
    for(func = 0; func < 4; func++){
      perf(&start);
      for(l = 0; l < nb_loop; l++){
        (*((void (*)(int, double*, double*, double*))tab[func]))(nb[n], a, b, c);
      }
      perf(&stop);
      perf_diff(&start, &stop);
      performance = perf_mflops(&stop, flop * nb[n]);
      printf("%lf, ", performance);
      perf_print_time(&stop, nb_loop);
      printf(",");
      // Verification
      /*if(fabs(result - cblas_ddot(n, a, 1, b, 1)) > ESP){
        printf("FAIL\n");
      }*/
    }
    printf("\n");

  }

  free(a);
  free(b);
  free(c);

  return 0;
}
