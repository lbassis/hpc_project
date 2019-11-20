#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "util.h"
#include "perf.h"
#include "my_lib.h"

#ifndef NB_LOOP
#define NB_LOOP 100000000
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 256
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 1048576
#endif


int test_version(char *id, long long int step, double (*ddot)(const long long int, const double*, const long long int, const double*, const long long int)) {

  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

  int l, nb_loop = NB_LOOP;
  double performance;
  perf_t start,stop;
  // Executions a vide, flush potentiel, ...

  // Performance d'une addition scalaire
  long long int n = 0;
  double *a = alloc_vec(MAX_SIZE);
  double *b = alloc_vec(MAX_SIZE);

  LAPACKE_dlarnv_work(1, ISEED, MAX_SIZE, a);
  LAPACKE_dlarnv_work(1, ISEED, MAX_SIZE, b);

  double result;
  for(n = MIN_SIZE; n < MAX_SIZE; n *= 2){
    long flop = 2*n/step;

    perf(&start);
    for(l = 0; l < nb_loop; l++){
      result = ddot(n/step, a, step, b, step);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, flop * nb_loop);
    printf("%s, %lld, %lld, %lf, ", id, n, step, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");

    nb_loop = (nb_loop < 10)? 5:(int)nb_loop/2;
  }

  free(a);
  free(b);

  return result;
}


int main() {
  printf("version, n, step, Mflops, us\n");

  double my_one = test_version("my", 1, &my_ddot);
  double mkl_one = test_version("mkl", 1, &cblas_ddot);
  double my_two = test_version("my", 2, &my_ddot);
  double mkl_two = test_version("mkl", 2, &cblas_ddot);



  if (my_one == mkl_one && my_two == mkl_two) {
    printf("\nOK\n");
  }
  else {
    printf("\nFailed\n");
  }

  return 0;
}
