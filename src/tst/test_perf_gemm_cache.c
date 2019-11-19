#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#include "util.h"
#include "my_lib.h"
#include "perf.h"


#ifndef NB_LOOP
#define NB_LOOP 20
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 50
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 250
#endif


int main() {

  long nb_loop = NB_LOOP;

  int IONE = 1;
  long long int ISEED[4] = {0,0,0,1};

  double performance;
  perf_t start,stop;
  printf("n, Mflops, ms, Mflops_mkl, ms_mkl\n");

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *b = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *c = alloc_mat(MAX_SIZE, MAX_SIZE);

  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, c);

  double result = 0;
  for(n = MIN_SIZE; n < MAX_SIZE; n+=2){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgemm_seq(CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  /* m */ n,
                  /* n */ n,
                  /* k */ n,
                  /* alpha */ 2.5, a, n, b, n, /* beta */ 1.3, c, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, 2 * n * n * n * NB_LOOP);
    printf("%d, %lf, ", n, performance);

    perf_print_time(&stop, nb_loop);

    perf(&start);
    for(l = 0; l < nb_loop; l++){
      cblas_dgemm(CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  /* m */ n,
                  /* n */ n,
                  /* k */ n,
                  /* alpha */ 2.5, a, n, b, n, /* beta */ 1.3, c, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, 2 * n * n * n * NB_LOOP);
    printf("%d, %lf, ", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
  }

  free(a);
  free(b);
  free(c);
  return 0;
}
