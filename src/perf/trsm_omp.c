#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#include "util.h"
#include "my_lib.h"
#include "perf.h"


#ifndef NB_LOOP
#define NB_LOOP 300
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 50
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 200
#endif


int main() {

  long nb_loop = NB_LOOP;

  int IONE = 1;
  long long int ISEED[4] = {0,0,0,1};

  double performance;
  perf_t start,stop;
  printf("version, n, Mflops, us\n");

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *b = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *c = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *d = alloc_mat(MAX_SIZE, MAX_SIZE);

  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, c);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, d);

  //long long ipiv[MAX_SIZE] = {};
  for(n = MIN_SIZE; n < MAX_SIZE; n+=2){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dtrsm(LAPACK_COL_MAJOR, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, n, 1.3, a, n, b, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "myblas", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");

    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dtrsm_openmp(LAPACK_COL_MAJOR, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, n, 1.3, a, n, d, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "myblas_omp", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");


    perf(&start);
    for(l = 0; l < nb_loop; l++){
      cblas_dtrsm(LAPACK_COL_MAJOR, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, n, 1.3, a, n, c, n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "mkl", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
  }

  free(a);
  free(b);
  free(c);
  free(d);
  return 0;
}
