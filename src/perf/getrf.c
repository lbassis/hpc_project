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

  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);

  long long ipiv[MAX_SIZE] = {};
  for(n = MIN_SIZE; n < MAX_SIZE; n+=2){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, NULL);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "myblas", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
    for(l = 0; l < n * n; l++) a[l] = b[l];


    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgetrf_omp_gemm(LAPACK_COL_MAJOR, n, n, a, n, NULL);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "myblas_omp_gemm", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
    for(l = 0; l < n * n; l++) a[l] = b[l];


    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgetrf_omp_trsm_gemm(LAPACK_COL_MAJOR, n, n, a, n, NULL);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "myblas_omp_trsm_gemm", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
    for(l = 0; l < n * n; l++) a[l] = b[l];

/*
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgetrf_Tile(LAPACK_COL_MAJOR, n, n, a, n, NULL);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "myblas_tile", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");
    for(l = 0; l < n * n; l++) a[l] = b[l];
*/

/*
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, b, n, ipiv);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, (2 * n / 3) * n * n * NB_LOOP);
    printf("%s, %d, %lf, ", "mkl", n, performance);

    for(l = 0; l < n * n; l++) a[l] = b[l];

    perf_print_time(&stop, nb_loop);
    printf("\n");
    */
  }
  free(a);
  free(b);

  (void)ipiv;
  return 0;
}
