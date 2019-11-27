#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#include "util.h"
#include "my_lib.h"
#include "perf.h"


#ifndef NB_LOOP
#define NB_LOOP 10
#endif

#ifndef MIN_SIZE
#define MIN_SIZE 200
#endif

#ifndef MAX_SIZE
#define MAX_SIZE 400
#endif

#define BLOC_SIZE 130

int main(void) {

  long nb_loop = NB_LOOP;

  int IONE = 1;
  long long int ISEED[4] = {0,0,0,1};

  double performance;
  perf_t start,stop, start2, stop2;
  printf("version, n, Mflops, us\n");

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *b = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *c = alloc_mat(MAX_SIZE, MAX_SIZE);

  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, c);

  for(n = MIN_SIZE; n < MAX_SIZE; n+=5){
    double **a_t = lapack2tile(n, n, BLOC_SIZE, a, MAX_SIZE);
    double **b_t = lapack2tile(n, n, BLOC_SIZE, b, MAX_SIZE);
    double **c_t = lapack2tile(n, n, BLOC_SIZE, c, MAX_SIZE);
    perf(&start);
    
    for(l = 0; l < nb_loop; l++){
      my_dgemm_Tile_omp2(CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  /* m */ n,
                  /* n */ n,
                  /* k */ n,
                  /* alpha */ 2.5, a_t, MAX_SIZE, b_t, MAX_SIZE, /* beta */ 1.3, c_t, MAX_SIZE);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, 2 * MAX_SIZE * MAX_SIZE * MAX_SIZE * NB_LOOP);
    printf("omp out, %d, %lf, ", n, performance);

    perf_print_time(&stop, nb_loop);
    printf("\n");


    perf(&start2);
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgemm_Tile_omp(CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  /* m */ n,
                  /* n */ n,
                  /* k */ n,
                  /* alpha */ 2.5, a_t, MAX_SIZE, b_t, MAX_SIZE, /* beta */ 1.3, c_t, MAX_SIZE);
    }
    perf(&stop);
    tile2lapack(n, n, BLOC_SIZE, a_t, a, MAX_SIZE);
    tile2lapack(n, n, BLOC_SIZE, b_t, b, MAX_SIZE);
    tile2lapack(n, n, BLOC_SIZE, c_t, c, MAX_SIZE);
    perf(&stop2);
    free(a_t);
    free(b_t);
    free(c_t);

    perf_diff(&start, &stop);
    perf_diff(&start2, &stop2);
    performance = perf_mflops(&stop, 2 * MAX_SIZE * MAX_SIZE * MAX_SIZE * NB_LOOP);
    printf("omp in, %d, %lf, ", n, performance);
    perf_print_time(&stop, nb_loop);
    printf("\n");
    //performance = perf_mflops(&stop2, 2 * MAX_SIZE * MAX_SIZE * MAX_SIZE * NB_LOOP);
    //printf("tile + conversion, %d, %lf, ", n, performance);
    //perf_print_time(&stop2, nb_loop);
    //printf("\n");
  }

  free(a);
  free(b);
  free(c);
  return 0;
}
