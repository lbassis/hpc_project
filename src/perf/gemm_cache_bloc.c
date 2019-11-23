#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#include "util.h"
#include "my_lib.h"
#include "perf.h"


#ifndef NB_LOOP
#define NB_LOOP 1
#endif



#ifndef MAX_SIZE
#define MAX_SIZE 300
#endif

#ifndef MIN_BLOC
#define MIN_BLOC 100
#endif

#ifndef MAX_BLOC
#define MAX_BLOC 250
#endif

int main() {

  long nb_loop = NB_LOOP;

  int IONE = 1;
  long long int ISEED[4] = {0,0,0,1};

  double performance;
  perf_t start,stop;
  printf("n, Mflops, ms\n");

  // Performance d'une addition scalaire
  int l = 0, n = 0;
  double *a = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *b = alloc_mat(MAX_SIZE, MAX_SIZE);
  double *c = alloc_mat(MAX_SIZE, MAX_SIZE);

  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, a);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, b);
  LAPACKE_dlarnv_work(IONE, ISEED, MAX_SIZE*MAX_SIZE, c);

  for(n = MIN_BLOC; n < MAX_BLOC; n+=5){
    perf(&start);
    for(l = 0; l < nb_loop; l++){
      my_dgemm_bloc(CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  /* m */ MAX_SIZE,
                  /* n */ MAX_SIZE,
                  /* k */ MAX_SIZE,
                  /* alpha */ 2.5, a, MAX_SIZE, b, MAX_SIZE, /* beta */ 1.3, c, MAX_SIZE,
                /* BLOC_SIZE */ n);
    }
    perf(&stop);
    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, 2 * MAX_SIZE * MAX_SIZE * MAX_SIZE * NB_LOOP);
    printf("%d, %lf, ", n, performance);

    perf_print_time(&stop, nb_loop);

    printf("\n");
  }

  free(a);
  free(b);
  free(c);
  return 0;
}
