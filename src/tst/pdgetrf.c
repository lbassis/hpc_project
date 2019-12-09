#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mkl.h>
#include <mpi.h>
#include "util.h"
#include "perf.h"
#include "defines.h"
#define TILE_SIZE 3

int main(void){
  printf("%s: \n", __FILE__);

  MPI_Init(NULL, NULL);

  int m = 10;
  int n = 8;
  int lda = m;
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};
  int dims[2] = {2, 2};

  int nb_proc, me;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  double* a = alloc_mat(lda, n);
  double* b = alloc_mat(lda, n);

  //LAPACKE_dlarnv_work(IONE, ISEED, n * lda, a);
  int i;
  for(i = 0; i < n * lda; i++) a[i] = i;

  double **a_Tile = lapack2tile( lda, n, 3, a, lda );
  double **b_Tile = lapack2tile( lda, n, 3, b, lda );
  double **out = alloc_dist_matrix(m, n, dims);

  if (me == 0)
    affiche(m, n, a, m, stdout);

  scatter_matrix(m, n, a_Tile, out, nb_proc, me, dims, MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);

  my_pdgetrf_tiled(LAPACK_COL_MAJOR, m, n, a_Tile, m, NULL);
  gather_matrix(m, n, out, b_Tile, nb_proc, me, dims, MPI_COMM_WORLD);

  tile2lapack( m, n, 3, b_Tile, b, lda );
  tile2lapack( m, n, 3, a_Tile, a, lda );

  if (me == 0){
    printf("____\n");
    affiche(m, n, b, m, stdout);
  }

  int MT = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int NT = (n + BLOC_SIZE - 1) / BLOC_SIZE;

  for(i = 0; i < MT * NT; i++) {
    free(a_Tile[i]);
    free(b_Tile[i]);
  }
  free(out);
  free(a);
  free(b);
  MPI_Finalize();


  return 0;
}


void test_matrix() {

  int i, j;
  int m = 10;
  int n = 10;
  int p = 2;
  int q = 2;
  int dim[2] = {p,q};

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      int proc = j%dim[1]+(i*dim[1])%(dim[0]*dim[1]);
      printf("%d ", proc);
    }
    printf("\n");
  }
}
