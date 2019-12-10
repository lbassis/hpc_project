#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mkl.h>
#include <mpi.h>
#include "util.h"
#include "perf.h"
#include "defines.h"


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
  init_random(lda, n, a, 1);
  int i, j;

  double **a_Tile = lapack2tile( lda, n, 3, a, lda );
  double **b_Tile = lapack2tile( lda, n, 3, b, lda );
  double **out = alloc_dist_matrix(m, n, dims);

  if (me == 0)
    affiche(m, n, a, m, stdout);

  scatter_matrix(m, n, a_Tile, out, nb_proc, me, dims, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  my_dgetrf_seq(LAPACK_COL_MAJOR, m, n, a, m, NULL);
  my_pdgetrf_tiled(LAPACK_COL_MAJOR, m, n, out, m, NULL, dims);

  /* int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE; */
  /* int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE; */
  /* int n_out = nb_bloc_n / dims[1] + (nb_bloc_n % dims[1] > me % dims[1]); */
  /* int m_out = nb_bloc_m / dims[0] + (nb_bloc_m % dims[0] > me % dims[0]); */

  /* printf("getrf d'avant\n"); */
  /* for (i = 0; i < nb_bloc_m; i++) { */
  /*   for (j = 0; j < nb_bloc_n; j++) { */
  /*     affiche(TILE_SIZE, TILE_SIZE, a_Tile[i + j*nb_bloc_m], TILE_SIZE, stdout); */
  /*   } */
  /* } */

  /* sleep(me*2); */
  /* printf("p%d:\n", me); */
  /* for (i = 0; i < m_out; i++) { */
  /*   for (j = 0; j < n_out; j++) { */
  /*     affiche(TILE_SIZE, TILE_SIZE, out[i + j*m_out], TILE_SIZE, stdout); */
  /*   } */
  /* } */

  //if (me == 0)
  //affiche(TILE_SIZE, TILE_SIZE, out[0], TILE_SIZE, stdout);
  gather_matrix(m, n, out, b_Tile, nb_proc, me, dims, MPI_COMM_WORLD);

  tile2lapack( m, n, 3, b_Tile, b, lda );
  //tile2lapack( m, n, 3, a_Tile, a, lda );

  if (me == 0){
    printf("____\n");
    affiche(m, n, b, m, stdout);
    printf("correct:\n");
    affiche(m, n, a, m, stdout);
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
