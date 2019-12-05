#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mkl.h>
#include <mpi.h>
#include "util.h"
#include "perf.h"



int main(void){
  printf("%s: \n", __FILE__);

  MPI_Init(NULL, NULL);

  int m = 5;
  int n = 5;
  int lda = m;
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};
  int dims[2] = {2, 2};

  int nb_proc, me;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  //MPI_Comm comm_cart;
  //int nb_row = sqrt(N);
  //int periods[2] = {};
  //int coords[2] = {};
  //int new_me = -1;

  //MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);
  //MPI_Comm_rank(comm_cart, &new_me);
  //MPI_Cart_coords(comm_cart, new_me, 2, coords);

  //int line[2] = {1, 0};

  //MPI_Comm comm_line;
  //MPI_Cart_sub(comm_cart, line, &comm_line);


  double* a = alloc_mat(lda, n);
  double* b = alloc_mat(lda, n);

  LAPACKE_dlarnv_work(IONE, ISEED, n * lda, a);

  double **a_Tile = lapack2tile( lda, n, 3, a, lda );
  double **b_Tile = lapack2tile( lda, n, 3, b, lda );
  double **out = NULL;

  if (me == 0)
    affiche(m, n, a, m, stdout);

  printf("____\n");
  scatter_matrix(m, n, a_Tile, &out, nb_proc, me, dims, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  printf("____\n");
  gather_matrix(m, n, out, b_Tile, nb_proc, me, dims, MPI_COMM_WORLD);

  tile2lapack( m, n, 3, b_Tile, b, lda );

  printf("____\n");
  if (me == 0)
    affiche(m, n, b, m, stdout);

  MPI_Finalize();
  return 0;
}
