#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#include "defines.h"
#include "util.h"
#include "my_lib.h"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  if(argc != 2){
    fprintf(stderr, "Need exacly one argument, the size of the matrix\n");
    return 1;
  }

  int m = atoi(argv[1]);
  int nb_proc, me;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  int dims[2] = {sqrt(nb_proc), sqrt(nb_proc)};
  //printf("p: %d, (%d, %d)\n", me, dims[0], dims[1]);

  double* a = alloc_mat(m, m);
  double* b = alloc_mat(m, m);


  init_random(m, m, a, 1);
  int i;

  double **a_Tile = lapack2tile( m, m, TILE_SIZE, a, m );
  double **b_Tile = lapack2tile( m, m, TILE_SIZE, b, m );
  double **out = alloc_dist_matrix(m, m, dims);

  if(me == 0){
    printf("op, imp, us\n");
    my_dgetrf_openmp(LAPACK_COL_MAJOR, m, m, a, m, NULL);
    init_random(m, m, a, 1);
    //my_dgetrf_seq(LAPACK_COL_MAJOR, m, m, a, m, NULL);
    //init_random(m, m, a, 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  scatter_matrix(m, m, (const double**)a_Tile, out, nb_proc, me, dims, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  my_pdgetrf_tiled(LAPACK_COL_MAJOR, m, m, out, m, NULL, dims);

  gather_matrix(m, m, out, b_Tile, nb_proc, me, dims, MPI_COMM_WORLD);



  int MT = (m + TILE_SIZE - 1) / TILE_SIZE;
  int NT = (m + TILE_SIZE - 1) / TILE_SIZE;

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
