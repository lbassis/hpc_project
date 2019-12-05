#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mkl.h>
#include <mpi.h>
#include "util.h"
#include "perf.h"


#define TILE_SIZE 3
#define START 101

void scatter_matrix(const int m,
                    const int n,
                    const double** in,
                    double*** out,
                    const int nb_proc,
                    const int me,
                    const int dim[2],
                    const MPI_Comm comm){
  int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE;
  int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE;
  int n_out = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me);
  int m_out = nb_bloc_m / dim[0] + (nb_bloc_n % dim[0] > me);
  *out = (double**) malloc(n_out * m_out * sizeof(double));
  int i_out = 0, j_out = 0;
  int i,j;
  MPI_Status status;

  /* printf("[%d] m_out = %d, n_out = %d\n", me, m_out, n_out); */
  if(me == 0){
    /* printf("nb_bloc_m = %d, nb_bloc_n = %d, dim[0] = %d, dim[1] = %d\n", nb_bloc_m, nb_bloc_n, dim[0], dim[1]); */
    for( i = 0; i < nb_bloc_m; i++) {
      for( j = 0; j < nb_bloc_n; j++) {
        int proc = (i%nb_bloc_m) + nb_bloc_m*(j%nb_bloc_n);
        if(proc == 0){
          out[i_out + j_out * m_out] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
          LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', TILE_SIZE, TILE_SIZE,
                               in[i + j * nb_bloc_m], TILE_SIZE,
                               out[i_out + j_out * m_out], TILE_SIZE);
          if(j_out == n_out - 1){
            j_out = 0;
            i_out++;
            if(i_out == m_out - 1){
              i_out = 0;
            }
          }else{
            j_out++;
          }
        }else{
          MPI_Send(in[i + j * nb_bloc_m], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, proc, START, comm);
        }
      }
    }
  }else{

    for(i_out = 0; i_out < m_out; i_out++){
      for(j_out = 0; j_out < n_out; j_out++){
        out[i_out + j_out * m_out] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
        MPI_Recv(out[i_out + j_out * m_out], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, START, comm, &status);
      }
    }

  }

  sleep(me*3+1);
  for(i_out = 0; i_out < m_out; i_out++){
    for(j_out = 0; j_out < n_out; j_out++){
      affiche(TILE_SIZE, TILE_SIZE, out[i_out + j_out * m_out], TILE_SIZE, stdout);
    }
  }
  printf("[%d]ok\n", me);
}
