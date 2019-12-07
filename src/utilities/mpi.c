#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mkl.h>
#include <mpi.h>
#include "util.h"
#include "perf.h"


#define TILE_SIZE 3
#define START 101

double **alloc_dist_matrix(int m, int n, int dims[]) {

  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE;
  int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE;
  int n_out = nb_bloc_n / dims[1] + (nb_bloc_n % dims[1] > me);
  int m_out = nb_bloc_m / dims[0] + (nb_bloc_n % dims[0] > me);

  return (double**) malloc(n_out * m_out * sizeof(double));
}

void scatter_matrix(const int m,
                    const int n,
                    const double** in,
                    double** out,
                    const int nb_proc,
                    const int me,
                    const int dim[2],
                    const MPI_Comm comm){
  int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE;
  int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE;
  int n_out = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me);
  int m_out = nb_bloc_m / dim[0] + (nb_bloc_n % dim[0] > me);
  int i_out = 0, j_out = 0;
  int i,j;
  MPI_Status status;

  if(me == 0){
    printf("scattering\n");
    for( i = 0; i < nb_bloc_m; i++) {
      for( j = 0; j < nb_bloc_n; j++) {
	int proc = j%dim[1]+(i*dim[1])%(dim[0]*dim[1]);
        if(proc == 0){
          out[i_out + j_out * m_out] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
          LAPACKE_dlacpy( LAPACK_COL_MAJOR, 'A', TILE_SIZE, TILE_SIZE,
                               in[i + j * nb_bloc_m], TILE_SIZE,
                               out[i_out + j_out * m_out], TILE_SIZE);
	  /* printf("p0:\n"); */
	  /* affiche(TILE_SIZE, TILE_SIZE, out[i_out+j_out*m_out], TILE_SIZE, stdout); */
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
	sleep(me+1);
	printf("p%d:\n", me);
	affiche(TILE_SIZE, TILE_SIZE, out[i_out+j_out*m_out], TILE_SIZE, stdout);
      }
    }
  }
}

void gather_matrix(const int m,
                    const int n,
                    const double** in,
                    double** out,
                    const int nb_proc,
                    const int me,
                    const int dim[2],
                    const MPI_Comm comm){
  int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE;
  int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE;
  int n_out = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me);
  int m_out = nb_bloc_m / dim[0] + (nb_bloc_n % dim[0] > me);
  int i_local = 0, j_local = 0;
  int i,j;
  MPI_Status status;
  if(me == 0){
    printf("gathering\n");
    for( i = 0; i < nb_bloc_m; i++) {
      for( j = 0; j < nb_bloc_n; j++) {
	int proc = j%dim[1]+(i*dim[1])%(dim[0]*dim[1]);
       if(proc == 0){
          LAPACKE_dlacpy( LAPACK_COL_MAJOR, 'A', TILE_SIZE, TILE_SIZE,
			       in[i_local + j_local * m_out], TILE_SIZE,
			       out[i + j * nb_bloc_m], TILE_SIZE);
          if(j_local == n_out - 1){
            j_local = 0;
            i_local ++;
	    if (i_local == m_out) {
	      i_local = 0;
	    }
	  }
	  else{
            j_local++;
          }
        }else{
          MPI_Recv(out[i + j * nb_bloc_m], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, proc, proc, comm, &status);
        }
      }
    }
  }else{

    for(i_local = 0; i_local < m_out; i_local++){
      for(j_local = 0; j_local < n_out; j_local++){
        MPI_Send(in[i_local + j_local * m_out], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, me, comm);
      }
    }
  }
}
