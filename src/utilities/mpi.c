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
  int n_out = nb_bloc_n / dims[1] + (nb_bloc_n % dims[1] > me % dims[1]);
  int m_out = nb_bloc_m / dims[0] + (nb_bloc_m % dims[0] > me % dims[0]);

  return (double**) malloc(n_out * m_out * sizeof(double));
}


void free_dist_tile_matrix(int m, int n, double** a, int dims[]) {

  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE;
  int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE;
  int n_out = nb_bloc_n / dims[1] + (nb_bloc_n % dims[1] > me % dims[1]);
  int m_out = nb_bloc_m / dims[0] + (nb_bloc_m % dims[0] > me % dims[0]);
  int i;
  for(i = 0; i < m_out * n_out; i++) free(a[i]);

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
  int n_out = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me % dim[1]);
  int m_out = nb_bloc_m / dim[0] + (nb_bloc_m % dim[0] > me % dim[0]);
  int i_out = 0, j_out = 0;
  int i,j;
  MPI_Status status;

  if(me == 0){
    printf("\nscattering %dx%d into %dx%d\n\n", m_out, n_out, nb_bloc_m, nb_bloc_n);

    for( i = 0; i < nb_bloc_m; i++) {
      for( j = 0; j < nb_bloc_n; j++) {
	      int proc = j%dim[1]+(i*dim[1])%(dim[0]*dim[1]);
        if(proc == 0){
          out[i_out + j_out * m_out] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
          LAPACKE_dlacpy( LAPACK_COL_MAJOR, 'A', TILE_SIZE, TILE_SIZE,
                               in[i + j * nb_bloc_m], TILE_SIZE,
                               out[i_out + j_out * m_out], TILE_SIZE);
	        printf("p0: (%d, %d)\n", i, j);
	        //affiche(TILE_SIZE, TILE_SIZE, out[i_out + j_out * m_out], TILE_SIZE, stdout);
          if(j_out == n_out - 1){
            j_out = 0;
            i_out++;
          }else{
            j_out++;
          }
        }else{
          printf("p%d: (%d, %d)\n", proc, i, j);
          MPI_Send(in[i + j * nb_bloc_m], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, proc, START, comm);
        }
      }
    }
  }else{

    for(i_out = 0; i_out < m_out; i_out++){
      for(j_out = 0; j_out < n_out; j_out++){
        out[i_out + j_out * m_out] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
        MPI_Recv(out[i_out + j_out * m_out], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, START, comm, &status);
        /* sleep(me+1); */
        /* printf("p%d:\n", me); */
        /* affiche(TILE_SIZE, TILE_SIZE, out[i_out+j_out*m_out], TILE_SIZE, stdout); */
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
  int n_out = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me % dim[1]);
  int m_out = nb_bloc_m / dim[0] + (nb_bloc_m % dim[0] > me % dim[0]);
  int i_out = 0, j_out = 0;
  int i,j;
  MPI_Status status;

  if(me == 0){
    printf("\ngathering %dx%d into %dx%d\n\n", m_out, n_out, nb_bloc_m, nb_bloc_n);
    for( i = 0; i < nb_bloc_m; i++) {
      for( j = 0; j < nb_bloc_n; j++) {
	      int proc = j%dim[1]+(i*dim[1])%(dim[0]*dim[1]);
	      //printf("i = %d, j = %d, proc = %d\n", i, j, proc);
        if(proc == 0){
          LAPACKE_dlacpy( LAPACK_COL_MAJOR, 'A', TILE_SIZE, TILE_SIZE,
                               in[i_out + j_out * m_out], TILE_SIZE,
                               out[i + j * nb_bloc_m], TILE_SIZE);
          free(in[i_out + j_out * m_out]);
          printf("\np0: (%d, %d)\n", i, j);
          //affiche(TILE_SIZE, TILE_SIZE, in[i_out + j_out * m_out], TILE_SIZE, stdout);

          if(j_out == n_out - 1){
            j_out = 0;
            i_out++;
          }else{
            j_out++;
          }
        }else{
          printf("%d <- %d: (%d, %d)\n", 0, proc, i, j);
          MPI_Recv(out[i + j * nb_bloc_m], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, proc, START, comm, &status);
        }
      }
    }
    printf("received everything\n");
  }else{
    printf("\np=%d %dx%d into %dx%d\n\n", me, m_out, n_out, nb_bloc_m, nb_bloc_n);

    for(i_out = 0; i_out < m_out; i_out++){
      for(j_out = 0; j_out < n_out; j_out++){
        //sleep(me+1);
        printf("%d -> %d: (%d, %d)\n", me, 0, i_out, j_out);
        MPI_Send(in[i_out + j_out * m_out], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, START, comm);
        free(in[i_out + j_out * m_out]);
	/* printf("p%d:\n", me); */
	/* affiche(TILE_SIZE, TILE_SIZE, out[i_out+j_out*m_out], TILE_SIZE, stdout); */
      }
    }
  }
}
