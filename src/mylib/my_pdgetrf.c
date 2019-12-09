#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <mkl.h>
#include <assert.h>
#include "my_blas.h"
#include "my_lapack.h"
#include "my_dgemm.h"
#include "my_dgetrf.h"

#define BLOC_SIZE TILE_SIZE
#define TILE_SIZE 130

static int find_r(const int n){
  assert(n > -1);
  int r = sqrt(n);
  while(n % r != 0) r--;
  return r;
}

void my_pdgetrf_tiled(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double** a,
	             const int lda,
		           int* ipiv){

  assert(layout == CblasColMajor);
	(void) ipiv;

  int nb_proc, me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

  //MPI_Status status = {};
  MPI_Comm comm_cart;
  int r = find_r(nb_proc); // r closest integer to sqrt(nb_proc) (r < sqrt(nb_proc)) that also divide nb_proc
  int dim[2] = {r, nb_proc / r};
  int periods[2] = {1, 1};

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 1, &comm_cart);


  int line[2] = {1, 0};
  int column[2] = {0, 1};

  MPI_Comm comm_line;
  MPI_Comm comm_col;
  MPI_Cart_sub(comm_cart, line, &comm_line);
  MPI_Cart_sub(comm_cart, column, &comm_col);


  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int n_local = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me);
  int m_local = nb_bloc_m / dim[0] + (nb_bloc_n % dim[0] > me);

  int min = fmin(nb_bloc_n, nb_bloc_m);
  double* bloc_dgetf2;
  double* tmp_bloc_dgetf2 = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
  double** line_trsm = (double**) malloc(n_local * sizeof(double));
  double** col_trsm = (double**) malloc(m_local * sizeof(double));
  double** tmp_line_trsm = (double**) malloc(n_local * sizeof(double));
  double** tmp_col_trsm = (double**) malloc(m_local * sizeof(double));
  int i = 0;
  int j = 0;
  for(i = 0; i < m_local; i++){
    tmp_col_trsm[i] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
  }

  for(j = 0; j < n_local; j++){
    tmp_line_trsm[j] = (double*) malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
  }


  int k = 0;
  int k_local = 0;
  for(k = 0; k < min; k++){


    int proc = ((k*dim[0])%(dim[0]*dim[1])) + (k%dim[1]);
    if(proc == me){
      bloc_dgetf2 = a[k_local * (m_local+1)];
      my_dgetf2(CblasColMajor,
                (k_local * dim[0] < m - 1) ? BLOC_SIZE : m - k_local * dim[0] * BLOC_SIZE,
                (k_local * dim[1] < n - 1) ? BLOC_SIZE : n - k_local * dim[1] * BLOC_SIZE,
                bloc_dgetf2,
                BLOC_SIZE,
                NULL);
    }else{
      bloc_dgetf2 = tmp_bloc_dgetf2;
    }

    int is_line_trsm = (me % dim[1] == k % dim[1]);
    int is_col_trsm = ((me - (me%dim[1])) % (dim[1] * dim[0]) == (k * dim[1]) % (dim[1] * dim[0]));

    MPI_Barrier(MPI_COMM_WORLD);
    if(is_line_trsm){ // same line as k
      MPI_Bcast(bloc_dgetf2, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, proc, comm_col);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(is_col_trsm){ // same column as k
      MPI_Bcast(bloc_dgetf2, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, proc, comm_line);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(is_col_trsm){
      for(i = k_local; i < m_local; i++){
        my_dtrsm(CblasColMajor,
                CblasRight,
                CblasUpper,
                CblasNoTrans,
                CblasNonUnit,
                /* m */ (i * dim[0] < m - 1) ? BLOC_SIZE : m - i * dim[0] * BLOC_SIZE,
                /* n */ BLOC_SIZE,
                /* alpha */ 1,
                /* L\U */ bloc_dgetf2,
	               /* lda */ BLOC_SIZE,
                 /* A[i][k] */ a[i + k_local * m_local],
                 /* ldb */ BLOC_SIZE);

        col_trsm[j] = a[k_local + j * m_local];
      }
    }else{
      for(i = k_local; i < m_local; i++){
        col_trsm[i] = tmp_col_trsm[i];
      }
    }


    if(is_line_trsm){
      for(j = k_local; j < n_local; j++){
        my_dtrsm(/*int *Layout*/ CblasColMajor,
                  /*int side*/      CblasLeft,
                  /*int uplo*/      CblasLower,
                  /*int transA*/    CblasNoTrans,
                  /*int diag*/      CblasUnit,
                  /*int m*/         BLOC_SIZE,
                  /*int n*/         (j * dim[1] < n - 1) ? BLOC_SIZE : n - j * dim[1] * BLOC_SIZE,
                  /*double alpha*/  1,
                  /*double *a*/     bloc_dgetf2,
                  /*int lda*/       BLOC_SIZE,
                  /*double *b*/     a[k_local + j * m_local],
                  /*int ldb*/       BLOC_SIZE);

        line_trsm[j] = a[k_local + j * m_local];
      }
    }else{
        for(j = k_local; j < n_local; j++){
          line_trsm[j] = tmp_line_trsm[j];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);



    for(i = k_local; i < m_local; i++){
      MPI_Bcast(col_trsm[i], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, (me - (me%dim[1])) % (dim[1] * dim[0]), comm_line);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(j = k_local; j < n_local; j++){
      MPI_Bcast(line_trsm[i], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, me % dim[1], comm_col);
    }


    for(i = k_local; i < m_local; i++){
      for(j = k_local; j < n_local; j++){
        my_dgemm_bloc (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (i * dim[0] < m - 1) ? BLOC_SIZE : m - i * dim[0] * BLOC_SIZE,
                     /* n */ (j * dim[1] < n - 1) ? BLOC_SIZE : n - j * dim[1] * BLOC_SIZE,
                     /* k */ BLOC_SIZE,
                     /* alpha */ -1.,
                     /* A[i][k] */ col_trsm[i],
	                   /* lda */ BLOC_SIZE,
		                 /* B[k][j] */ line_trsm[j],
		                 /* ldb */ BLOC_SIZE,
                     /* beta */ 1.,
		                 /* C[i][j] */ a[i + j * m_local],
		                 /* ldc */ BLOC_SIZE);
      }
    }
    if(proc == me){
      k_local++;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for(i = 0; i < m_local; i++){
    free(tmp_col_trsm[i]);
  }

  for(j = 0; j < n_local; j++){
    free(tmp_line_trsm[j]);
  }
  free(tmp_col_trsm);
  free(tmp_line_trsm);
  free(col_trsm);
  free(line_trsm);
  free(tmp_bloc_dgetf2);

  (void)lda;
}
