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
#include "defines.h"




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
		           int* ipiv,
               const int dim[2]){

  assert(layout == CblasColMajor);
	(void) ipiv;

  int nb_proc, me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

  //MPI_Status status = {};
  MPI_Comm comm_cart;
  //int r = find_r(nb_proc); // r closest integer to sqrt(nb_proc) (r < sqrt(nb_proc)) that also divide nb_proc
  //int dim[2] = {r, nb_proc / r};// /!\ même dimentionqur ppoiuter le scatterz
  int periods[2] = {1, 1};
  printf("size of grid (%d, %d)\n", dim[0], dim[1]);

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 0, &comm_cart);
  int new_me;
  MPI_Comm_rank(comm_cart, &new_me);
  int coords[2];
  MPI_Cart_coords(comm_cart, new_me, 2, coords);
  printf("p: %d, cart: %d, coords: (%d, %d)\n", me, new_me, coords[0], coords[1]);
  int line[2] = {0, 1};
  int column[2] = {1, 0};


  MPI_Comm comm_line;
  MPI_Comm comm_col;
  MPI_Cart_sub(comm_cart, line, &comm_line);
  MPI_Cart_sub(comm_cart, column, &comm_col);
  int me_line, me_col;
  MPI_Comm_rank(comm_line, &me_line);
  MPI_Comm_rank(comm_col, &me_col);

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int n_local = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me % dim[1]);
  int m_local = nb_bloc_m / dim[0] + (nb_bloc_m % dim[0] > ((me - (me % dim[1]))/dim[1])%dim[0]);
  int min = fmin(nb_bloc_n, nb_bloc_m);
  double* bloc_dgetf2;
  double* tmp_bloc_dgetf2 = (double*) calloc(TILE_SIZE * TILE_SIZE,  sizeof(double));
  double** line_trsm = (double**) malloc(n_local * sizeof(double));
  double** col_trsm = (double**) malloc(m_local * sizeof(double));
  double** tmp_line_trsm = (double**) malloc(n_local * sizeof(double));
  double** tmp_col_trsm = (double**) malloc(m_local * sizeof(double));

  int has_last_col = ((me_line-1+dim[1])%dim[1] == n % dim[1]);
  int has_last_line = ((me_col-1+dim[0])%dim[0] == m % dim[0]);

  //printf("p%d has_last_line:%d, has_last_col:%d\n", me, has_last_line, has_last_col);
  int i = 0;
  int j = 0;
  for(i = 0; i < m_local; i++){
    tmp_col_trsm[i] = (double*) calloc(TILE_SIZE * TILE_SIZE, sizeof(double));
  }

  for(j = 0; j < n_local; j++){
    tmp_line_trsm[j] = (double*) calloc(TILE_SIZE * TILE_SIZE, sizeof(double));
  }

  int k = 0;
  int k_local_m = 0, k_local_n = 0;
  for(k = 0; k < min; k++){

    if (me == 0) {
      printf("premier bloc en k = %d:\n", k);
      affiche(TILE_SIZE, TILE_SIZE, a[0], TILE_SIZE, stdout);
    }
    int proc = ((k*dim[0])%(dim[0]*dim[1])) + (k%dim[1]);
    int is_col_trsm = (me % dim[1] == k % dim[1]) || me == proc;
    int is_line_trsm = ((me - (me%dim[1]))/dim[1] == k % dim[0]) || me == proc;

    //printf("p%d: proc maintenant c'est %d (%d en ligne et %d en colonne)\n", me, proc, k%dim[1], k%dim[0]);
    if(proc == me){
      bloc_dgetf2 = a[k_local_m + k_local_n * m_local];
      my_dgetf2(CblasColMajor,
		(has_last_line && k_local_m == m_local - 1)? (m % TILE_SIZE) : TILE_SIZE,
		(has_last_col && k_local_n == n_local - 1)? (n % TILE_SIZE) : TILE_SIZE,
                bloc_dgetf2,
                BLOC_SIZE,
                NULL);
    }else{
      bloc_dgetf2 = tmp_bloc_dgetf2;
    }

    if(is_line_trsm){ // same line as k
      MPI_Bcast(bloc_dgetf2, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k%dim[1], comm_line);
    }

    if(is_col_trsm){ // same column as k
      MPI_Bcast(bloc_dgetf2, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k%dim[0], comm_col);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //sleep(me*2);
    printf("p%d a recu le getf2\n", me);
    affiche(TILE_SIZE, TILE_SIZE, bloc_dgetf2, TILE_SIZE, stdout);

    if(is_col_trsm){
      printf("pointer: %p\n",  a[i + k_local_n * m_local]);
      for(i = k_local_m + (proc == me); i < m_local; i++){
	printf("p%d va faire son trsm\n", me);
	printf("142 change %d\n", i + k_local_n * m_local);
        my_dtrsm(CblasColMajor,
                CblasRight,
                CblasUpper,
                CblasNoTrans,
                CblasNonUnit,
		 /* m */ (has_last_line && i == m_local - 1)? (m % TILE_SIZE) : TILE_SIZE,
                /* n */ BLOC_SIZE,
                /* alpha */ 1,
                /* L\U */ bloc_dgetf2,
  	               /* lda */ BLOC_SIZE,
                 /* A[i][k] */ a[i + k_local_n * m_local],
                 /* ldb */ BLOC_SIZE);
	printf("p%d a fait son trsm\n", me);
        col_trsm[i] = a[i + k_local_n * m_local];
      }
    }else{

      for(i = k_local_m; i < m_local; i++){
        col_trsm[i] = tmp_col_trsm[i];
      }
    }


    if(is_line_trsm){
      printf("p%d est line trsm! j = %d < %d\n", me, k_local_n + (proc == me), n_local);
      for(j = k_local_n + (proc == me); j < n_local; j++){
	printf("p%d: j = %d, taille_differente = %d, n_local = %d, result = %d\n", me, j, (has_last_col && j == n_local - 1), n_local, (has_last_col && j == n_local - 1)? (n % TILE_SIZE) : TILE_SIZE);
	my_dtrsm(/*int *Layout*/ CblasColMajor,
                  /*int side*/      CblasLeft,
                  /*int uplo*/      CblasLower,
                  /*int transA*/    CblasNoTrans,
                  /*int diag*/      CblasUnit,
                  /*int m*/         BLOC_SIZE,
		 /*int n*/          (has_last_col && j == n_local - 1)? (n % TILE_SIZE) : TILE_SIZE,
                  /*double alpha*/  1,
                  /*double *a*/     bloc_dgetf2,
                  /*int lda*/       BLOC_SIZE,
                  /*double *b*/     a[k_local_m + j * m_local],
                  /*int ldb*/       BLOC_SIZE);
        line_trsm[j] = a[k_local_m + j * m_local];
      }
    }else{

        for(j = k_local_n; j < n_local; j++){
          line_trsm[j] = tmp_line_trsm[j];
        }
    }
    if (is_line_trsm) {
      k_local_m++;
    }
    if (is_col_trsm) {
      k_local_n++;
    }
    for(i = k_local_m; i < m_local; i++){
      MPI_Bcast(col_trsm[i], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k % dim[0], comm_line);//comm_line
    }

    for(j = k_local_n; j < n_local; j++){
      MPI_Bcast(line_trsm[j], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k % dim[1], comm_col);//comm_col
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(i = k_local_m; i < m_local; i++){
      for(j = k_local_n; j < n_local; j++){
        my_dgemm_bloc (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (has_last_line && i == m_local - 1)? (m % TILE_SIZE) : TILE_SIZE,
                     /* n */ (has_last_col && j == n_local - 1)? (n % TILE_SIZE) : TILE_SIZE,
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
    printf("coucou\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_free(&comm_line);
  MPI_Comm_free(&comm_col);
  MPI_Comm_free(&comm_cart);
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
