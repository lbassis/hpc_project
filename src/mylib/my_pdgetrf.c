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


#ifdef PERF
#include "perf.h"
#endif


void my_pdgetrf_tiled(CBLAS_LAYOUT layout,
		      const int m,
		           const int n,
		           double** a,
	             const int lda,
		           int* ipiv,
               const int dim[2]){

  assert(layout == CblasColMajor);
	(void) ipiv;

  #ifdef PERF
  perf_t start = {}, stop = {}, total_getf2 = {}, total_dtrsm = {}, total_gemm = {};
  perf_t total_comm = {}, malloc_free = {};
	#endif


  int nb_proc, me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

  MPI_Comm comm_cart;
  int periods[2] = {1, 1};
  //printf("size of grid (%d, %d)\n", dim[0], dim[1]);

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 0, &comm_cart);
  int new_me;
  MPI_Comm_rank(comm_cart, &new_me);
  int coords[2];
  MPI_Cart_coords(comm_cart, new_me, 2, coords);
  //printf("p: %d, cart: %d, coords: (%d, %d)\n", me, new_me, coords[0], coords[1]);
	//MPI_Barrier(MPI_COMM_WORLD);

  int line[2] = {0, 1};
  int column[2] = {1, 0};


  MPI_Comm comm_line;
  MPI_Comm comm_col;
  MPI_Cart_sub(comm_cart, line, &comm_line);
  MPI_Cart_sub(comm_cart, column, &comm_col);
  int me_line, me_col;
  MPI_Comm_rank(comm_line, &me_line);
  MPI_Comm_rank(comm_col, &me_col);

  int nb_bloc_n = (n + TILE_SIZE - 1) / TILE_SIZE;
  int nb_bloc_m = (m + TILE_SIZE - 1) / TILE_SIZE;
  int n_local = nb_bloc_n / dim[1] + (nb_bloc_n % dim[1] > me % dim[1]);
  int m_local = nb_bloc_m / dim[0] + (nb_bloc_m % dim[0] > ((me - (me % dim[1]))/dim[1])%dim[0]);
  int min = fmin(nb_bloc_n, nb_bloc_m);

  #ifdef PERF
  if(me == 0)
    perf(&start);
  #endif

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

  #ifdef PERF
  if(me == 0){
    perf(&stop);
    perf_diff(&start, &malloc_free);
  }
  #endif


  int k = 0;
  int k_local_m = 0, k_local_n = 0;
  for(k = 0; k < min; k++){

    #ifdef PERF
    if(me == 0)
		  perf(&start);
		#endif


    int proc = ((k*dim[0])%(dim[0]*dim[1])) + (k%dim[1]);
    int is_col_trsm = (me % dim[1] == k % dim[1]) || me == proc;
    int is_line_trsm = ((me - (me%dim[1]))/dim[1] == k % dim[0]) || me == proc;

    //printf("p%d: proc maintenant c'est %d (%d en ligne et %d en colonne)\n", me, proc, k%dim[1], k%dim[0]);
    if(proc == me){
      bloc_dgetf2 = a[k_local_m + k_local_n * m_local];
      my_dgetf2(CblasColMajor,
		(has_last_line && k_local_m == m_local - 1 && ((m % TILE_SIZE) != 0))? (m % TILE_SIZE) : TILE_SIZE,
		(has_last_col && k_local_n == n_local - 1 && ((n % TILE_SIZE) != 0))? (n % TILE_SIZE) : TILE_SIZE,
                bloc_dgetf2,
                TILE_SIZE,
                NULL);
    }else{
      bloc_dgetf2 = tmp_bloc_dgetf2;
    }

    #ifdef PERF
    if(me == 0){
      perf(&stop);
      perf_diff(&start, &stop);
      perf_add(&stop, &total_getf2);
      perf(&start);
    }
		#endif

    if(is_line_trsm){ // same line as k
      MPI_Bcast(bloc_dgetf2, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k%dim[1], comm_line);
    }

    if(is_col_trsm){ // same column as k
      MPI_Bcast(bloc_dgetf2, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k%dim[0], comm_col);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef PERF
    if(me == 0){
      perf(&stop);
      perf_diff(&start, &stop);
      perf_add(&stop, &total_comm);
      perf(&start);
    }
		#endif

    if(is_col_trsm){
      for(i = k_local_m + (proc == me); i < m_local; i++){
				//printf("p%d: col (%d, %d), size: %d\n", me, i, k_local_n, (has_last_line && i == m_local - 1)? (m % TILE_SIZE) : TILE_SIZE);
	       my_dtrsm(CblasColMajor,
                CblasRight,
                CblasUpper,
                CblasNoTrans,
                CblasNonUnit,
		            /* m */ (has_last_line && i == m_local - 1 && ((m % TILE_SIZE) != 0))? (m % TILE_SIZE) : TILE_SIZE,
                /* n */ TILE_SIZE,
                /* alpha */ 1,
                /* L\U */ bloc_dgetf2,
  	            /* lda */ TILE_SIZE,
                /* A[i][k] */ a[i + k_local_n * m_local],
                /* ldb */ TILE_SIZE);
	       col_trsm[i] = a[i + k_local_n * m_local];
      }
    }else{
      for(i = k_local_m; i < m_local; i++){
        col_trsm[i] = tmp_col_trsm[i];
      }
    }

		//sleep(me+1);
    if(is_line_trsm){
      for(j = k_local_n + (proc == me); j < n_local; j++){
				//printf("p%d: line (%d, %d), size: %d\n", me, k_local_m, j, (has_last_col && j == n_local - 1)? (n % TILE_SIZE) : TILE_SIZE);
	       my_dtrsm(/*int *Layout*/ CblasColMajor,
                  /*int side*/      CblasLeft,
                  /*int uplo*/      CblasLower,
                  /*int transA*/    CblasNoTrans,
                  /*int diag*/      CblasUnit,
                  /*int m*/         TILE_SIZE,
		              /*int n*/         (has_last_col && j == n_local - 1 && ((n % TILE_SIZE) != 0))? (n % TILE_SIZE) : TILE_SIZE,
                  /*double alpha*/  1,
                  /*double *a*/     bloc_dgetf2,
                  /*int lda*/       TILE_SIZE,
                  /*double *b*/     a[k_local_m + j * m_local],
                  /*int ldb*/       TILE_SIZE);
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

    #ifdef PERF
    if(me == 0){
      perf(&stop);
      perf_diff(&start, &stop);
      perf_add(&stop, &total_dtrsm);
      perf(&start);
    }
		#endif

    for(i = k_local_m; i < m_local; i++){
      MPI_Bcast(col_trsm[i], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k % dim[0], comm_line);//comm_line
    }

    for(j = k_local_n; j < n_local; j++){
      MPI_Bcast(line_trsm[j], TILE_SIZE * TILE_SIZE, MPI_DOUBLE, k % dim[1], comm_col);//comm_col
    }
    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef PERF
    if(me == 0){
      perf(&stop);
      perf_diff(&start, &stop);
      perf_add(&stop, &total_comm);
      perf(&start);
    }
		#endif

    for(i = k_local_m; i < m_local; i++){
      for(j = k_local_n; j < n_local; j++){
        my_dgemm_bloc (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (has_last_line && i == m_local - 1 && ((m % TILE_SIZE) != 0))? (m % TILE_SIZE) : TILE_SIZE,
                     /* n */ (has_last_col && j == n_local - 1 && ((n % TILE_SIZE) != 0))? (n % TILE_SIZE) : TILE_SIZE,
                     /* k */ TILE_SIZE,
                     /* alpha */ -1.,
                     /* A[i][k] */ col_trsm[i],
  	                   /* lda */ TILE_SIZE,
  		                 /* B[k][j] */ line_trsm[j],
  		                 /* ldb */ TILE_SIZE,
                     /* beta */ 1.,
  		                 /* C[i][j] */ a[i + j * m_local],
  		                 /* ldc */ TILE_SIZE);
      }
    }
    #ifdef PERF
    if(me == 0){
      perf(&stop);
      perf_diff(&start, &stop);
      perf_add(&stop, &total_gemm);
      perf(&start);
    }
		#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  #ifdef PERF
  if(me == 0){
    perf(&start);
  }
  #endif
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


  #ifdef PERF
  if(me == 0){
    perf(&stop);
    perf_diff(&start, &stop);
    perf_add(&stop, &malloc_free);
    printf("getf2, mpi, ");
    perf_print_time(&total_getf2, 1);
    printf("\n");
    printf("trsm, mpi, ");
    perf_print_time(&total_dtrsm, 1);
    printf("\n");
    printf("gemm, mpi, ");
    perf_print_time(&total_gemm, 1);
    printf("\n");
    printf("communication, mpi, ");
    perf_print_time(&total_comm, 1);
    printf("\n");
    printf("malloc+free, mpi, ");
    perf_print_time(&total_comm, 1);
    printf("\n");
  }
  #endif
}
