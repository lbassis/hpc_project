#include "algonum.h"
#include "codelets.h"



void
my_dgetrf_tiled_starpu( CBLAS_LAYOUT layout,
                        int M, int N, int b, double **A )
{

    int BLOC_SIZE = b;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int m = M;
    int n = N;

    int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
    int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
    int min = fmin(nb_bloc_n, nb_bloc_m);
    starpu_data_handle_t *bloc_dgetf2;
    starpu_data_handle_t *bloc_dtrsm;
    starpu_data_handle_t *handlesA;
    starpu_data_handle_t tf2, trsm, hAij;

    bloc_dgetf2 = calloc(nb_bloc_n * nb_bloc_m, sizeof(starpu_data_handle_t));
    bloc_dtrsm = calloc(nb_bloc_n * nb_bloc_m, sizeof(starpu_data_handle_t));
    handlesA = calloc(nb_bloc_n * nb_bloc_m, sizeof(starpu_data_handle_t) );

    int k = 0;
    for(k = 0; k < min; k++){
      //bloc_dgetf2 = a[k * (nb_bloc_m+1)];
      tf2 = get_starpu_handle( 0, bloc_dgetf2, A, k, k, b, MT );

      insert_dgetf2((k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
                    (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
                    tf2,
                    BLOC_SIZE);

      int i = 0;

      for(i = k + 1; i < nb_bloc_m; i++){
        trsm = get_starpu_handle( 1, bloc_dtrsm, A, i, k, b, MT );
        insert_dtrsm(CblasRight,
                 CblasUpper,
                 CblasNoTrans,
                 CblasNonUnit,
                 /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                 /* n */ BLOC_SIZE,
                 /* alpha */ 1,
                 /* L\U */ tf2,
  	             /* lda */ BLOC_SIZE,
                 /* A[i][k] */ trsm,
                 /* ldb */ BLOC_SIZE);
      }
      int j = 0;
      for(j = k + 1; j < nb_bloc_n; j++){
        trsm = get_starpu_handle( 2, bloc_dtrsm, A, k, j, b, MT );
        insert_dtrsm(/*int side*/      CblasLeft,
                  /*int uplo*/      CblasLower,
                  /*int transA*/    CblasNoTrans,
                  /*int diag*/      CblasUnit,
                  /*int m*/         BLOC_SIZE,
                  /*int n*/         (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                  /*double alpha*/  1,
                  /*double *a*/     tf2,
                  /*int lda*/       BLOC_SIZE,
  	              /*double *b*/     trsm,
                  /*int ldb*/       BLOC_SIZE);
      }
      for(i = k + 1; i < nb_bloc_m; i++){
        for(j = k + 1; j < nb_bloc_n; j++){
          tf2 = get_starpu_handle( 3, bloc_dgetf2, A, i, k, b, MT );
          trsm = get_starpu_handle( 4, bloc_dtrsm, A, k, j, b, MT );
          hAij = get_starpu_handle( 5, handlesA, A, i, j, b, MT );
          insert_dgemm (CblasNoTrans,
                       CblasNoTrans,
                       /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                       /* n */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                       /* k */ BLOC_SIZE,
                       /* alpha */ -1.,
                       /* A[i][k] */ tf2,
		                   /* lda */ BLOC_SIZE,
		                   /* B[k][j] */ trsm,
		                   /* ldb */ BLOC_SIZE,
                       /* beta */ 1.,
		                   /* C[i][j] */ hAij,
		                   /* ldc */ BLOC_SIZE);
        }
      }
    }

    unregister_starpu_handle( nb_bloc_n * nb_bloc_m, handlesA );
    unregister_starpu_handle( nb_bloc_n * nb_bloc_m, bloc_dtrsm );
    unregister_starpu_handle( nb_bloc_n * nb_bloc_m, bloc_dgetf2 );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesA );
    free( bloc_dtrsm );
    free( bloc_dgetf2 );
}
