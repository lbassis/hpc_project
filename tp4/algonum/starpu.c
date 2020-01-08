#include <math.h>

#include "algonum.h"
#include "codelets.h"

void
my_starpu_init()
{
    int hres;
#if defined(ENABLE_MPI)
    {
        int flag = 0;
        MPI_Initialized( &flag );

        hres = starpu_mpi_init( NULL, NULL, !flag );
    }
#else
    {
        hres = starpu_init( NULL );
    }
#endif

#if defined(ENABLE_CUDA)
    starpu_cublas_init();
#endif

    (void)hres;
}

void
my_starpu_exit()
{
#if defined(ENABLE_CUDA)
    starpu_cublas_init();
#endif

#if defined(ENABLE_MPI)
    starpu_mpi_shutdown();
#else
    starpu_shutdown();
#endif
}

static inline int
get_starpu_rank()
{
#if defined(ENABLE_MPI)
    int rank;
    starpu_mpi_comm_rank( MPI_COMM_WORLD, &rank );
    return rank;
#else
    return 0;
#endif
}

int
get_starpu_owner( int m, int n )
{
#if defined(ENABLE_MPI)
    int nb_proc;//, me;
    //starpu_mpi_comm_rank(MPI_COMM_WORLD, &me);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &nb_proc);

    int p = sqrt(nb_proc);
    /*MPI_Comm comm_cart;
    int periods[2] = {1, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 0, &comm_cart);
    int new_me;
    MPI_Comm_rank(comm_cart, &new_me);
    int coords[2];
    MPI_Cart_coords(comm_cart, new_me, 2, coords);

    int line[2] = {0, 1};
    int column[2] = {1, 0};


    MPI_Comm comm_line;
    MPI_Comm comm_col;
    MPI_Cart_sub(comm_cart, line, &comm_line);
    MPI_Cart_sub(comm_cart, column, &comm_col);
    int me_line, me_col;
    MPI_Comm_rank(comm_line, &me_line);
    MPI_Comm_rank(comm_col, &me_col);*/
    return (m % p) * p + (n % p);
#else
    return 0;
#endif
    (void)m;
    (void)n;
}

starpu_data_handle_t
get_starpu_handle( int id, starpu_data_handle_t *handles, double **A, int m, int n, int b, int MT )
{
    starpu_data_handle_t *tile_handle = handles + n * MT + m;

    /* If the starpu_data_handle_t is NULL, we need to register the data */
    if ( *tile_handle == NULL ) {
        int home_node = -1;
        void *user_ptr = NULL;
        int myrank = get_starpu_rank();
        int owner  = get_starpu_owner( m, n );

        if ( myrank == owner ) {
            user_ptr = A[ MT * n + m ];
            if ( user_ptr != NULL ) {
                home_node = STARPU_MAIN_RAM;
            }
        }

        starpu_matrix_data_register( tile_handle, home_node, (uintptr_t) user_ptr,
                                     b, b, b, sizeof( double ) );

        starpu_data_set_coordinates( *tile_handle, 2, m, n );

#if defined(ENABLE_MPI)
        /**
         * We need to give a unique tag to each data
         * Be careful to take into account the multiple decriptors that can be used in parallel.
         */
        {
            int me;
            starpu_mpi_comm_rank(MPI_COMM_WORLD, &me);
            int64_t tag = ((int_64_t)id << 32) | ( n * MT + m );
            starpu_mpi_data_register( *tile_handle, tag, owner );
        }
#endif /* defined(CHAMELEON_USE_MPI) */
    }

    return *tile_handle;
}

starpu_data_handle_t
get_starpu_handle_lap( int id, starpu_data_handle_t *handle,
                       int i, int j, int m, int n, double *A, int lda, int MT )
{
    /* If the starpu_data_handle_t is NULL, we need to register the data */
    if ( *handle == NULL ) {
        int home_node = -1;
        void *user_ptr = NULL;
        int myrank = get_starpu_rank();
        int owner = 0;

        if ( myrank == owner ) {
            user_ptr = A;
            if ( user_ptr != NULL ) {
                home_node = STARPU_MAIN_RAM;
            }
        }

        starpu_matrix_data_register( handle, home_node, (uintptr_t) user_ptr,
                                     lda, m, n, sizeof( double ) );

        starpu_data_set_coordinates( *handle, 2, i, j );

#if defined(ENABLE_MPI)
        /**
         * We need to give a unique tag to each data
         * Be careful to take into account the multiple decriptors that can be used in parallel.
         */
        {
            int64_t tag = ((int_64_t)id << 32) | ((int_64_t)me << 24) | ( j * MT + i );
            starpu_mpi_data_register( *handle, tag, owner );
        }
#endif /* defined(CHAMELEON_USE_MPI) */
    }

    return *handle;
}

void
unregister_starpu_handle( int nb, starpu_data_handle_t *handles )
{
    int i;
    for (i=0; i<nb; i++, handles++) {
        if (*handles != NULL) {
            starpu_data_unregister_submit( *handles );
        }
    }
}
