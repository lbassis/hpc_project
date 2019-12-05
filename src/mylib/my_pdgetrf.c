/*#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>


int my_pdgetrf(){
  MPI_Init(NULL, NULL);

  int me = -1, N = -1, tag = 12345;
  double A[MAX][MAX] = {};
  double B[MAX][MAX] = {};
  double C[MAX][MAX] = {};

  double tmp_A[MAX][MAX] = {};
  double tmp_B[MAX][MAX] = {};

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  MPI_Status status = {};
  MPI_Comm comm_cart;
  int nb_row = sqrt(N);
  int dims[2] = {nb_row, nb_row};
  int periods[2] = {};
  int coords[2] = {};
  int new_me = -1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);
  MPI_Comm_rank(comm_cart, &new_me);
  MPI_Cart_coords(comm_cart, new_me, 2, coords);

  int line[2] = {1, 0};

  MPI_Comm comm_line;
  MPI_Cart_sub(comm_cart, line, &comm_line);
  //printf("%d -> %d -> (%d, %d)\n", me, new_me, coords[0], coords[1]);

  int i = 0;
  for(i = 0; i < nb_row; i++){
    copy(tmp_A, A);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(A, MAX * MAX, MPI_DOUBLE, (coords[1] + i) % nb_row, comm_line);
    gemm(A, B, C);
    copy(A, tmp_A);
    int next = me + 1;
    int prev = me - 1;
    if(me == 0){
      prev = N - 1;
    }
    if(me == N - 1){
      next = 0;
    }
    MPI_Bsend(B, MAX * MAX, MPI_DOUBLE, next, tag, MPI_COMM_WORLD);
    MPI_Recv(tmp_B, MAX * MAX, MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    copy(B, tmp_B);
  }


  MPI_Finalize();

  return 0;
}
*/
