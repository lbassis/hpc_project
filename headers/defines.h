#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl_cblas.h>

#define ESP 1.11e-16

//#define BLOC_SIZE 130
#define BLOC_SIZE TILE_SIZE
#define TILE_SIZE 130

#define START 101 // MPI scatter and gather tag
