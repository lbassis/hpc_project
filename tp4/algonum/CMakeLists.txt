###
#
#  @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                       Univ. Bordeaux. All rights reserved.
#
#  @author Mathieu Faverge
#  @date 2019-11-13
#
###
cmake_minimum_required (VERSION 2.8)

# Disable in source directory build
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )

# use, i.e. don't skip the full RPATH for the build tree
set( CMAKE_SKIP_BUILD_RPATH FALSE )

# when building, don't use the install RPATH already
# (but later on when installing)
set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )

option(BUILD_SHARED_LIBS
  "Build shared libraries" ON)

include_directories( include )

set( algonum_srcs
  core_dplrnt.c
  tile.c
  perf.c
  test-gemm.c
  test-getrf.c
  )

if( ENABLE_STARPU )
  list( APPEND algonum_srcs
    codelet_dplrnt.c
    codelet_dlacpy.c
    starpu.c
    starpu_dplrnt.c
    starpu_dlacpy.c
    )
endif()

add_library( algonum
  ${algonum_srcs}
  )

# Configuration with MKL
set( LAPACKE_LIBRARY_DIRS_DEP
  $ENV{MKLROOT}/lib/intel64
  )

set( LAPACKE_LIBRARIES_DEP
  mkl_gf_lp64
  mkl_sequential
  mkl_core
)

set( DEPS_LIBS
  ${LAPACKE_LIBRARIES_DEP}
  m
  )

if( ENABLE_STARPU )
  list( APPEND DEPS_LIBS
    ${STARPU_LIBRARIES}
    )
endif()

link_directories( ${LAPACKE_LIBRARY_DIRS_DEP} )
target_link_libraries( algonum
  ${DEPS_LIBS}
  )

install(TARGETS algonum
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES
  include/algonum.h
  include/cblas.h
  include/flops.h
  include/lapacke.h
  include/perf.h
  DESTINATION include )

