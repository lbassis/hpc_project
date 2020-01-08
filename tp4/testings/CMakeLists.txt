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

if ( NOT DEFINED ALGONUM_DIR )
  set( ALGONUM_DIR ${CMAKE_SOURCE_DIR}/../install/ )
endif()

# Configuration with MKL
set( LAPACKE_LIBRARY_DIRS_DEP
  $ENV{MKLROOT}/lib/intel64
  )
set( LAPACKE_LIBRARIES_DEP
  mkl_gf_lp64
  mkl_sequential
  mkl_core
)

include_directories( ${ALGONUM_DIR}/include )

link_directories( ${LAPACKE_LIBRARY_DIRS_DEP} )
link_directories( ${ALGONUM_DIR}/lib )

set( DEPS_LIBS
  algonum
  myblas
  ${LAPACKE_LIBRARIES_DEP}
  m
  )

set(TESTINGS
  check_dgemm.c
  check_dgetrf.c
  perf_dgemm.c
  perf_dgetrf.c
  )

foreach( _test ${TESTINGS} )
  get_filename_component(_name_we ${_test} NAME_WE)
  add_executable(${_name_we} ${_test})
  target_link_libraries(${_name_we} ${DEPS_LIBS})
endforeach()

