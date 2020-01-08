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

include_directories( ${CMAKE_SOURCE_DIR}/algonum/include )

option( ENABLE_STARPU "Define if StarPU support is enabled or not" OFF )
if (ENABLE_STARPU)
  add_definitions(-DENABLE_STARPU)
endif()

option( ENABLE_MPI "Define if MPI support should be enabled" OFF )
if (ENABLE_MPI)
  add_definitions(-DENABLE_MPI)
endif()

option( ENABLE_CUDA "Define if CUDA support is enabled or not" OFF )
if (ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
endif()

if( ENABLE_STARPU )
  include(FindPkgConfig)
  find_package(PkgConfig QUIET)

  set (starpu_version "1.3")
  if( ENABLE_MPI )
    pkg_search_module(STARPU starpumpi-${starpu_version})
  else()
    pkg_search_module(STARPU starpu-${starpu_version})
  endif()

  if (STARPU_FOUND AND STARPU_LIBRARIES)
    message(STATUS "Looking for STARPU - found using PkgConfig")
  else()
    message(ERROR "${Magenta}Looking for STARPU - not found using PkgConfig."
      "\n   Perhaps you should add the directory containing libstarpu.pc"
      "\n   to the PKG_CONFIG_PATH environment variable.${ColourReset}")
  endif()

  include_directories(${STARPU_INCLUDE_DIRS})
  link_directories(${STARPU_LIBRARY_DIRS})
endif()

add_subdirectory( algonum )
add_subdirectory( myblas )
add_subdirectory( testings )
