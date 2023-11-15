cmake_minimum_required(VERSION 3.12.0)

project(mimi)

option(MIMI_COMPILE_SPLINEPY "compile splinepy" OFF)

# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/install)

# we need to set `-fPIC` flag to any dependency that we build with mimi this
# include mfem.
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -fPIC)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(MIMI_FLAGS -O3)
  set(MIMI_DEFS NDEBUG)
else()
  set(MIMI_FLAGS)
  set(MIMI_DEFS)
endif()

add_subdirectory(third_party)
add_subdirectory(src/mimi)
