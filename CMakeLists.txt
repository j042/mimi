cmake_minimum_required(VERSION 3.12.0)

project(mimi)

option(MIMI_COMPILE_SPLINEPY OFF)

#set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX}/lib)

#find_package(OpenMP REQUIRED)

#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU"
#    OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  set(MFEMIGA_FLAGS -O3 -fvisibility=default)
#endif()

if (CMAKE_BUILD_TYPE MATCHES Release)
  set(MIMI_FLAGS)
  set(MIMI_DEFS NDEBUG)
else()
  set(MIMI_FLAGS -O3)
  set(MIMI_DEFS)
endif()

add_subdirectory(third_party)
add_subdirectory(src)