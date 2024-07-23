cmake_minimum_required(VERSION 3.12.0)

project(mimi)

option(MIMI_COMPILE_SPLINEPY "compile splinepy" OFF)
option(MIMI_USE_OMP "Use openmp. else std::thread" ON)

# frequently used path during development
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/lib)
if(MIMI_USE_OMP)
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/libomp)
endif()

# we need to set `-fPIC` flag to any dependency that we build with mimi this
# including mfem.
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -fPIC)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

  set(MIMI_FLAGS -O3)
  set(MIMI_DEFS NDEBUG)
else()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(MIMI_WARNING_FLAGS -Werror -Wall -Wextra -Wpedantic
                           -Wzero-as-null-pointer-constant -Wno-unused)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MIMI_WARNING_FLAGS
        -Werror
        -Wall
        -Wmost
        -Wextra
        -Wpedantic
        -Wunreachable-code
        -Wshadow
        -Wfloat-equal
        -Weffc++
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wzero-as-null-pointer-constant)
  else()
    message(WARNING "Untested compiler. mimi is tested with GNU and Clang.")
  endif()

  set(MIMI_FLAGS "${MIMI_FLAGS} ${MIMI_WARNING_FLAGS}")
  set(MIMI_DEFS)
endif()

add_subdirectory(third_party)
add_subdirectory(src/mimi)
