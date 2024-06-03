cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

set(lib_dest "lib")
set(incl_dest "include")

find_package(ZLIB REQUIRED)

add_library(cnpy STATIC "cnpy.cpp")
target_compile_options(cnpy PRIVATE -fPIC)
target_include_directories(
  cnpy PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${incl_dest}>)
set_target_properties(cnpy PROPERTIES CXX_STANDARD 11)

install(TARGETS "cnpy" ARCHIVE DESTINATION ${lib_dest})
install(FILES "cnpy.h" DESTINATION ${incl_dest})
