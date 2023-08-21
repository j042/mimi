set(SRCS py/py_boundary_conditions.cpp py/py_solid.cpp py/mimi.cpp)

pybind11_add_module(mimi ${SRCS})

# include
target_include_directories(mimi PRIVATE ${PROJECT_SOURCE_DIR}/src)

# link
target_link_libraries(mimi PRIVATE mfem)
# target_link_libraries(mimi PRIVATE splinepy)

if(MFEM_USE_SUITESPARSE)
  # find_package(SuiteSparse REQUIRED)
  target_link_libraries(mimi PRIVATE $ENV{CONDA_PREFIX}/lib/libumfpack.a)
  target_include_directories(mimi PRIVATE $ENV{CONDA_PREFIX}/include)
endif()
target_compile_options(mimi PRIVATE ${MIMI_FLAGS})
target_compile_definitions(mimi PRIVATE ${MIMI_DEFS})
set_property(TARGET mimi PROPERTY CXX_STANDARD 17)