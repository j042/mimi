set(SRCS
    solvers/newton.cpp
    py/py_boundary_conditions.cpp
    py/py_solid.cpp
    py/py_linear_elasticity.cpp
    py/py_nearest_distance.cpp
    py/py_nonlinear_form.cpp
    py/py_material.cpp
    py/py_nonlinear_solid.cpp
    py/py_nonlinear_visco_solid.cpp
    py/py_ad.cpp
    py/py_nonlinear_integrator.cpp
    py/mimi_core.cpp)

pybind11_add_module(mimi_core ${SRCS})

# include
target_include_directories(mimi_core PRIVATE ${PROJECT_SOURCE_DIR}/src)

# link
target_link_libraries(mimi_core PRIVATE mfem)
target_link_libraries(mimi_core PRIVATE splinepy::splinepy_python)

target_compile_options(mimi_core PRIVATE ${MIMI_FLAGS})
target_compile_definitions(mimi_core PRIVATE ${MIMI_DEFS})
set_property(TARGET mimi_core PROPERTY CXX_STANDARD 17)

if(MIMI_USE_OMP)
  find_package(OpenMP REQUIRED)
  target_compile_definitions(mimi_core PRIVATE MIMI_USE_OMP)
  target_link_libraries(mimi_core PRIVATE OpenMP::OpenMP_CXX)
  # target_compile_options(mimi_core PRIVATE -fopenmp)
  target_compile_options(mimi_core PRIVATE ${OpenMP_C_FLAGS}
                                           ${OpenMP_CXX_FLAGS})
endif()
