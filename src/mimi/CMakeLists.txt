set(SRCS
    py/py_boundary_conditions.cpp py/py_solid.cpp py/py_linear_elasticity.cpp
    py/py_nearest_distance.cpp py/py_nonlinear_form.cpp py/mimi.cpp)

pybind11_add_module(mimi ${SRCS})

# include
target_include_directories(mimi PRIVATE ${PROJECT_SOURCE_DIR}/src)

# link
target_link_libraries(mimi PRIVATE mfem)
target_link_libraries(mimi PRIVATE splinepy::splinepy_python)

target_compile_options(mimi PRIVATE ${MIMI_FLAGS})
target_compile_definitions(mimi PRIVATE ${MIMI_DEFS})
set_property(TARGET mimi PROPERTY CXX_STANDARD 17)
