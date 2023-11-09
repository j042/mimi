add_subdirectory(mfem)
add_subdirectory(pybind11)

if(MIMI_COMPILE_SPLINEPY)
  message("buildling splinepy")
  set(SPLINEPY_COMPILE_PYTHON ON)

  add_subdirectory(splinepy)
else()
  message("finding splinepy")
  find_package(splinepy REQUIRED GLOBAL)
endif()
