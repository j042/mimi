#pragma once

#include <vector>

#include <mfem.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mimi/utils/print.hpp"

namespace mimi::py {
namespace py = pybind11;

/*
 * mfem::Vector to numpy wrapper.
 * Should avoid copy. But I think it copies.
 */
template<typename ReturnType, typename MfemContainerType, typename... SizeType>
py::array_t<ReturnType> NumpyCopy(MfemContainerType& mfem_container,
                                  SizeType... sizes) {
  MIMI_FUNC()

  return py::array_t<ReturnType>(std::vector{static_cast<size_t>(sizes)...},
                                 mfem_container.GetData());
}

template<typename ReturnType, typename MfemContainerType, typename... SizeType>
py::array_t<ReturnType> NumpyView(MfemContainerType& mfem_container,
                                  SizeType... sizes) {
  MIMI_FUNC()

  return py::array_t<ReturnType>(std::vector{static_cast<size_t>(sizes)...},
                                 mfem_container.GetData(),
                                 py::none());
}

} // namespace mimi::py
