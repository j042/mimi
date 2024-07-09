#pragma once

#include <mfem.hpp>

#include "mimi/utils/containers.hpp"

namespace mimi::materials {

using MaterialState =
    typename mimi::utils::DataSeries<mfem::DenseMatrix, mfem::Vector, double>;

}
