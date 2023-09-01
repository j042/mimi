#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/print.hpp"

namespace mimi::integrators {

class PenaltyContact : public mfem::NonlinearFormIntegrator {
public:
};

} // namespace mimi::integrators
