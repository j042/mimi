#include <pybind11/pybind11.h>

#include <mfem.hpp>

#include "mimi/forms/nonlinear.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_nonlinear_form(py::module_& m) {
  using NLF = mimi::forms::Nonlinear;
  py::class_<NLF, std::shared_ptr<NLF>> klasse(m, "PyNonlinearForm");

  klasse.def(
      "boundary_residual",
      [](const NLF& nlf) -> py::array_t<double> {
        // create return array
        py::array_t<double> residual(nlf.Height());

        // create mfem vector view
        mfem::Vector res_vector(static_cast<double*>(residual.request().ptr),
                                nlf.Height());
        // initialize 0.0
        res_vector = 0.0;

        // 1. loop boundary integrals
        // 2. loop saved boundary residuals and add
        // see NLF::Mult()
        for (const auto& boundary_integ : nlf.boundary_face_nfi_) {
          const auto& bel_vecs = *boundary_integ->boundary_element_vectors_;
          const auto& bel_vdofs =
              boundary_integ->precomputed_->boundary_v_dofs_;
          for (const auto& boundary_marks :
               boundary_integ->marked_boundary_elements_) {
            res_vector.AddElementVector(*bel_vdofs[boundary_marks],
                                        bel_vecs[boundary_marks]);
          }
        }

        return residual;
      },
      "Returns latest boundary residual of NonlinearForm");
}

} // namespace mimi::py