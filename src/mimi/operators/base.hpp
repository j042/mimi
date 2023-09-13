#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mfem.hpp>

#include "mimi/forms/nonlinear.hpp"
#include "mimi/solvers/newton.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::operators {

class OperatorBase {
public:
  using LinearFormPointer_ = std::shared_ptr<mfem::LinearForm>;
  using BilinearFormPointer_ = std::shared_ptr<mfem::BilinearForm>;
  using NonlinearFormPointer_ = std::shared_ptr<mimi::forms::Nonlinear>; //

  mfem::FiniteElementSpace& fe_space_;

  /// @brief container to hold any necessary / related linear forms
  std::unordered_map<std::string, LinearFormPointer_> linear_forms_{};
  /// @brief container to hold any necessary / related bilinear forms
  std::unordered_map<std::string, BilinearFormPointer_> bilinear_forms_{};
  /// @brief container to hold any necessary / related nonlinear forms
  std::unordered_map<std::string, NonlinearFormPointer_> nonlinear_forms_{};

  /// @brief newton solver should be configured outside of this class
  std::shared_ptr<mimi::solvers::Newton> newton_solver_{nullptr};

  /// @brief ctor
  /// @param fe_space
  OperatorBase(mfem::FiniteElementSpace& fe_space) : fe_space_(fe_space) {}

  /// @brief adds linear form
  /// @param key
  /// @param lf
  virtual void AddLinearForm(std::string const& key,
                             const LinearFormPointer_& lf) {
    MIMI_FUNC()

    linear_forms_[key] = lf;
  }

  /// @brief adds bilinearform
  /// @param key
  /// @param bf
  virtual void AddBilinearForm(std::string const& key,
                               const BilinearFormPointer_& bf) {
    MIMI_FUNC();

    bilinear_forms_[key] = bf;
  }

  /// @brief adds nonlinear form
  /// @param key
  /// @param nf
  virtual void AddNonlinearForm(std::string const& key,
                                const NonlinearFormPointer_& nf) {
    MIMI_FUNC()

    nonlinear_forms_[key] = nf;
  }

  /// @brief sets newton solver for the operator
  /// @param newton_solver
  virtual void
  SetNewtonSolver(std::shared_ptr<mimi::solvers::Newton> const& newton_solver) {
    MIMI_FUNC();

    newton_solver_ = newton_solver;
  }

  virtual void Setup() = 0;
};

} // namespace mimi::operators
