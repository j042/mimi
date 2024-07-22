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
  using MixedBilinearFormPointer_ = std::shared_ptr<mfem::MixedBilinearForm>;
  using NonlinearFormPointer_ = std::shared_ptr<mimi::forms::Nonlinear>; //

  const mfem::Array<int>* dirichlet_dofs_{nullptr};

  // set dt
  double dt_{0.0};

  /// @brief container to hold any necessary / related linear forms
  std::unordered_map<std::string, LinearFormPointer_> linear_forms_{};
  /// @brief container to hold any necessary / related bilinear forms
  std::unordered_map<std::string, BilinearFormPointer_> bilinear_forms_{};
  /// @brief container to hold any necessary / related mixed bilinear forms
  std::unordered_map<std::string, MixedBilinearFormPointer_>
      mixed_bilinear_forms_{};
  /// @brief container to hold any necessary / related nonlinear forms
  std::unordered_map<std::string, NonlinearFormPointer_> nonlinear_forms_{};

  /// @brief newton solver should be configured outside of this class
  std::shared_ptr<mimi::solvers::Newton> newton_solver_{nullptr};

  /// @brief ctor
  /// @param fe_space
  OperatorBase() = default;

  virtual ~OperatorBase() {}

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

  /// @brief adds mixed bilinear form
  /// @param key
  /// @param mbf
  virtual void AddMixedBilinearForm(std::string const& key,
                                    const MixedBilinearFormPointer_& mbf) {
    MIMI_FUNC();

    mixed_bilinear_forms_[key] = mbf;
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

  virtual void PostTimeAdvance(const mfem::Vector& x,
                               const mfem::Vector& v) = 0;

  virtual void Setup() = 0;
};

// TODO: maybe make it part of OperatorBase OR get array of FE-spaces
class OperatorTwoBases {
public:
  using LinearFormPointer_ = std::shared_ptr<mfem::LinearForm>;
  using BilinearFormPointer_ = std::shared_ptr<mfem::BilinearForm>;
  using MixedBilinearFormPointer_ = std::shared_ptr<mfem::MixedBilinearForm>;
  using NonlinearFormPointer_ = std::shared_ptr<mimi::forms::Nonlinear>; //

  mfem::FiniteElementSpace& fe_space_1_;
  mfem::FiniteElementSpace& fe_space_2_;

  const mfem::Array<int>* dirichlet_dofs_1_{nullptr};
  const mfem::Array<int>* dirichlet_dofs_2_{nullptr};

  // set dt
  double dt_{0.0};

  /// @brief container to hold any necessary / related linear forms
  std::unordered_map<std::string, LinearFormPointer_> linear_forms_{};
  /// @brief container to hold any necessary / related bilinear forms
  std::unordered_map<std::string, BilinearFormPointer_> bilinear_forms_{};
  /// @brief container to hold any necessary / related mixed bilinear forms
  std::unordered_map<std::string, MixedBilinearFormPointer_>
      mixed_bilinear_forms_{};
  /// @brief container to hold any necessary / related nonlinear forms
  std::unordered_map<std::string, NonlinearFormPointer_> nonlinear_forms_{};

  /// @brief newton solver should be configured outside of this class
  std::shared_ptr<mimi::solvers::Newton> newton_solver_{nullptr};

  /// @brief ctor
  /// @param fe_space
  OperatorBase(mfem::FiniteElementSpace& fe_space_1,
               mfem::FiniteElementSpace& fe_space_2)
      : fe_space_1_(fe_space_1),
        fe_space_2_(fe_space_2) {}

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

  /// @brief adds mixed bilinear form
  /// @param key
  /// @param mbf
  virtual void AddMixedBilinearForm(std::string const& key,
                                    const MixedBilinearFormPointer_& mbf) {
    MIMI_FUNC();

    mixed_bilinear_forms_[key] = mbf;
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

  // TODO:
  virtual void PostTimeAdvance(const mfem::Vector& v,
                               const mfem::Vector& protected) = 0;

  virtual void Setup() = 0;
};

} // namespace mimi::operators
