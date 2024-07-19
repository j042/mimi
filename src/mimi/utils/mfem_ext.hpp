#pragma once

#include <mfem.hpp>

namespace mimi::utils {

class MeshExt;

class FaceElementTransformationsExt : public mfem::FaceElementTransformations {
protected:
  friend class MeshExt;
};

class MeshExt : public mfem::Mesh {
public:
  using Base_ = mfem::Mesh;
  using Base_::Base_;

  void GetFaceElementTransformations(int FaceNo,
                                     int mask,
                                     FaceElementTransformationsExt& f_tr);

  /// just reimplementing GetBdrFaceTransformation to fill in the pointer
  /// instead of returning internal ptr
  void GetBdrFaceTransformations(int BdrElemNo,
                                 FaceElementTransformationsExt* tr);
};

} // namespace mimi::utils
