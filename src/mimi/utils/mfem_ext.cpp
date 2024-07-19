#include "mimi/utils/mfem_ext.hpp"

namespace mimi::utils {
MeshExt::GetFaceElementTransformations(int FaceNo,
                                       int mask,
                                       FaceElementTransformationsExt& f_tr) {
  FaceInfo& face_info = faces_info[FaceNo];

  int cmask = 0;
  f_tr.SetConfigurationMask(cmask);
  f_tr.Elem1 = NULL;
  f_tr.Elem2 = NULL;

  // setup the transformation for the first element
  f_tr.Elem1No = face_info.Elem1No;
  if (mask & mfem::FaceElementTransformations::HAVE_ELEM1) {
    GetElementTransformation(f_tr.Elem1No, &Transformation);
    f_tr.Elem1 = &Transformation;
    cmask |= 1;
  }

  //  setup the transformation for the second element
  //     return NULL in the Elem2 field if there's no second element, i.e.
  //     the face is on the "boundary"
  f_tr.Elem2No = face_info.Elem2No;
  if ((mask & mfem::FaceElementTransformations::HAVE_ELEM2)
      && f_tr.Elem2No >= 0) {
#ifdef MFEM_DEBUG
    if (NURBSext && (mask & mfem::FaceElementTransformations::HAVE_ELEM1)) {
      MFEM_ABORT("NURBS mesh not supported!");
    }
#endif
    GetElementTransformation(f_tr.Elem2No, &Transformation2);
    f_tr.Elem2 = &Transformation2;
    cmask |= 2;
  }

  // setup the face transformation
  if (mask & mfem::FaceElementTransformations::HAVE_FACE) {
    GetFaceTransformation(FaceNo, &f_tr);
    cmask |= 16;
  } else {
    f_tr.SetGeometryType(GetFaceGeometry(FaceNo));
  }

  // setup Loc1 & Loc2
  int face_type = GetFaceElementType(FaceNo);
  if (mask & mfem::FaceElementTransformations::HAVE_LOC1) {
    int elem_type = GetElementType(face_info.Elem1No);
    GetLocalFaceTransformation(face_type,
                               elem_type,
                               f_tr.Loc1.Transf,
                               face_info.Elem1Inf);
    cmask |= 4;
  }
  if ((mask & mfem::FaceElementTransformations::HAVE_LOC2)
      && f_tr.Elem2No >= 0) {
    int elem_type = GetElementType(face_info.Elem2No);
    GetLocalFaceTransformation(face_type,
                               elem_type,
                               f_tr.Loc2.Transf,
                               face_info.Elem2Inf);

    // NC meshes: prepend slave edge/face transformation to Loc2
    if (Nonconforming() && IsSlaveFace(face_info)) {
      ApplyLocalSlaveTransformation(f_tr, face_info, false);
    }
    cmask |= 8;
  }

  f_tr.SetConfigurationMask(cmask);

  // This check can be useful for internal debugging, however it will fail on
  // periodic boundary faces, so we keep it disabled in general.
#if 0
#ifdef MFEM_DEBUG
   double dist = f_tr.CheckConsistency();
   if (dist >= 1e-12)
   {
      mfem::out << "\nInternal error: face id = " << FaceNo
                << ", dist = " << dist << '\n';
      f_tr.CheckConsistency(1); // print coordinates
      MFEM_ABORT("internal error");
   }
#endif
#endif
}

void MeshExt::GetBdrFaceTransformations(int BdrElemNo,
                                        FaceElementTransformationsExt* tr) {
  int fn = GetBdrElementFaceIndex(BdrElemNo);

  // Check if the face is interior, shared, or nonconforming.
  if (FaceIsTrueInterior(fn) || faces_info[fn].NCFace >= 0) {
    return;
  }
  GetFaceElementTransformations(fn, 21, *tr);
  tr->Attribute = boundary[BdrElemNo]->GetAttribute();
  tr->ElementNo = BdrElemNo;
  tr->ElementType = mfem::ElementTransformation::BDR_FACE;
  tr->mesh = this;
}
} // namespace mimi::utils
