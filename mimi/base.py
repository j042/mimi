import splinepy
import numpy as np

from mimi import mimi_core


def to_splinepy(pysolid_or_dict):
    """
    Extracts nurbs from mimi.PySolid and creates splinepy.NURBS (or BSpline)
    """

    if isinstance(pysolid_or_dict, mimi_core.Solid):
        dict_spline = pysolid_or_dict.nurbs()
    elif isinstance(pysolid_or_dict, dict):
        dict_spline = pysolid_or_dict
    else:
        raise TypeError("Unsupported type. Expects mimi.Solid or dict.")

    ws = dict_spline["weights"]
    s = None
    if all(ws == ws[0]):
        dict_spline.pop("weights")
        s = splinepy.BSpline(**dict_spline)
    else:
        s = splinepy.NURBS(**dict_spline)

    to_m, to_s = splinepy.io.mfem.dof_mapping(s)
    s.cps[:] = s.cps[to_s]

    return s, np.array(to_m, dtype=int), np.array(to_s, dtype=int)
