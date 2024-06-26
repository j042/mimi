import splinepy

from mimi import mimi_core


def to_splinepy(pysolid):
    """
    Extracts nurbs from mimi.PySolid and creates splinepy.NURBS (or BSpline)
    """
    if not isinstance(pysolid, mimi_core.Solid):
        raise TypeError("Expecting mimi.PySolid types.")

    dict_spline = pysolid.nurbs()
    ws = dict_spline["weights"]
    s = None
    if all(ws == ws[0]):
        dict_spline.pop("weights")
        s = splinepy.BSpline(**dict_spline)
    else:
        s = splinepy.NURBS(**dict_spline)

    to_m, to_s = splinepy.io.mfem.dof_mapping(s)
    s.cps[:] = s.cps[to_s]

    return s, to_m, to_s
