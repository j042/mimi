import splinepy as sp
import mimi
import gustaf as gus

import numpy as np

sp.settings.NTHREADS = 4


def py_mat(F, P):
    I = np.eye(F.shape[0])
    C = F.T @ F
    E = 0.5 * (C - I)
    S = 50 * np.trace(E) * I + 2 * 200 * E
    P[:] = F @ S 


def neo(F, sig):
    F = F.T
    I = np.eye(F.shape[0], dtype=float)
    det_F = np.linalg.det(F)
    muF = 200. / det_F
    B = F @ F.T
    sig[:] = muF * B + (-muF + 50. * (det_F-1.)) * I

#  create nl solid
nl = mimi.PyNonlinearSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(1)

# create material
# PyMaterial is platzhalter
mat = mimi.PythonMaterial()
mat.mat = neo
mat.density = 1
mat.viscosity = -1
mat.use_cauchy = True
nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
# bc.initial.dirichlet(1, 0).dirichlet(1, 1)
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
# bc.initial.dirichlet(3, 0).dirichlet(3, 1)
bc.initial.body_force(1, -1)

nl.boundary_condition = bc

nl.setup(1)
#nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, False)

rhs = nl.linear_form_view2("rhs")

nl.time_step_size = 0.005

x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
s.show_options["control_point_ids"] = False
# s.show_options["knots"] = False
s.show_options["resolutions"] = 50
s.cps[:] = x[to_s]

plt = gus.show(s, close=False)
for i in range(10000):
    if True:
        s.cps[:] = x[to_s]
        gus.show(
            [str(i), s],
            vedoplot=plt,
            interactive=False,
        )
    nl.step_time2()

gus.show(s, vedoplot=plt, interactive=True)
