import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

#  create nl solid
nl = mimi.NonlinearSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(2)

# create material
mat = mimi.CompressibleOgdenNeoHookean()
mat.density = 1
mat.viscosity = 1

# define material properties (young's modulus, poisson's ratio)
mat.set_young_poisson(210, 0.3)

nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
to_m = np.array(to_m)
to_s = np.array(to_s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
# define dirichlet boundaries - those values won't be touched by solvers
bc.initial.dirichlet(2, 0).dirichlet(2, 1)  # .dirichlet(3, 0).dirichlet(3, 1)
bc.initial.constant_velocity(3, 0, 0.5)
bc.initial.constant_velocity(3, 1, 0.5)
# bc.initial.body_force(1, -50)

# get true dof indices to apply dirichlet BC
mi = s.multi_index
b3 = to_s[mi[-1, :]]
b2 = to_s[mi[0, :]]

nl.boundary_condition = bc

nl.setup(4)
nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, True)

# rhs = nl.linear_form_view2("rhs")

nl.time_step_size = 0.01

x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
s.show_options["control_point_ids"] = False
s.show_options["resolutions"] = 50
s.cps[:] = x[to_s]


def show():
    s.cps[:] = x[to_s]
    gus.show(
        [str(i), s],
        vedoplot=plt,
        interactive=False,
    )


plt = gus.show(s, close=False)
for i in range(10000):
    if i < 300:
        bc.initial.constant_velocity(3, 0, 0.5)
        bc.initial.constant_velocity(3, 1, 0.5)
    elif i < 600:
        bc.initial.constant_velocity(3, 0, 0.0)
        bc.initial.constant_velocity(3, 1, -1)

    #    if i < 300:
    #        x[b3] += [0.01, 0.01]
    #        x[b2] -= [0.01, 0.01]
    #    elif i > 300 and i < 900:
    #        x[b3] += [0, -0.01]
    #        x[b2] -= [0, -0.01]
    #    elif i > 900 and i < 1200:
    #        x[b3] += [-0.01, 0.01]
    #        x[b2] -= [-0.01, 0.01]
    #    if i == 1500:
    #        rhs[:] = 0.0
    s.cps[:] = x[to_s]
    gus.show(
        [str(i), s],
        vedoplot=plt,
        interactive=False,
    )
    nl.step_time2()

gus.show(s, vedoplot=plt, interactive=True)
