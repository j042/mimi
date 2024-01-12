import time

import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

#  create nl solid
nl = mimi.PyNonlinearViscoSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(3)

# create material
mat = mimi.PyJ2AdiabaticViscoIsotropicHardening()
mat.density = 1


mat.viscosity = -1

# define material properties (young's modulus, poisson's ratio)
# mat.set_young_poisson(210000, 0.3)

# instead, one can also use lame's parameter lambda and mu
# define material properties (lamda, mu)
mat.set_lame((790000 - (79000 * 2 / 3)) * 10, 790000)

mat.heat_fraction = 0.9
mat.specific_heat = 450
mat.initial_temperature = 800

mat.hardening = mimi.PyJohnsonCookThermoViscoHardening()
mat.hardening.A = 100
mat.hardening.B = mat.hardening.A * 2.5
mat.hardening.n = 0.2835
mat.hardening.C = 0.034
mat.hardening.eps0_dot = 0.004
mat.hardening.reference_temperature = 20
mat.hardening.melting_temperature = 1500
mat.hardening.m = 1.3558


nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
bc.initial.body_force(1, -7)
# bc.initial.traction(3, 1, -50)

nl.boundary_condition = bc

nl.setup(1)
nl.configure_newton("nonlinear_visco_solid", 1e-10, 1e-8, 20, False)

rhs = nl.linear_form_view2("rhs")
print(rhs)

nl.time_step_size = 0.005
x_ref = nl.solution_view("displacement", "x_ref").reshape(-1, nl.mesh_dim())
x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
v = nl.solution_view("displacement", "x_dot").reshape(-1, nl.mesh_dim())
s.show_options["control_point_ids"] = False
s.show_options["control_points"] = False
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
            close=False,
            interactive=False,
        )
    # remove body force
    # if i == 75:
    #    rhs[:] *= -1.0
    # if i == 150:
    #    rhs[:] = 0.0

    nl.step_time2()
    print(i, np.linalg.norm(x - x_ref))
    # print(x[:4])
    # print(v[:4])
    # time.sleep(.5)

gus.show(s, vedoplot=plt, interactive=True)
