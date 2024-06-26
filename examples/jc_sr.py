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
nl.subdivide(3)

# create material
mat = mimi.J2ViscoIsotropic()
mat.density = 1

mat.viscosity = -10

# define material properties (lamda, mu)
mat.set_lame((790000 - (79000 * 2 / 3)) * 10, 790000)

mat.hardening = mimi.JohnsonCookViscoHardening()
mat.hardening.A = 100
mat.hardening.B = mat.hardening.A * 2.5
mat.hardening.n = 0.2835
mat.hardening.C = 0.034
mat.hardening.eps0_dot = 0.004

nl.set_material(mat)

# create splinepy nurbs to show
s, to_m, to_s = mimi.to_splinepy(nl)
o_cps = s.cps.copy()

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
bc.initial.body_force(1, -10)

nl.boundary_condition = bc

nl.setup(1)
nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 40, False)

nl.time_step_size = 0.005
u_ref = nl.solution_view("displacement", "x_ref").reshape(-1, nl.mesh_dim())
u = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
v = nl.solution_view("displacement", "x_dot").reshape(-1, nl.mesh_dim())
s.show_options["control_point_ids"] = False
s.show_options["control_points"] = False
s.show_options["resolutions"] = 50
s.cps[:] = u[to_s] + o_cps

plt = gus.show(s, close=False)
for i in range(10000):
    if True:
        s.cps[:] = u[to_s] + o_cps
        gus.show(
            [str(i), s],
            vedoplot=plt,
            close=False,
            interactive=False,
        )

    nl.step_time2()
    print(i, np.linalg.norm(u - u_ref))

gus.show(s, vedoplot=plt, interactive=True)
