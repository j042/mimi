import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 1

#  create nl solid
nl = mimi.PyNonlinearSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(3)

# create material
mat = mimi.PyJ2NonlinearIsotropicHardening()
mat.density = 10

mat.viscosity = 10

# define material properties (young's modulus, poisson's ratio)
mat.set_young_poisson(210000, 0.3)

# instead, one can also use lame's parameter lambda and mu
# define material properties (lamda, mu)
# mat.set_lame(26333, 79000)

mat.hardening = mimi.PyJohnsonCookHardening()
mat.hardening.A = 100
mat.hardening.B = mat.hardening.A * 2.5
mat.hardening.n = 0.2835

nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
bc.initial.body_force(1, -100)
# bc.initial.traction(3, 1, -100)

nl.boundary_condition = bc

nl.setup(1)
nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 40, True)

rhs = nl.linear_form_view2("rhs")
print(rhs)

nl.time_step_size = 0.005
x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
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
    if i == 75:
        rhs[:] *= -1.0
    if i == 150:
        rhs[:] = 0.0
    # nl.step_time2()
    nl.fixed_point_solve2()
    nl.advance_time2()

gus.show(s, vedoplot=plt, interactive=True)
