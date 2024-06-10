import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

#  create nl solid
nl = mimi.PyNonlinearSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(3)

# create material
mat = mimi.PyJ2()
mat.density = 1
mat.viscosity = 10

# define material properties (young's modulus, poisson's ratio)
mat.set_young_poisson(210000, 0.3)

# instead, one can also use lame's parameter lambda and mu
# define material properties (lamda, mu)
# mat.set_lame(26333, 79000)

mat.isotropic_hardening = 0
mat.kinematic_hardening = 0
mat.sigma_y = 165 * 3 ** (0.5)
nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
# bc.initial.body_force(1, -1000)
bc.initial.traction(3, 1, -30)

nl.boundary_condition = bc

nl.setup(4)
nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 40, False, False)
rhs = nl.linear_form_view2("rhs")
print(rhs)

nl.time_step_size = 0.01
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
            interactive=False,
        )
    # remove body force
    if i == 200:
        rhs[:] = 0.0
    nl.step_time2()

gus.show(s, vedoplot=plt, interactive=True)
