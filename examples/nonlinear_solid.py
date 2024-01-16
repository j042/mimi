import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

#  create nl solid
nl = mimi.PyNonlinearSolid()
nl.read_mesh("tests/data/cube-nurbs.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(3)

# create material
# PyMaterial is platzhalter
mat = mimi.PyStVenantKirchhoff()
mat = mimi.PyCompressibleOgdenNeoHookean()
mat.density = 1
mat.viscosity = -1

# define material properties (young's modulus, poisson's ratio)
mat.set_young_poisson(2100, 0.3)

# instead, one can also use lame's parameter lambda and mu
# define material properties (lamda, mu)
# mat.set_lame(26333, 79000)

nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
# bc.initial.dirichlet(1, 0).dirichlet(1, 1)
bc.initial.dirichlet(1, 0).dirichlet(1, 1).dirichlet(1, 2)
bc.initial.traction(0, 1, -100)
# bc.initial.body_force(1, -100)

nl.boundary_condition = bc

nl.setup(4)
nl.configure_newton("nonlinear_solid", 1e-12, 1e-12, 11, False)

rhs = nl.linear_form_view2("rhs")

nl.time_step_size = 0.01

x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
s.show_options["control_point_ids"] = False
# s.show_options["control_points"] = False
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

#    if i == 0:
#        exit()

gus.show(s, vedoplot=plt, interactive=True)
