import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

#  create nl solid
nl = mimi.PyNonlinearSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(1)

# create material
# PyMaterial is platzhalter
mat = mimi.PyStVenantKirchhoff()
mat.density = 1
mat.viscosity = -1
mat.lambda_ = 1
mat.mu = 1
nl.set_material(mat)

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
#bc.initial.dirichlet(1, 0).dirichlet(1, 1)
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
#bc.initial.dirichlet(3, 0).dirichlet(3, 1)
bc.initial.body_force(1, -0.01)

nl.boundary_condition = bc

nl.setup(2)

rhs = nl.linear_form_view2("rhs")

nl.time_step_size = 0.05

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
            [str(i),s],
            vedoplot=plt,
            interactive=False,
        )
    nl.step_time2()

gus.show(s, vedoplot=plt, interactive=True)
