import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

le = mimi.PyLinearElasticity()
le.read_mesh("tests/data/balken.mesh")

le.set_parameters(3000, .3, 4, -1)

le.elevate_degrees(2)
le.subdivide(1)

s = sp.NURBS(**le.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2,1)
bc.initial.body_force(1, -10)

le.boundary_condition = bc

tic.toc()

le.setup(4)

tic.toc()
tic.summary(print_=True)

rhs = le.linear_form_view2("rhs")

le.time_step_size = .1

x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())
s.show_options["control_point_ids"] = False
#s.show_options["knots"] = False
s.show_options["resolutions"] = 50
s.cps[:] = x[to_s]

plt = gus.show(s, close=False)
for i in range(1000):
    #if i % 10 == 0:
    tic.toc("steped")
    if True:
        s.cps[:] = x[to_s]
        gus.show(s, vedoplot=plt, interactive=False,)
    tic.toc("showed")
    le.step_time2()

tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)

