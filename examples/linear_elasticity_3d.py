import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.LinearElasticity()
le.read_mesh("tests/data/cube-nurbs.mesh")

# set param
le.set_parameters(5000, 0.3, 4, -1)

# refine
le.elevate_degrees(1)
le.subdivide(2)

# create splinepy partner
s, to_m, to_s = mimi.to_splinepy(le)
o_cps = s.cps.copy()

# set bc
bc = mimi.BoundaryConditions()
bc.initial.dirichlet(1, 0).dirichlet(1, 1).dirichlet(1, 2)
bc.initial.body_force(1, -200)
le.boundary_condition = bc

tic.toc()

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

tic.toc()
tic.summary(print_=True)

# set step size
le.time_step_size = 0.01

# get view of solution, displacement
u = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())

# set visualization options
s.show_options["control_point_ids"] = False
s.show_options["resolutions"] = 50
s.cps[:] = u[to_s] + o_cps

# initialize a plotter
plt = gus.show(s, close=False)
for i in range(100):
    tic.toc("stepped")
    s.cps[:] = u[to_s] + o_cps
    gus.show(
        s,
        vedoplot=plt,
        interactive=False,
    )
    tic.toc("showing")
    le.step_time2()

tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
