import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.LinearElasticity()
le.read_mesh("tests/data/square-nurbs.mesh")

# set param
le.set_parameters(1e9, 0.4, 1000000, 10000)

# refine
le.elevate_degrees(2)
le.subdivide(2)

# create splinepy partner
s, to_m, to_s = mimi.to_splinepy(le)
o_cps = s.cps.copy()

# set bc
curv = sp.Bezier(
    [3],
    [
        [-2.5, 1.3],
        [0.3, 0.7],
        [0.7, 0.7],
        [1.5, 1.3],
    ],
)
curv.cps[:] += [0.05, 1]

scene = mimi.NearestDistanceToSplines()
scene.add_spline(curv)
scene.plant_kd_tree(100000, 4)
scene.coefficient = 1e11

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1)
bc.current.contact(1, scene)
le.boundary_condition = bc

tic.toc()

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("linear_elasticity", 1e-14, 1e-8, 20, False)

tic.toc("bilinear, linear forms assembly")

# set step size
le.time_step_size = 0.001

# get view of solution, displacement
u = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())

tic.summary(print_=True)
# set visualization options
s.show_options["control_point_ids"] = False
s.show_options["resolutions"] = 100
s.show_options["control_points"] = False
s.cps[:] = u[to_s] + o_cps

tic.summary(print_=True)
# initialize a plotter
plt = gus.show([s, curv], close=False)
for i in range(10000):
    tic.toc("stepped")
    s.cps[:] = u[to_s] + o_cps
    gus.show(
        [s, curv],
        vedoplot=plt,
        interactive=False,
    )
    tic.toc("showing")
    le.step_time2()
    if i < 50:
        curv.cps[:] -= [0, 0.005]
    else:
        curv.cps[:] -= [0.005, 0]
    scene.plant_kd_tree(100000, 4)

tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
