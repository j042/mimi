import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.NonlinearSolid()
le.read_mesh("tests/data/sqn.mesh")

# refine
le.elevate_degrees(1)
le.subdivide(2)

# mat
mat = mimi.J2()
mat.density = 7e4
mat.viscosity = 10
mat.set_young_poisson(1e10, 0.3)
mat.isotropic_hardening = 1e8
mat.kinematic_hardening = 0
mat.sigma_y = 1e7
le.set_material(mat)

# create splinepy partner
s, to_m, to_s = mimi.to_splinepy(le)
o_cps = s.cps.copy()

# set bc
curv = sp.Bezier(
    [3],
    [
        [-10.5, 2.3],
        [-0.3, 0.7],
        [0.3, 0.7],
        [10.5, 2.3],
    ],
)
curv.cps[:] += [0.0, 0.75]


scene = mimi.NearestDistanceToSplines()
scene.add_spline(curv)
scene.plant_kd_tree(100000, 4)
scene.coefficient = 0.5e11

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1)
bc.initial.periodic(3, 4)
bc.current.contact(1, scene)
le.boundary_condition = bc

tic.toc()

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("nonlinear_solid", 1e-14, 1e-8, 20, False)

tic.toc("bilinear, linear forms assembly")

# set step size
le.time_step_size = 0.001

# get view of solution, displacement
u = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())
u_ref = le.solution_view("displacement", "x_ref").reshape(-1, le.mesh_dim())
dof_map = le.dof_map("displacement")

tic.summary(print_=True)
# set visualization options
s.show_options["resolutions"] = [100, 30]
curv.show_options["control_points"] = False
# s.cps[:] = x_ref[to_s]

tic.summary(print_=True)


def move():
    if i < 100:
        curv.cps[:] -= [0, 0.005]
    else:
        curv.cps[:] -= [0.04, 0]

    scene.plant_kd_tree(1000, 4)


def show():
    s.cps = (o_cps + u[dof_map])[to_s]
    # s.cps = u[to_s] + o_cps
    gus.show(
        [
            str(i),
            s,
            curv,
        ],
        vedoplot=plt,
        interactive=False,
    )


coe = 0.1e11
scene.coefficient = coe
# initialize a plotter
plt = gus.show([s, curv], close=False, interactive=False)
for i in range(1000):
    move()
    le.step_time2()
    show()


tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
