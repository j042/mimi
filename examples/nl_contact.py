import splinepy as sp
import mimi

sp.settings.NTHREADS = 4

# init, read mesh
nl = mimi.NonlinearSolid()
nl.read_mesh("tests/data/square-nurbs.mesh")

# refine
nl.elevate_degrees(1)
nl.subdivide(3)

# mat
mat = mimi.CompressibleOgdenNeoHookean()
mat.density = 7e4
mat.viscosity = -1
mat.set_young_poisson(1e10, 0.3)
nl.set_material(mat)

# create splinepy partner
s, to_m, to_s = mimi.to_splinepy(nl)
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
scene.coefficient = 0.5e11

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1)
bc.current.contact(1, scene)
nl.boundary_condition = bc

# setup needs to be called this assembles bilinear forms, linear forms
nl.setup(4)

nl.configure_newton("nonlinear_solid", 1e-10, 1e-8, 100, False)

# set step size
nl.time_step_size = 0.001

# get view of solution, displacement
u = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())

# set visualization options
s.show_options["resolutions"] = [100, 30]
s.show_options["control_points"] = False
curv.show_options["control_points"] = False
s.cps[:] = u[to_s] + o_cps


def move():
    if i < 100:
        curv.cps[:] -= [0, 0.005]
    else:
        curv.cps[:] -= [0.005, 0]
    scene.plant_kd_tree(10000, 4)


def show():
    s.cps[:] = u[to_s] + o_cps
    sp.show(
        [
            str(i),
            s,
            curv,
        ],
        vedoplot=plt,
        interactive=False,
    )


scene.coefficient = 1e11

# initialize a plotter
plt = sp.show([s, curv], close=False)
for i in range(1000):
    move()
    nl.step_time2()
    show()

sp.show(s, vedoplot=plt, interactive=True)
