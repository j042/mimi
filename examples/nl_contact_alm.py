import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.PyNonlinearSolid()
le.read_mesh("tests/data/sqn.mesh")

# set param

# refine
le.elevate_degrees(1)
le.subdivide(3)

# mat
mat = mimi.PyCompressibleOgdenNeoHookean()
mat = mimi.PyJ2()
mat.density = 7e4
mat.viscosity = 10
mat.set_young_poisson(1e10, 0.3)
mat.isotropic_hardening = 1e8
mat.kinematic_hardening = 0
mat.sigma_y = 1e7
le.set_material(mat)
# create splinepy partner
s = sp.NURBS(**le.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

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
curv.cps[:] += [0, 0.75]


scene = mimi.PyNearestDistanceToSplines()
scene.add_spline(curv)
scene.plant_kd_tree(1001, 4)
scene.coefficient = 1e10

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1)
bc.current.contact(1, scene)
le.boundary_condition = bc

rc = mimi.PyRuntimeCommunication()
# rc.set_int("contact_quadrature_order", 50)
# rc.set_int("nonlinear_solid_quadrature_order", 3)
rc.fname = "tout/n.npz"
rc.append_should_save("contact_history", 1)
rc.append_should_save("contact_forces", 1)
rc.setup_real_history("area", 10000)
rc.setup_real_history("force_x", 10000)
rc.setup_real_history("force_y", 10000)
le.runtime_communication = rc
tic.toc()

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("nonlinear_solid", 1e-10, 1e-8, 20, False)

tic.toc("bilinear, linear forms assembly")

# set step size
le.time_step_size = 0.001

# get view of solution, displacement
x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())

tic.summary(print_=True)
# set visualization options
# s.show_options["control_points"] = False
# s.show_options["knots"] = False
s.show_options["resolutions"] = [100, 30]
# s.show_options["control_points"] = False
curv.show_options["control_points"] = False
s.cps[:] = x[to_s]

tic.summary(print_=True)


def move():
    if i < 200:
        curv.cps[:] -= [0, 0.005]
    else:
        curv.cps[:] -= [0.04, 0]

    scene.plant_kd_tree(1001, 4)


def sol():
    le.update_contact_lagrange()
    le.fixed_point_solve2()


def c_sol():
    le.fixed_point_solve2()
    print(ni.gap_norm())


def adv():
    le.advance_time2()


def show():
    s.cps[:] = x[to_s]
    gus.show(
        [
            str(i),
            s,
            curv,
        ],
        vedoplot=plt,
        interactive=False,
    )


# initialize a plotter
plt = gus.show([s, curv], interactive=False, close=False)
for i in range(400):
    move()
    # le.fixed_point_alm_solve2(10, 3, 10, 0, 1e-8, 1e-5, 1e-5, True)
    # le.advance_time2()
    le.step_time2()
    tic.toc(f"{i}-step")
    show()


tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
