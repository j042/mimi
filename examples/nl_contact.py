import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.PyNonlinearSolid()
le.read_mesh("tests/data/square-nurbs.mesh")

# set param

# refine
le.elevate_degrees(1)
le.subdivide(3)

# mat
mat = mimi.PyCompressibleOgdenNeoHookean()
# mat = mimi.PyJ2()
mat.density = 7e4
mat.viscosity = -1
mat.set_young_poisson(1e10, 0.3)
# mat.isotropic_hardening = 0
# mat.kinematic_hardening = 0
# mat.sigma_y = 1e4
le.set_material(mat)
# create splinepy partner
s = sp.NURBS(**le.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

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

scene = mimi.PyNearestDistanceToSplines()
scene.add_spline(curv)
scene.plant_kd_tree(100000, 4)
scene.coefficient = 0.5e11

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1)
bc.current.contact(1, scene)
le.boundary_condition = bc

tic.toc()

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("nonlinear_solid", 1e-14, 1e-8, 20, False)

tic.toc("bilinear, linear forms assembly")

# set step size
le.time_step_size = 0.005

# get view of solution, displacement
x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())

tic.summary(print_=True)
# set visualization options
s.show_options["control_points"] = False
# s.show_options["knots"] = False
s.show_options["resolutions"] = [100, 30]
s.show_options["control_points"] = False
s.cps[:] = x[to_s]

tic.summary(print_=True)


def move():
    if i < 100:
        curv.cps[:] -= [0, 0.005]
    else:
        curv.cps[:] -= [0.005, 0]
    scene.plant_kd_tree(10000, 4)


def sol():
    le.update_contact_lagrange()
    le.fixed_point_solve2()


def adv():
    le.fill_contact_lagrange(0)
    le.advance_time2()


def show():
    s.cps[:] = x[to_s]
    gus.show(
        [str(i), s, curv],
        vedoplot=plt,
        interactive=False,
    )


# initialize a plotter
plt = gus.show([s, curv], close=False)
n = le.nonlinear_from2("contact")
for i in range(1000):
    move()
    old = 1
    b_old = 1
    for j in range(50):
        sol()
        le.configure_newton("nonlinear_solid", 1e-8, 1e-10, 3, True)
        rel, ab = le.newton_final_norms("nonlinear_solid")
        bdr_norm = np.linalg.norm(n.boundary_residual())
        bdr_diff = b_old - bdr_norm
        bdr_rel = b_old / bdr_norm
        b_old = bdr_norm
        print("augumenting")
        print(rel, bdr_rel, bdr_diff, bdr_norm)
        print()
        if abs(bdr_diff) < 1e-5:
            break
    print("final solve!")
    sol()
    le.configure_newton("nonlinear_solid", 1e-8, 1e-10, 3, False)
    adv()
    show()


tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
