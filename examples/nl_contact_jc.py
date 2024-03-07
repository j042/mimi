import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.PyNonlinearViscoSolid()
le.read_mesh("tests/data/sqn.mesh")

# set param

# refine
le.elevate_degrees(1)
le.subdivide(3)

# mat
mat = mimi.PyJ2AdiabaticViscoIsotropicHardening()
mat.density = 7800
mat.viscosity = -1
mat.set_young_poisson(205.0e9, 0.29)
mat.heat_fraction = 0.9
mat.specific_heat = 450
mat.initial_temperature = 50
hardening = mimi.PyJohnsonCookThermoViscoHardening()
hardening.A = 288e6
hardening.B = 695e6
hardening.C = 0.034
hardening.n = 0.034
hardening.m = 1.3558
hardening.eps0_dot = 0.004
hardening.reference_temperature = 20
hardening.melting_temperature = 1500
mat.hardening = hardening

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
curv.cps[:] += [0.0, 0.75]


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

le.configure_newton("nonlinear_visco_solid", 1e-14, 1e-8, 20, False, True)

tic.toc("bilinear, linear forms assembly")

# set step size
le.time_step_size = 0.001

# get view of solution, displacement
x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())

tic.summary(print_=True)
# set visualization options
# s.show_options["control_points"] = False
# s.show_options["knots"] = False
s.show_options["resolutions"] = [100, 50]
s.show_options["control_points"] = False
curv.show_options["control_points"] = False
s.cps[:] = x[to_s]

tic.summary(print_=True)


def move():
    if i < 100:
        curv.cps[:] -= [0, 0.005]
    else:
        curv.cps[:] -= [0.04, 0]

    scene.plant_kd_tree(1000, 4)


def sol():
    le.update_contact_lagrange()
    le.fixed_point_solve2()


def c_sol():
    le.fixed_point_solve2()
    print(ni.gap_norm())


def adv():
    le.advance_time2()
    le.fill_contact_lagrange(0)


def show():
    s.cps[:] = x[to_s]
    gus.show(
        [
            # str(i) + " " + str(j) + " " + str(ab) + " " + str(ni.gap_norm()),
            str(i) + " " + str(ni.gap_norm()),
            s,
            curv,
        ],
        vedoplot=plt,
        interactive=False,
    )


coe = 5e11
# initialize a plotter
plt = gus.show(
    [s, curv],
    close=False,
)  # offscreen=True)
n = le.nonlinear_from2("contact")
ni = n.boundary_integrator(0)
for i in range(2000):
    if i < 820:
        move()
    old = 1
    b_old = 1
    scene.coefficient = coe
    for j in range(20):
        sol()
        le.configure_newton(
            "nonlinear_visco_solid", 1e-6, 1e-8, 5, True, False
        )
        rel, ab = le.newton_final_norms("nonlinear_visco_solid")
        bdr_norm = np.linalg.norm(n.boundary_residual())
        print("augumenting")
        print()
        if ni.gap_norm() < 1e-4:
            print(ni.gap_norm(), "exit!")
            break
    print("final solve!")
    le.configure_newton("nonlinear_visco_solid", 1e-7, 1e-8, 20, True, True)
    le.update_contact_lagrange()
    scene.coefficient = 0.0
    c_sol()
    rel, ab = le.newton_final_norms("nonlinear_visco_solid")

    le.configure_newton("nonlinear_visco_solid", 1e-8, 1e-10, 3, False, False)
    scene.coefficient = coe
    adv()
    show()
    plt.screenshot(f"pl/{10000+i}.png")

tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
