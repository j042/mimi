import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 2

# init, read mesh
le = mimi.NonlinearViscoSolid()
le.read_mesh("tests/data/sqn.mesh")

# refine
le.elevate_degrees(1)
le.subdivide(4)

# mat
mat = mimi.J2AdiabaticViscoIsotropicHardening()
mat.density = 7800
mat.viscosity = -1
mat.set_young_poisson(205.0e9, 0.29)
mat.heat_fraction = 0.9
mat.specific_heat = 450
mat.initial_temperature = 50
hardening = mimi.JohnsonCookThermoViscoHardening()
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
to_m = np.array(to_m)
to_s = np.array(to_s)
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

scene = mimi.NearestDistanceToSplines()
scene.add_spline(curv)
scene.plant_kd_tree(100000, 4)
scene.coefficient = 0.5e11

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(3, 0)
bc.initial.traction(0, 1, 1e9)
bc.current.contact(1, scene)
le.boundary_condition = bc

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("nonlinear_visco_solid", 1e-14, 1e-8, 20, False)

# set step size
le.time_step_size = 0.0001

# get view of solution, displacement
x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())
v = le.solution_view("displacement", "x_dot").reshape(-1, le.mesh_dim())
x_ref = le.solution_view("displacement", "x_ref").reshape(-1, le.mesh_dim())

mi = s.multi_index
b2 = to_s[mi[0, :]]
b3 = to_s[mi[-1, :]]

# set visualization options
s.show_options["resolutions"] = [100, 50]
s.show_options["control_points"] = False
curv.show_options["control_points"] = False
s.cps[:] = x[to_s]


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
            str(i) + " " + str(j) + " " + str(ab) + " " + str(ni.gap_norm()),
            # str(i) + " " + str(ni.gap_norm()),
            s,
            curv,
        ],
        vedoplot=plt,
        interactive=False,
    )


def bc():
    x[b2, 0] = x_ref[b2, 0]
    v[b2, 0] = 0.0
    x[b3, 0] = x_ref[b3, 0]
    v[b3, 0] = 0.0


coe = 1e11
# initialize a plotter
plt = gus.show(
    [s, curv],
    close=False,
)  # offscreen=True)
n = le.nonlinear_from2("contact")
ni = n.boundary_integrator(0)
for i in range(2000):
    bc()
    old = 1
    b_old = 1
    scene.coefficient = coe
    for j in range(20):
        sol()
        le.configure_newton("nonlinear_visco_solid", 1e-6, 1e-8, 3, True)
        rel, ab = le.newton_final_norms("nonlinear_visco_solid")
        bdr_norm = np.linalg.norm(n.boundary_residual())
        print("augumenting")
        print()
        if ni.gap_norm() < 1e-4:
            print(ni.gap_norm(), "exit!")
            break
    print("final solve!")
    le.configure_newton("nonlinear_visco_solid", 1e-7, 1e-8, 20, True)
    le.update_contact_lagrange()
    scene.coefficient = 0.0
    c_sol()
    rel, ab = le.newton_final_norms("nonlinear_visco_solid")

    le.configure_newton("nonlinear_visco_solid", 1e-8, 1e-10, 3, False)
    scene.coefficient = coe
    adv()
    show()
    plt.screenshot(f"pl/{10000+i}.png")

gus.show(s, vedoplot=plt, interactive=True)
