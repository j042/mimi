import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 2

tool = sp.Bezier([2], [[2.5, 2], [1.22, -0.5], [3, 0.5]])
tool.cps[:] += [0.05, 0.3]

# create solid
nl = mimi.PyNonlinearSolid()

# read mesh
nl.read_mesh("tests/data/macro.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(3)

n_threads = 4
# we don't refine here
# create material
mat = mimi.PyCompressibleOgdenNeoHookean()
mat.density = 7800
mat.viscosity = -1  # maybe some higher value?
mat.set_young_poisson(205.0e9, 0.29)
nl.set_material(mat)

# create contact scene
scene = mimi.PyNearestDistanceToSplines()
scene.add_spline(tool)
scene.plant_kd_tree(10001, 4)
scene.coefficient = 1e13

# create BC
bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0)
bc.initial.dirichlet(0, 1)

bc.current.contact(3, scene)

nl.boundary_condition = bc
nl.setup(n_threads)

DT = 1e-4
nl.time_step_size = DT

# create splinepy nurbs to show
s = sp.NURBS(**nl.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]

nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 40, True, False)
n = nl.nonlinear_from2("contact")
ni = n.boundary_integrator(0)
x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())


def move():
    tool.cps[:] -= [DT * 5, 0]
    scene.plant_kd_tree(10001, 4)


def sol():
    nl.update_contact_lagrange()
    nl.fixed_point_solve2()


def c_sol():
    nl.fixed_point_solve2()
    # print(ni.gap_norm())


def adv():
    nl.advance_time2()
    nl.fill_contact_lagrange(0)


def show():
    s.cps[:] = x[to_s]
    gus.show(
        [
            str(i),  # + " " + str(ni.gap_norm()),
            s,
            tool,
        ],
        vedoplot=plt,
        interactive=False,
    )


s.show_options["control_points"] = False
tool.show_options["control_points"] = False

coe = 1e12
# initialize a plotter
plt = gus.show(
    [s, tool],
    close=False,
)  # offscreen=True)

for i in range(10000):
    move()
    scene.coefficient = coe
    for j in range(20):
        sol()
        nl.configure_newton("nonlinear_solid", 1e-6, 1e-8, 5, True, False)
        print("augumenting")
        print()
        if ni.gap_norm() < 1e-6:
            print(ni.gap_norm(), "exit!")
            break
    print("final solve!")
    nl.configure_newton("nonlinear_solid", 1e-7, 1e-8, 20, True, True)
    nl.update_contact_lagrange()
    scene.coefficient = 0.0
    c_sol()
    rel, ab = nl.newton_final_norms("nonlinear_solid")

    nl.configure_newton("nonlinear_solid", 1e-8, 1e-10, 3, False, False)
    scene.coefficient = coe
    adv()
    show()


gus.show(s, vedoplot=plt, interactive=True)
