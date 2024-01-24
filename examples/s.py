import splinepy as sp
import mimi
import gustaf as gus
import numpy as np

sp.settings.NTHREADS = 4

# init, read mesh
le = mimi.PyNonlinearSolid()
le.read_mesh("tests/data/es.mesh")

# refine
le.elevate_degrees(1)
le.subdivide(3)

# mat
mat = mimi.PyCompressibleOgdenNeoHookean()
mat.density = 4000
mat.viscosity = 10000
mat.set_young_poisson(1e7, 0.3)
le.set_material(mat)

# create splinepy partner
s = sp.NURBS(**le.nurbs())
to_m, to_s = sp.io.mfem.dof_mapping(s)
to_m = np.array(to_m)
to_s = np.array(to_s)
s.cps[:] = s.cps[to_s]

print("fine")
outlineo = {
    "degrees": [2, 1],
    "control_points": [
        [0.5064575645756457, 0.2007200720072007],
        [0.5821033210332104, 0.2007200720072007],
        [0.6236162361623616, 0.23222322232223222],
        [0.6559040590405905, 0.2556255625562556],
        [0.6559040590405905, 0.2907290729072907],
        [0.6559040590405905, 0.33033303330333036],
        [0.6134686346863468, 0.35193519351935193],
        [0.5830258302583026, 0.36723672367236726],
        [0.45202952029520294, 0.38973897389738976],
        [0.2564575645756458, 0.42304230423042305],
        [0.18035055350553506, 0.4513951395139514],
        [0.10424354243542436, 0.47974797479747977],
        [0.05212177121771218, 0.5472547254725473],
        [0.0, 0.6147614761476148],
        [0.0, 0.7011701170117012],
        [0.0, 0.7956795679567957],
        [0.05627306273062731, 0.8640864086408641],
        [0.11254612546125461, 0.9324932493249325],
        [0.21125461254612546, 0.9662466246624662],
        [0.30996309963099633, 1.0],
        [0.47601476014760147, 1.0],
        [0.6512915129151291, 1.0],
        [0.7347785977859779, 0.9738973897389739],
        [0.8182656826568265, 0.9477947794779478],
        [0.8740774907749077, 0.8928892889288929],
        [0.4870848708487085, 0.0],
        [0.6771217712177122, 0.0],
        [0.7873616236162362, 0.04635463546354635],
        [0.8976014760147601, 0.0927092709270927],
        [0.9488007380073801, 0.171017101710171],
        [1.0, 0.24932493249324933],
        [1.0, 0.3321332133213321],
        [1.0, 0.414041404140414],
        [0.9515682656826568, 0.48244824482448245],
        [0.9031365313653137, 0.5508550855085509],
        [0.8118081180811808, 0.5877587758775877],
        [0.7204797047970479, 0.6246624662466247],
        [0.533210332103321, 0.6453645364536453],
        [0.4095940959409594, 0.6597659765976598],
        [0.3726937269372694, 0.6777677767776777],
        [0.33579335793357934, 0.6957695769576958],
        [0.33579335793357934, 0.7308730873087309],
        [0.33579335793357934, 0.7623762376237624],
        [0.36485239852398527, 0.7844284428442845],
        [0.39391143911439114, 0.8064806480648065],
        [0.4575645756457565, 0.8064806480648065],
        [0.5202952029520295, 0.8064806480648065],
        [0.5645756457564576, 0.7776777677767777],
        [0.5968634686346863, 0.7560756075607561],
        [0.6107011070110702, 0.7101710171017102],
    ],
    "knot_vectors": [
        [
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            4.0,
            4.0,
            5.0,
            5.0,
            6.0,
            6.0,
            7.0,
            7.0,
            8.0,
            8.0,
            9.0,
            9.0,
            10.0,
            10.0,
            11.0,
            11.0,
            12.0,
            12.0,
            12.0,
        ],
        [0.0, 0.0, 1.0, 1.0],
    ],
}
outline = sp.BSpline(**outlineo)  # this one will have negative jac
# outline.elevate_degrees([0, 0])
# outline.normalize_knot_vectors()
o, u = outline.extract.boundaries([2, 3])
u.cps[:] = u.cps[::-1].copy()
u = u.copy()
# u.show(control_points=True)
u.cps[24] -= 1
o.cps[0] += [-5, 0]

mi = s.multi_index
b3 = to_s[mi[-1, :]]


ns = 500
path = outline.extract.spline(1, [0.01, 0.99]).sample([ns, 2])
up = path[:ns]
down = path[ns:]
mid = np.linspace(down, up, len(b3))[1:-1]


print("fine")
# set bc
scene0 = mimi.PyNearestDistanceToSplines()
scene0.add_spline(o)
scene0.plant_kd_tree(1001, 4)
scene0.coefficient = 1e3
scene1 = mimi.PyNearestDistanceToSplines()
scene1.add_spline(u)
scene1.plant_kd_tree(1001, 4)
scene1.coefficient = 1e3

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(3, 0).dirichlet(3, 1)
bc.current.contact(0, scene1)
bc.current.contact(1, scene0)
le.boundary_condition = bc

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("nonlinear_solid", 1e-14, 1e-8, 20, False)

# set step size
le.time_step_size = 0.0003

# get view of solution, displacement
x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())

s.show_options["resolutions"] = [100, 30]
s.show_options["control_points"] = False
o.show_options["control_points"] = False
u.show_options["control_points"] = False
s.cps[:] = x[to_s]

cam = dict(
    position=(-0.251124, 0.293697, 5.26880),
    focal_point=(-0.251124, 0.293697, 0),
    viewup=(0, 1.00000, 0),
    roll=0,
    distance=5.26880,
    clipping_range=(4.93517, 5.70148),
)


def move():
    if i > int(ns - 1):
        return
    x[b3] = np.array([down[i], *[mm[i] for mm in mid], up[i]])
    return


def sol():
    le.update_contact_lagrange()
    le.fixed_point_solve2()


def c_sol():
    le.fixed_point_solve2()
    print(gn())


def adv():
    le.advance_time2()
    le.fill_contact_lagrange(0)


def show():
    s.cps[:] = x[to_s]
    gus.show(
        [str(i) + " " + str(j) + " " + str(ab) + " " + str(gn()), s, o, u],
        vedoplot=plt,
        interactive=False,
        cam=cam,
    )

    move()


coe = 1e9
le.fill_contact_lagrange(0)
# initialize a plotter
plt = gus.show([s, o, u], close=False)
n = le.nonlinear_from2("contact")
ni = n.boundary_integrator(0)
ni2 = n.boundary_integrator(1)


def gn():
    return ni.gap_norm() + ni2.gap_norm()


for i in range(5000):
    move()
    old = 1
    b_old = 1
    scene0.coefficient = coe
    scene1.coefficient = coe
    for j in range(10):
        sol()
        le.configure_newton("nonlinear_solid", 1e-6, 1e-8, 5, True)
        rel, ab = le.newton_final_norms("nonlinear_solid")
        bdr_norm = np.linalg.norm(n.boundary_residual())
        print("augumenting")
        print()
        if gn() < 1e-5:
            print(gn(), "exit!")
            break
    print("final solve!")
    le.configure_newton("nonlinear_solid", 1e-8, 1e-8, 20, True)
    le.update_contact_lagrange()
    scene0.coefficient = 0.0
    scene1.coefficient = 0.0
    c_sol()
    rel, ab = le.newton_final_norms("nonlinear_solid")

    le.configure_newton("nonlinear_solid", 1e-8, 1e-10, 3, False)
    scene0.coefficient = coe
    scene1.coefficient = coe

    adv()
    show()

gus.show(s, vedoplot=plt, interactive=True)
