import splinepy as sp
import mimi
import gustaf as gus
import numpy as np


sp.settings.NTHREADS = 4

tic = gus.utils.tictoc.Tic()

# init, read mesh
le = mimi.PyNonlinearSolid()
le.read_mesh("tests/data/square-nurbs.mesh")
young = 1e10
poisson = 0.3
lambda_ = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
mu = young / (2.0 * (1.0 + poisson))
# set param
mat = mimi.PyCompressibleOgdenNeoHookean()
mat.density = 5000e1
mat.viscosity = 10
mat.lambda_ = lambda_
mat.mu = mu
le.set_material(mat)
#le.set_parameters(1e9, 0.4, 1000000, 10000)

# refine
le.elevate_degrees(1)
le.subdivide(3)

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
curv.cps[:] += [0., .75]

def nor(spl, on):
    d=spl.derivative(on, [1])
    d = d[:,[1,0]]
    d[:, 1]  *= -1
    return d

#curv.spline_data["n"] = sp.SplineDataAdaptor(curv, function=nor)
#curv.show_options["data_name"] = "n"
#curv.show_options["arrow_data"] = "n"
#curv.show()

scene = mimi.PyNearestDistanceToSplines()
scene.add_spline(curv)
scene.plant_kd_tree(1000, 4)
scene.coefficient = 1e3


bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1)
bc.current.contact(1, scene)
#bc.current.contact(3, scene)
le.boundary_condition = bc

tic.toc()

# setup needs to be called this assembles bilinear forms, linear forms
le.setup(4)

le.configure_newton("nonlinear_solid", 1e-14, 1e-8, 15, False)

tic.toc("bilinear, linear forms assembly")

# set step size
le.time_step_size = 0.1
# get view of solution, displacement
x = le.solution_view("displacement", "x").reshape(-1, le.mesh_dim())
v = le.solution_view("displacement", "x_dot").reshape(-1, le.mesh_dim())

tic.summary(print_=True)
# set visualization options
s.show_options["control_point_ids"] = False
# s.show_options["knots"] = False
s.show_options["resolutions"] = [50,10]
s.show_options["control_points"] = False
s.cps[:] = x[to_s]

tic.summary(print_=True)
# initialize a plotter
plt = gus.show([s, curv], close=False)

def move():
#    if i < 50:
#        curv.cps[:] -= [0, 0.005]
    curv.cps[:] -= [0, 0.0015]
#    else:
#        curv.cps[:] -= [0.005, 0]
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
        [str(i) + " " + str(j), s, curv],
        vedoplot=plt,
        interactive=False,
    )

n = le.nonlinear_from2("contact")
t_x = x.copy()
t_v = v.copy()
for i in range(200):
    move()
    old = 1
    b_old = 1
    for j in range(100):
        print("augstep", j)
        if j == 0:
            scene.coefficient = 1e8
        elif j  == 5 or j == 10:
            scene.coefficient *= 2
           
        sol()
        #le.fixed_point_advance2(x, t_v)
        le.configure_newton("nonlinear_solid", 1e-14, 1e-8, 15, True)
        #le.fixed_point_advance2(t_x, t_v)
        rel, ab = le.newton_final_norms("nonlinear_solid")
        print()
        bdr_norm = np.linalg.norm(n.boundary_residual())
        bdr_diff = b_old - bdr_norm
        b_old = bdr_norm
        print(rel, rel / old, bdr_diff )
        if np.linalg.norm(n.boundary_residual()) < 1e-10:# or ab < 1e-10 or bdr_diff < 1:
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            print("no residual, next!")
            break
        if ab < 1e-10:
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            print("abs, next!")
            break
        if abs(bdr_diff) < 1e-5:
            print("bdr converge, next!")
            print("bdr converge, next!")
            print("bdr converge, next!")
            print("bdr converge, next!")
            print("bdr converge, next!")
            print("bdr converge, next!")
            print("bdr converge, next!")
            print("bdr converge, next!")
            break
        print()
        old = rel
    le.configure_newton("nonlinear_solid", 1e-14, 1e-8, 15, False)
    sol()
    adv()
    show()


tic.summary(print_=True)
gus.show(s, vedoplot=plt, interactive=True)
