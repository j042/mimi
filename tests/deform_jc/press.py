import splinepy

import mimi

PENALTY_FACTOR = 0.5e12
N_THREADS = 4
DT = 0.1e-4

PENALTY_FACTOR = 0.1e8
N_THREADS = 1
DT = 1e-4

INTERACTIVE_SHOW = False


def to_K(cel):
    return cel + 273.15


def move(spl):
    spl.cps[:, 1] -= 20e-3 * DT
    scene.plant_kd_tree(1001, 4)


plt = None


def show(i):
    global plt
    spline.cps = x[to_s]
    plt = splinepy.show(
        [str(i), tool, spline],
        vedoplot=plt,
        interactive=INTERACTIVE_SHOW,
        close=False,
    )


tool = splinepy.Bezier([2], [[-5e-4, 4e-3], [2e-3, 3e-3], [4.5e-3, 4e-3]])
tool.cps += [0, 0.0005]
# tool = splinepy.Bezier([1], [[-2e-3, 4e-3], [6e-3, 4e-3]])
# tool = splinepy.Bezier([1], [[-1e-2, 4.1e-3], [14e-3, 4.1e-3]])

nl = mimi.PyNonlinearViscoSolid()
nl.read_mesh("box.mesh")
nl.elevate_degrees(1)
nl.subdivide(3)

mat = mimi.PyJ2AdiabaticViscoIsotropicHardening()
mat.density = 7800
mat.viscosity = 10
mat.set_young_poisson(205e9, 0.29)
# mat.set_young_poisson(0, 0)
mat.heat_fraction = 0.9
mat.specific_heat = 450
mat.initial_temperature = to_K(20)
mat.hardening = mimi.PyJohnsonCookThermoViscoHardening()
mat.hardening.A = 288e6
mat.hardening.B = 695e6
mat.hardening.C = 0.034
mat.hardening.n = 0.2835
mat.hardening.m = 1.3558
mat.hardening.eps0_dot = 0.004
mat.hardening.reference_temperature = to_K(20)
mat.hardening.melting_temperature = to_K(1500)

# mat = mimi.PyStVenantKirchhoff()
# mat.density = 7800
# mat.viscosity = -1
# mat.set_young_poisson(205e9, 0.29)

nl.set_material(mat)

scene = mimi.PyNearestDistanceToSplines()
scene.add_spline(tool)
scene.plant_kd_tree(1001, 4)
scene.coefficient = PENALTY_FACTOR

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 1).dirichlet(0, 0)
bc.current.contact(1, scene)

nl.boundary_condition = bc

nl.setup(N_THREADS)
nl.time_step_size = DT

spline, to_m, to_s = mimi.to_splinepy(nl)
spline.show_options["control_points"] = False
spline.show_options["axes"] = True
x = nl.solution_view("displacement", "x").reshape(-1, 2)
nl.configure_newton("nonlinear_visco_solid", 1e-16, 1e-10, 20, False)

show(-1)
for i in range(5000):
    move(tool)
    # nl.step_time2()
    nl.fixed_point_alm_solve2(20, 2, 15, 0, 1e-14, 1e-8, 1e-5, False)
    nl.advance_time2()
    if i % 5 == 0:
        show(i)
