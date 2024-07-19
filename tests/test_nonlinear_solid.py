import numpy as np

import mimi


def balken(subd, order):
    nl = mimi.NonlinearSolid()
    nl.read_mesh("tests/data/balken.mesh")
    if order > 0:
        nl.elevate_degrees(order)
    if subd > 0:
        nl.subdivide(subd)
    return nl


def test_nonlinear_solid_neohook():
    # create nl solid
    nl = balken(1, 2)

    # create material
    mat = mimi.CompressibleOgdenNeoHookean()
    mat.density = 1
    mat.viscosity = -1
    mat.set_young_poisson(2100, 0.3)

    nl.set_material(mat)

    rc = mimi.RuntimeCommunication()
    rc.set_real("ode_coefficient", 0.5)

    nl.runtime_communication = rc

    bc = mimi.BoundaryConditions()
    bc.initial.dirichlet(2, 0).dirichlet(2, 1)
    bc.initial.body_force(1, -5)

    nl.boundary_condition = bc

    nl.setup(1)

    nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, False)
    nl.time_step_size = 0.05

    # load reference and compare
    u = nl.solution_view("displacement", "x").ravel()
    for i in range(10):
        nl.step_time2()
        ref = np.genfromtxt(f"tests/data/ref/neohook_h1_p2/x_{i}.txt")
        assert np.allclose(u, ref)
