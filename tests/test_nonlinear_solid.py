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


def balken_plasticity(subd, order, mat):
    nl = balken(subd, order)

    mat.density = 1
    mat.viscosity = -1
    mat.melting_temperature = 1500
    mat.initial_temperature = 20
    mat.specific_heat = 450
    mat.heat_fraction = 0.9
    mat.set_young_poisson(2100, 0.3)
    mat.hardening = mimi.JohnsonCookTemperatureAndRateDependentHardening()
    mat.hardening.A = 70
    mat.hardening.B = 140
    mat.hardening.n = 0.2835
    mat.hardening.m = 1.3558
    mat.hardening.eps0_dot = 0.004
    mat.hardening.reference_temperature = 20

    nl.set_material(mat)

    rc = mimi.RuntimeCommunication()
    rc.set_real("ode_coefficient", 0.5)

    nl.runtime_communication = rc

    bc = mimi.BoundaryConditions()
    bc.initial.dirichlet(2, 0).dirichlet(2, 1)
    bc.initial.body_force(1, -3)

    nl.boundary_condition = bc

    nl.setup(1)
    nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, False)

    nl.time_step_size = 0.5

    return nl, nl.solution_view("displacement", "x").ravel()


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


def test_nonlinear_solid_j2():
    nl, u = balken_plasticity(1, 2, mimi.J2())
    for i in range(10):
        nl.step_time2()
        ref = np.genfromtxt(f"tests/data/ref/j2_h1_p2/x_{i}.txt")
        assert np.allclose(u, ref)


def test_nonlinear_solid_j2_simo():
    nl, u = balken_plasticity(1, 2, mimi.J2Simo())
    for i in range(10):
        nl.step_time2()
        ref = np.genfromtxt(f"tests/data/ref/j2_simo_h1_p2/x_{i}.txt")
        assert np.allclose(u, ref)


def test_nonlinear_solid_j2_log():
    nl, u = balken_plasticity(1, 2, mimi.J2Log())
    for i in range(10):
        nl.step_time2()
        ref = np.genfromtxt(f"tests/data/ref/j2_log_h1_p2/x_{i}.txt")
        assert np.allclose(u, ref)
