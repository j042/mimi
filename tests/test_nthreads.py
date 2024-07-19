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


def balken_plasticity(subd, order, mat_f, nthreads):
    nl = balken(subd, order)
    mat = mat_f()
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

    bc = mimi.BoundaryConditions()
    bc.initial.dirichlet(2, 0).dirichlet(2, 1)
    bc.initial.body_force(1, -3)

    nl.boundary_condition = bc

    nl.setup(nthreads)
    nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, False)

    nl.time_step_size = 0.5

    return nl, nl.solution_view("displacement", "x").ravel()


def balken_elasticity(subd, order, mat_f, nthreads):
    nl = balken(subd, order)

    mat = mat_f()

    mat.density = 1
    mat.viscosity = -1
    mat.set_young_poisson(2100, 0.3)

    nl.set_material(mat)

    bc = mimi.BoundaryConditions()
    bc.initial.dirichlet(2, 0).dirichlet(2, 1)
    bc.initial.body_force(1, -5)

    nl.boundary_condition = bc

    nl.setup(nthreads)

    nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, False)
    nl.time_step_size = 0.05

    # load reference and compare
    u = nl.solution_view("displacement", "x").ravel()

    return nl, u


def compare_list_of_nthreads(case_name, f, params, list_of_nt):
    cases = []
    for nt in list_of_nt:
        cases.append(f(*params, nt))

    # loop
    for _ in range(10):  # time steps
        ref = cases[0][1]
        cases[0][0].step_time2()
        for c in cases[1:]:
            c[0].step_time2()
            assert np.allclose(ref, c[1]), f"{case_name} failed."


def test_nonlinear_solid_stvk():
    compare_list_of_nthreads(
        "stvk", balken_elasticity, [1, 2, mimi.StVenantKirchhoff], [1, 2, 3, 4]
    )


def test_nonlinear_solid_stvk():
    compare_list_of_nthreads(
        "neohook",
        balken_elasticity,
        [1, 2, mimi.CompressibleOgdenNeoHookean],
        [1, 2, 3, 4],
    )


def test_nonlinear_solid_j2():
    compare_list_of_nthreads(
        "j2", balken_plasticity, [1, 2, mimi.J2], [1, 2, 3, 4]
    )


def test_nonlinear_solid_j2_simo():
    compare_list_of_nthreads(
        "j2_simo", balken_plasticity, [1, 2, mimi.J2Simo], [1, 2, 3, 4]
    )


def test_nonlinear_solid_j2_log():
    compare_list_of_nthreads(
        "j2_log", balken_plasticity, [1, 2, mimi.J2Log], [1, 2, 3, 4]
    )
