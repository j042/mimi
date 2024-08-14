"""examples/stokes_dirichlet.py

Poiseuille flow
"""

import splinepy as sp
import mimi


def apply_dirichlet_dofs(nl):
    def parabolic_function(points):
        x = points[:, 0]
        y = points[:, 1]
        return y * (1 - y)

    s, _, to_s = mimi.to_splinepy(nl.vel_nurbs())

    # Currently not available in main splinepy branch
    fi = sp.helpme.integrate.FieldIntegrator(s)

    # Get values for boundary dofs
    velx_rhs, boundary_dofs = fi.apply_dirichlet_boundary_conditions(
        function=parabolic_function, west=True, return_values=True
    )

    # Apply parabolic inflow function to x-velocity on left inlet
    # For DoFs of y-velocity just add +1
    boundary_dofs_m = 2 * to_s[boundary_dofs]

    nl.solution_view("velocity", "v")[boundary_dofs_m] = velx_rhs


sp.settings.NTHREADS = 4

# create nl solid
nl = mimi.Stokes()
nl.read_mesh("tests/data/balken.mesh")

# refine
nl.elevate_degrees(2)
nl.subdivide(5)

# create material
mat = mimi.FluidMaterial()
mat.density = 1
mat.viscosity = 1

# define material properties (young's modulus, poisson's ratio)
nl.set_material(mat)

rc = mimi.RuntimeCommunication()
rc.set_int("use_iterative_solver", 0)
nl.runtime_communication = rc

# Apply Dirichlet BCs everywhere but on the outlet
bc = mimi.BoundaryConditions()
dirichlet_bids = [0, 1, 2]
for bid in dirichlet_bids:
    bc.initial.dirichlet(bid, 0).dirichlet(bid, 1)

nl.boundary_condition = bc

nl.setup(4)
nl.configure_newton("stokes", 1e-12, 1e-8, 10, True)

apply_dirichlet_dofs(nl)

nl.static_solve()

# Plotting
common_show_options = {
    "control_points": False,
    "cmap": "jet",
    "scalarbar": True,
    "lighting": "off",
    "control_points": False,
    "knots": False,
}

v = nl.solution_view("velocity", "v").reshape(-1, 2)
p = nl.solution_view("pressure", "p")

ps, pto_m, pto_s = mimi.to_splinepy(nl)
s, to_m, to_s = mimi.to_splinepy(nl.vel_nurbs())
s2 = s.copy()
vs = s.copy()
vs.cps = v[to_s, :]

s.spline_data["v"] = vs
s.show_options(
    arrow_data="v", arrow_data_scale=1, control_points=False, knots=False
)

ps.cps = p.reshape(-1, 1)[pto_s]
s2.spline_data["p"] = ps
s2.show_options(data="p", **common_show_options)

sp.show([s, "Velocity"], [s2, "Pressure"])
