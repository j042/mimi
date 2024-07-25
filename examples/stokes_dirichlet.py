import splinepy as sp
import mimi
import gustaf as gus

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

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(0, 0).dirichlet(0, 1).dirichlet(1, 0).dirichlet(
    1, 1
).dirichlet(2, 0)
# bc.initial.traction(2, 0, 1)

nl.boundary_condition = bc

nl.setup(4)
nl.configure_newton("stokes", 1e-12, 1e-8, 10, True)

v = nl.solution_view("velocity", "v").reshape(-1, 2)
p = nl.solution_view("pressure", "p")

ps, pto_m, pto_s = mimi.to_splinepy(nl)
s, to_m, to_s = mimi.to_splinepy(nl.vel_nurbs())

# parabolic profile assignemnt to control points
# better way is to actually assemble
m_coord = s.cps[to_m].ravel()
lx_dofs = nl.boundary_dof_ids("velocity", 2, 0)
ly_dofs = nl.boundary_dof_ids("velocity", 2, 1)

nl.solution_view("velocity", "v")[lx_dofs] = m_coord[ly_dofs] * (
    1 - m_coord[ly_dofs]
)

nl.static_solve()


s2 = s.copy()
vs = s.copy()
vs.cps = v[to_s]
s.spline_data["v"] = vs
s.show_options(arrow_data="v", control_points=False)

ps.cps = p.reshape(-1, 1)[pto_s]
s2.spline_data["p"] = ps
s2.show_options(data="p", control_points=False)

sp.show(s, s2)
