import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

# create nl solid
nl = mimi.NonlinearSolid()
nl.read_mesh("tests/data/balken.mesh")
# refine
nl.elevate_degrees(1)
nl.subdivide(2)

# create material
mat = mimi.CompressibleOgdenNeoHookean()
mat.density = 1
mat.viscosity = -1

# define material properties (young's modulus, poisson's ratio)
mat.set_young_poisson(2100, 0.3)
nl.set_material(mat)

# create splinepy nurbs to show
s, to_m, to_s = mimi.to_splinepy(nl)
o_cps = s.cps.copy()

bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2, 1)
bc.initial.body_force(1, -5)

nl.boundary_condition = bc

nl.setup(2)
nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 10, False)

nl.time_step_size = 0.05

u = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())
s.show_options["control_point_ids"] = False
s.show_options["resolutions"] = 50
s.cps[:] = u[to_s] + o_cps

plt = gus.show(s, close=False)
for i in range(10000):
    if True:
        s.cps[:] = u[to_s] + o_cps
        gus.show(
            [str(i), s],
            vedoplot=plt,
            interactive=False,
        )
    nl.step_time2()

gus.show(s, vedoplot=plt, interactive=True)
