import splinepy as sp
import mimi
import gustaf as gus

sp.settings.NTHREADS = 4

# create nl solid
nl = mimi.Stokes()
nl.read_mesh("tests/data/balken.mesh")
# refine
# nl.elevate_degrees(1)
# nl.subdivide(1)

# create material
mat = mimi.FluidMaterial()
mat.density = 1
mat.viscosity = 1

# define material properties (young's modulus, poisson's ratio)
nl.set_material(mat)

# create splinepy nurbs to show
# s, to_m, to_s = mimi.to_splinepy(nl)

bc = mimi.BoundaryConditions()
# bc.initial.dirichlet(2, 0)
bc.initial.body_force(1, 1)

nl.boundary_condition = bc

nl.setup(1)
nl.configure_newton("stokes", 1e-12, 1e-8, 10, False)

nl.static_solve()
