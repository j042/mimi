import os
import argparse

import splinepy as spp
import mimi
import numpy as np
import gustaf as gus

SHOW = False

parse = argparse.ArgumentParser(description="FeatFlow (Turek) CSM benchmark")
parse.add_argument("-H", dest="H")
parse.add_argument("-P", dest="P")
parse.add_argument("-T", dest="T")
parse.add_argument("-NT", dest="NT")
args = parse.parse_args()

H = int(args.H)
P = int(args.P)
T = int(args.T)
nthreads = int(args.NT)

STEPSIZE = 0.02 / (2 ** (T - 1))
N_TIMESTEP = int(10 / STEPSIZE)
EXPORT_PATH = f"H{H}P{P}T{T}"
A_FILE = "a.txt"
if not os.path.isdir(EXPORT_PATH):
    os.mkdir(EXPORT_PATH)

if os.path.isfile(os.path.join(EXPORT_PATH, A_FILE)):
    os.remove(os.path.join(EXPORT_PATH, A_FILE))

# create nl solid
nl = mimi.NonlinearSolid()

# load mesh
nl.read_mesh("tail.mesh")

nl.elevate_degrees(P)
nl.subdivide(H)
nl.time_step_size = STEPSIZE


# create material
mat = mimi.StVenantKirchhoff()
mat.density = 1000.0
mat.viscosity = -1  # no damping
mat.set_young_poisson(1.4e6, 0.4)

nl.set_material(mat)

# set bc
bc = mimi.BoundaryConditions()
bc.initial.dirichlet(2, 0).dirichlet(2, 1)  # fix left
bc.initial.body_force(1, -2000)  # g = 2

nl.boundary_condition = bc

nl.setup(nthreads)  # multithreading slows down here
nl.configure_newton("nonlinear_solid", 1e-12, 1e-8, 20, False)


# setup visualization and io
s = spp.NURBS(**nl.nurbs())
to_m, to_s = spp.io.mfem.dof_mapping(s)
s.cps[:] = s.cps[to_s]
s.show_options["resolutions"] = 50
s.show_options["control_points"] = False

x = nl.solution_view("displacement", "x").reshape(-1, nl.mesh_dim())

# compute reference once
ref = s.evaluate([[1, 0.5]])
print(ref)

cam = dict(
    position=(0.425000, 0.200000, 0.677251),
    focal_point=(0.425000, 0.200000, 0),
    viewup=(0, 1.00000, 0),
    roll=0,
    distance=0.677251,
    clipping_range=(0.634366, 0.732868),
)


# run
if SHOW:
    plt = spp.show(s, close=False)
for i in range(N_TIMESTEP):
    s.cps[:] = x[to_s]
    diff = (s.evaluate([[1, 0.5]]) - ref).ravel()
    if SHOW:
        spp.show(
            [str(i) + " " + str(s.evaluate([[1, 0.5]]) - ref), s],
            vedoplot=plt,
            interactive=False,
            cam=cam,
        )
    tic = gus.utils.tictoc.Tic()
    nl.step_time2()
    tic.toc()
    tic.summary(print_=True)

    np.savez(os.path.join(EXPORT_PATH, f"{10000 + i}.npz"), x=x)
    with open(os.path.join(EXPORT_PATH, A_FILE), "a") as f:
        f.write(" ".join([str(diff[0]), str(diff[1])]))
        f.write("\n")


if SHOW:
    spp.show(s, vedoplot=plt, interactive=True)
