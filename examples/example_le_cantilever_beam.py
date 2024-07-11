import mimi
import gustaf as gus
import numpy as np

mesh = gus.Volumes(
    vertices=[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    volumes=[
        [0, 2, 7, 3],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [5, 0, 4, 7],
        [5, 0, 7, 1],
        [7, 0, 3, 1],
    ],
)

tets = mesh.volumes
verts = mesh.vertices

faces = mesh.to_faces(False)
boundary_faces = faces.single_faces()

BC = {1: [], 2: [], 3: []}
for i in boundary_faces:
    # mark boundaries at x = 0 with 1
    if np.max(verts[faces.const_faces[i], 0]) < 0.1:
        BC[1].append(i)
    # mark boundaries at x = 1 with 2
    elif np.min(verts[faces.const_faces[i], 0]) > 0.9:
        BC[2].append(i)
    # mark rest of the boundaries with 3
    else:
        BC[3].append(i)

mesh.BC = BC

gus.io.mfem.export("test_mesh.mesh", mesh)

ex2 = mimi.LECantileverBeam("test_mesh.mesh", "test", "Example2")
ex2.solve()
print(f"Volume: {ex2.volume} \nCompliance: {ex2.compliance}")