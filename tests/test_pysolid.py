import mimi


def test_read_2d_mesh():
    s = mimi.PySolid()
    s.read_mesh("tests/data/square-nurbs.mesh")

    assert s.mesh_dim() == 2
    assert s.n_vertices() == 4
    assert s.n_elements() == 1
    assert s.n_boundary_elements() == 4
    assert s.n_subelements() == 4
    assert s.mesh_degrees() == [1, 1]

    # now order 3 mesh
    s.read_mesh("tests/data/square-nurbs-3.mesh")

    assert s.mesh_dim() == 2
    assert s.n_vertices() == 16
    assert s.n_elements() == 1
    assert s.n_boundary_elements() == 4
    assert s.n_subelements() == 4
    assert s.mesh_degrees() == [3, 3]


def test_read_3d_mesh():
    s = mimi.PySolid()
    s.read_mesh("tests/data/cube-nurbs.mesh")

    assert s.mesh_dim() == 3
    assert s.n_vertices() == 8
    assert s.n_elements() == 1
    assert s.n_boundary_elements() == 6
    assert s.n_subelements() == 6
    assert s.mesh_degrees() == [1, 1, 1]

    s.read_mesh("tests/data/cube-nurbs-3.mesh")

    assert s.mesh_dim() == 3
    assert s.n_vertices() == 64
    assert s.n_elements() == 1
    assert s.n_boundary_elements() == 6
    assert s.n_subelements() == 6
    assert s.mesh_degrees() == [3, 3, 3]


def test_subdivide():
    s = mimi.PySolid()
    s.read_mesh("tests/data/square-nurbs.mesh")

    s.subdivide(1)

    assert s.mesh_dim() == 2
    assert s.n_vertices() == 9
    assert s.n_elements() == 4
    assert s.n_boundary_elements() == 8
    assert s.n_subelements() == 12
    assert s.mesh_degrees() == [1, 1]

    s.read_mesh("tests/data/cube-nurbs.mesh")

    s.subdivide(1)

    assert s.mesh_dim() == 3
    assert s.n_vertices() == 27
    assert s.n_elements() == 8
    assert s.n_boundary_elements() == 24
    assert s.n_subelements() == 36
    assert s.mesh_degrees() == [1, 1, 1]


def test_elevate_degrees():
    s0 = mimi.PySolid()
    s1 = mimi.PySolid()

    s0.read_mesh("tests/data/square-nurbs.mesh")
    s1.read_mesh("tests/data/square-nurbs-3.mesh")

    def elevate_and_compare(first, second):
        first.elevate_degrees(2)

        assert first.mesh_dim() == second.mesh_dim()
        assert first.n_vertices() == second.n_vertices()
        assert first.n_elements() == second.n_elements()
        assert first.n_boundary_elements() == second.n_boundary_elements()
        assert first.n_subelements() == second.n_subelements()
        assert first.mesh_degrees() == second.mesh_degrees()

    elevate_and_compare(s0, s1)

    s0.read_mesh("tests/data/cube-nurbs.mesh")
    s1.read_mesh("tests/data/cube-nurbs-3.mesh")

    elevate_and_compare(s0, s1)
