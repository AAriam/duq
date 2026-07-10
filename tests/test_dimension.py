from duq.dimension import Dimension


def test_equality():
    assert Dimension("F") == Dimension("M.L.T^-2")
    assert Dimension("E") == Dimension("M.L^2.T^-2")
