import numpy as np
import pytest

from larvaworld.lib import aux


def test_angular_funcs():
    p1 = (-1, -1)
    pmid = (0, 0)
    p2 = (-1, 1)
    p3 = (1, 1)
    a1 = 30
    a2 = 45

    pps = aux.rotate_points_around_point([p1, p2], np.pi / 2, pmid)
    assert pytest.approx(pps[0]) == p2
    assert pytest.approx(pps[1]) == p3
