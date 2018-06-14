from math import isnan

import sys
from nose.tools import assert_equal
from numpy.ma import array
from numpy.ma.testutils import assert_array_almost_equal
from numpy.testing.utils import assert_approx_equal

from mimir.data import mlclass
from mimir.data.preprocessing import add_bias
from mimir.logistic.model import cost, grad
from mimir.optimization.one_dim import sign, bracket, nan_guard, golden_section, line_search


def test_sign():
    assert_approx_equal(sign(.8, .4), .8)
    assert_approx_equal(sign(.4, -.8), -.4)


def test_bracket():
    data = mlclass.ex4()
    x = add_bias(data['x'])
    y = data['y']
    theta = array((.01, .01, .01))
    p = -grad(x, y, theta, 0.)
    c = lambda alpha: nan_guard(cost(x, y, theta + alpha*p, 0.))

    assert_array_almost_equal((-1.61803, 0, 1, 880.894, 0.778512, 651.614), bracket(0., 1., c), decimal=3)


def test_golden_section():
    data = mlclass.ex4()
    x = add_bias(data['x'])
    y = data['y']
    theta = array((.01, .01, .01))
    p = -grad(x, y, theta, 0.)
    c = lambda alpha: nan_guard(cost(x, y, theta + alpha*p, 0.))
    assert_array_almost_equal((0.000739624, 0.685489), golden_section((-1.61803, 0, 1, 880.894, 0.778512, 651.614), c))


def test_line_search():
    data = mlclass.ex4()
    x = add_bias(data['x'])
    y = data['y']
    theta = array((.01, .01, .01))
    p = -grad(x, y, theta, 0.)
    c = lambda alpha: cost(x, y, theta + alpha*p, 0.)
    assert_array_almost_equal((0.000739624, 0.685489), line_search(0., 1., c))