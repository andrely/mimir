from numpy.ma import array, vstack
from numpy.random.mtrand import normal
from numpy.testing import assert_array_almost_equal

from mimir import logistic, maxent
from mimir.data import mlclass
from mimir.data.iris import iris
from mimir.data.preprocessing import add_bias, binarize
from mimir.optimization.newton import newton


def test_newton():
    data = mlclass.ex4()
    x = add_bias(data['x'])
    y = data['y']
    theta = array([.01, .01, .01])
    c = lambda theta: logistic.model.cost(x, y, theta, 0.)
    g = lambda theta: logistic.model.grad(x, y, theta, 0.)
    h = lambda theta: logistic.model.hessian(x, theta, 0.)
    assert_array_almost_equal([-16.3787, 0.1483, 0.1589], newton(c, g, h, theta)[0], decimal=3)

    y = vstack([data['y'], 1 - data['y']]).T
    theta = array([[.01, .01, .01]])
    c = lambda theta: maxent.model.cost(x, y, theta, 0.)
    g = lambda theta: maxent.model.grad(x, y, theta, 0.)
    h = lambda theta: maxent.model.hessian(x, theta, 0.)
    assert_array_almost_equal([[-16.3787, 0.1483, 0.1589]], newton(c, g, h, theta)[0], decimal=3)


def test_newton_iris():
    data = iris()
    x = add_bias(data['x'])
    y = binarize(data['y'])
    theta = normal(scale=.001, size=(y.shape[1] - 1)*x.shape[1])
    theta.shape = y.shape[1] - 1, x.shape[1]

    c = lambda theta: maxent.model.cost(x, y, theta, 1.)
    g = lambda theta: maxent.model.grad(x, y, theta, 1.)
    h = lambda theta: maxent.model.hessian(x, theta, 1.)

    assert_array_almost_equal([[17.8988, -0.783738, 1.24289, -3.87904, -1.65902],
                               [11.7486, 0.260549, -0.33588, -1.83314, -2.06362]],
                              newton(c, g, h, theta)[0], decimal=3)