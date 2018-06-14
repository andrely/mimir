from numpy.ma import array, vstack
from numpy.random.mtrand import normal
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing.data import Normalizer, StandardScaler

from mimir import logistic
from mimir import maxent
from mimir.data import mlclass
from mimir.data.iris import iris
from mimir.data.preprocessing import add_bias, binarize
from mimir.optimization.steepest import steepest_gd


def test_steepest():
    data = mlclass.ex4()
    x = add_bias(StandardScaler().fit_transform(data['x']))
    y = data['y']
    theta = array([.01, .01, .01])
    c = lambda theta: logistic.model.cost(x, y, theta)
    g = lambda theta: logistic.model.grad(x, y, theta)
    assert_array_almost_equal([-0.0254469, 1.14114, 1.21333], steepest_gd(c, g, theta, max_iter=500)[0], decimal=1)

    y = vstack([data['y'], 1 - data['y']]).T
    theta = array([[.01, .01, .01]])
    c = lambda theta: maxent.model.cost(x, y, theta)
    g = lambda theta: maxent.model.grad(x, y, theta)
    assert_array_almost_equal([[-0.0254469, 1.14114, 1.21333]], steepest_gd(c, g, theta, max_iter=500)[0], decimal=1)


def test_steepest_iris():
    data = iris()
    x = add_bias(StandardScaler().fit_transform(data['x']))
    y = binarize(data['y'])
    theta = normal(scale=.001, size=(y.shape[1] - 1) * x.shape[1])
    theta.shape = y.shape[1] - 1, x.shape[1]

    c = lambda theta: maxent.model.cost(x, y, theta, 1.)
    g = lambda theta: maxent.model.grad(x, y, theta, 1.)

    assert_array_almost_equal([[0.0425072, -1.76158, 1.40147, -2.77042, -2.63817],
                               [2.06606, -0.0531037, -0.120857, -1.19605, -2.26611]],
                              steepest_gd(c, g, theta, max_iter=500, rho=.5)[0], decimal=2)
