import numpy
from numpy.random import random
from numpy.random.mtrand import normal
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler

from mimir import maxent
from mimir.data.iris import iris
from mimir.data.preprocessing import add_bias, binarize
from mimir.optimization.stochastic_gd import sgd, MiniBatch


def test_sgd_iris():
    numpy.random.seed(1)

    data = iris()
    x = add_bias(StandardScaler().fit_transform(data['x']))
    y = binarize(data['y'])
    theta = normal(scale=.001, size=(y.shape[1] - 1) * x.shape[1])
    theta.shape = y.shape[1] - 1, x.shape[1]

    c = lambda theta: maxent.model.cost(x, y, theta, 1.)
    g = lambda theta, batch: maxent.model.grad(batch[0], batch[1], theta, 1.)
    b = MiniBatch(x, y, size=50)
    stats = {'method': 'sgd'}

    stats_ = sgd(c, g, b, theta, rho=.5, max_iter=100, stats=stats)[0]
    assert_array_almost_equal([[-0.27884785, -1.21284649,  0.88989122, -1.77701123, -1.68204016],
                               [ 1.01925233, -0.14850723, -0.41679621, -0.58286958, -1.29281991]],
                              stats_, decimal=3)

    numpy.random.seed()