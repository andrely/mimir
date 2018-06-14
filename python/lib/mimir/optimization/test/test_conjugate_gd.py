from numpy.random.mtrand import normal
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler

from mimir import maxent
from mimir.data.iris import iris
from mimir.data.preprocessing import add_bias, binarize
from mimir.optimization.conjugate_gd import conjugate_gd_fr


def test_cgd_fr_iris():
    data = iris()
    sc = StandardScaler().fit(data['x'])
    x = add_bias(sc.transform(data['x']))
    y = binarize(data['y'])
    theta = normal(scale=.001, size=(y.shape[1] - 1) * x.shape[1])
    theta.shape = y.shape[1] - 1, x.shape[1]

    c = lambda theta: maxent.model.cost(x, y, theta, 1.)
    g = lambda theta: maxent.model.grad(x, y, theta, 1.)

    assert_array_almost_equal([[0.5157, -1.6937, 1.5391, -2.9251, -2.7841],
                               [2.3700, -0.0502, -0.0128, -1.4165, -2.3897]],
                              conjugate_gd_fr(c, g, theta, max_iter=50)[0], decimal=2)