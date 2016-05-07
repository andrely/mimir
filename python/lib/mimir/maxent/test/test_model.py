from numpy import array, hstack, ones
from numpy.ma.testutils import assert_almost_equal, assert_equal
from numpy.random.mtrand import seed
from scipy.sparse import lil_matrix

from mimir.maxent.feature import Encoder
from mimir.maxent.model import MaxEntModel


def f1(x, y, i):
    return int(x[i] == 1 and y == 1)


def f0(x, y, i):
    return int(x[i] == 1 and y == 0)


def test_iis():
    inner_features = [lambda x, y: f1(x, y, 0), lambda x, y: f0(x, y, 0),
                      lambda x, y: f1(x, y, 1), lambda x, y: f0(x, y, 1),
                      lambda x, y: f1(x, y, 2), lambda x, y: f0(x, y, 2)]
    enc = Encoder(*inner_features)
    model = MaxEntModel(encoder=enc)
    X = [[1, 0, 1], [0, 0, 1], [1, 0, 1],  [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    y = array([0, 0, 0, 1, 1, 1])
    model.fit(X, y)
    assert_equal([model.predict(e) for e in X], [0, 0, 0, 1, 1, 1])
    assert_almost_equal(model.w, [ 2.2038, -0.5035, 3.1935, -2.7670, -2.4950,  3.0060, 1.], decimal=3)


def test_gd():
    model = MaxEntModel()
    X = lil_matrix([[1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1],  [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
    y = array([[0, 0, 0, 1, 1, 1]]).T
    model.fit(X, y, method='gd', iterations=100, C=0.)

    pred = [model.predict(X[i, :]) for i in range(X.shape[0])]
    assert_equal(pred, [0, 0, 0, 1, 1, 1])
    assert_almost_equal(model.w, [[-0.0906, -0.6742, -0.7803,  1.2476 ]], decimal=3)


def test_sgd():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1],  [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    X = hstack((ones((6, 1)), X))
    y = array([[0, 0, 0, 1, 1, 1]]).T

    seed(1)

    model = MaxEntModel()
    model.fit(X, y, method='sgd', iterations=50, C=0.)
    pred = [model.predict(X[i, :]) for i in range(X.shape[0])]
    assert_equal(pred, [0, 0, 0, 1, 1, 1])
    assert_almost_equal(model.w, [[-0.260, -1.320, -2.019, 2.655]], decimal=3)
