from numpy import array
from numpy.ma.testutils import assert_almost_equal, assert_equal

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
    model = MaxEntModel(enc)
    X = [[1, 0, 1], [0, 0, 1], [1, 0, 1],  [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    y = array([0, 0, 0, 1, 1, 1])
    model.fit(X, y)
    assert_equal([model.predict(e) for e in X], [0, 0, 0, 1, 1, 1])
    assert_almost_equal(model.w, [ 2.2038, -0.5035, 3.1935, -2.7670, -2.4950,  3.0060, 1.], decimal=3)


def test_gd():
    inner_features = [lambda x, y: f1(x, y, 0), lambda x, y: f0(x, y, 0),
                      lambda x, y: f1(x, y, 1), lambda x, y: f0(x, y, 1),
                      lambda x, y: f1(x, y, 2), lambda x, y: f0(x, y, 2)]
    enc = Encoder(*inner_features)
    model = MaxEntModel(enc)
    X = [[1, 0, 1], [0, 0, 1], [1, 0, 1],  [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    y = array([0, 0, 0, 1, 1, 1])
    model.fit(X, y, method='gd', iterations=50)
    assert_equal([model.predict(e) for e in X], [0, 0, 0, 1, 1, 1])
    assert_almost_equal(model.w, [0.025, 0.01639, 0.0083,  0., 0.0083, 0.0247], decimal=3)
