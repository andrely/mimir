from nose.tools import assert_equal
from numpy.linalg import norm
from numpy.ma import array
from numpy.ma.testutils import assert_almost_equal

from mimir.data.mlclass import ex4, ex5, make_poly
from mimir.data.preprocessing import add_bias
from mimir.logistic.model import prob, cost, grad, hessian, update, log_prob, LogisticModel


def test_prob():
    X = array([[1., 55.5, 69.5], [1., 41., 81.5]])
    theta = array([-16.3787, 0.148341, 0.158908])

    assert_almost_equal(prob(X, theta), [0.9478, 0.9343], decimal=3)


def test_log_prob():
    data = ex4()
    X = add_bias(data['x'])
    theta = array([0.01, 0.01, 0.01])

    assert_almost_equal(log_prob(X, theta),
                        [-0.2497, -0.2553, -0.2194, -0.2389, -0.2739, -0.2598, -0.2763, -0.2679, -0.2254, -0.2421,
                         -0.2727, -0.2587, -0.2432, -0.2936, -0.2486, -0.2679, -0.2587, -0.2305, -0.2727, -0.2848,
                         -0.2861, -0.2799, -0.3014, -0.2739, -0.3066, -0.327, -0.2949, -0.2432, -0.3354, -0.2962,
                         -0.2486, -0.3001, -0.2861, -0.2668, -0.2775, -0.2564, -0.3256, -0.2799, -0.2763, -0.2936,
                         -0.3397, -0.3383, -0.3544, -0.3789, -0.3256, -0.3053, -0.3484, -0.3455, -0.4218, -0.327,
                         -0.3242, -0.3619, -0.3514, -0.2924, -0.3001, -0.2836, -0.3146, -0.3805, -0.3146, -0.3484,
                         -0.2739, -0.2727, -0.3092, -0.3604, -0.3187, -0.3412, -0.2668, -0.316, -0.2787, -0.2924,
                         -0.3066, -0.2873, -0.3665, -0.2988, -0.3619, -0.3514, -0.3027, -0.3298, -0.3426, -0.3066],
                        decimal=3)


def test_cost():
    data = ex4()
    X = add_bias(data['x'])
    y = data['y']
    theta = array([.01, 0.01, 0.01])

    assert_almost_equal(cost(X, y, theta, l=0.), 0.7785, decimal=3)
    assert_almost_equal(cost(X, y, array([-0.23201762, -6.826957, -13.900436]), l=0.), 651.61406, decimal=3)


def test_cost_reg():
    data = ex4()
    X = add_bias(data['x'])
    y = data['y']
    theta = array([.01, .01, .01])

    assert_almost_equal(cost(X, y, theta, l=8000), 0.7885, decimal=3)


def test_grad():
    data = ex4()
    X = add_bias(data['x'])
    y = data['y']
    theta = array([.01, .01, .01])

    assert_almost_equal(grad(X, y, theta, l=0.), [0.2420, 6.8370, 13.9104], decimal=3)


def test_grad_reg():
    data = ex4()
    X = add_bias(data['x'])
    y = data['y']
    theta = array([.01, .01, .01])

    assert_almost_equal(grad(X, y, theta, l=80.), [0.2420, 6.8470, 13.9204], decimal=3)


def test_hessian():
    data = ex4()
    X = add_bias(data['x'])
    theta = array([.1, .1, .1])

    assert_almost_equal(hessian(X, theta, l=0.), [[0.0001, 0.0022, 0.0044],
                                                  [0.0022, 0.0630, 0.1214],
                                                  [0.0044, 0.1214, 0.2536]],
                        decimal=3)


def test_hessian_reg():
    data = ex4()
    X = add_bias(data['x'])
    theta = array([.01, .01, .01])

    assert_almost_equal(hessian(X, theta, l=80.), [[0.1905, 7.1009, 12.7284],
                                                   [7.1009, 284.236, 478.956],
                                                   [12.7284, 478.956, 869.698]],
                        decimal=3)


def test_update():
    data = ex4()
    X = add_bias(data['x'])
    y = data['y']
    theta = array([.05, .05, .05])

    assert_almost_equal(update(X, y, theta, rho=.05, l=0.), [-1.4166, 0.02035, 0.02791], decimal=3)


def test_train():
    data = ex4()
    X = add_bias(data['x'])
    y = data['y']
    model = LogisticModel(rho=1., C=0.)
    model.train(X, y, verbose=False)

    assert_almost_equal(model.theta, [-16.3787, 0.1483, 0.1589], decimal=3)
    assert_equal(model.stats['iterations'], 5)


def test_train_reg():
    data = ex5()
    X = make_poly(data['x'])
    y = data['y']
    model = LogisticModel(rho=1., C=0.)
    model.train(X, y, verbose=False)

    assert_almost_equal(norm(model.theta), 7173, decimal=0)
    assert_equal(model.stats['iterations'], 13)

    model = LogisticModel(rho=1., C=1.)
    model.train(X, y, verbose=False)
    assert_almost_equal(norm(model.theta), 4.240, decimal=3)
    assert_equal(model.stats['iterations'], 4)

    model = LogisticModel(rho=1., C=10.)
    model.train(X, y, verbose=False)
    assert_almost_equal(norm(model.theta), 0.9384, decimal=3)
    assert_equal(model.stats['iterations'], 3)
