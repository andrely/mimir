import numpy as np
import theano
import theano.tensor as T
from numpy.ma.testutils import assert_almost_equal, assert_equal

from mimir.data.mlclass import ex4, ex5, make_poly
from mimir.data.preprocessing import add_bias
from mimir.logistic.model import LogisticGraph, theano_train


def test_a():
    data = ex4()
    X = add_bias(data['x'])
    theta = np.array([.01, .01, .01])
    g = LogisticGraph(theta)
    assert_almost_equal(theano.function([g.x], g.a)(X),
                        [[1.26, 0.0], [1.235, 0.0], [1.405, 0.0], [1.31, 0.0], [1.155, 0.0], [1.215, 0.0],
                         [1.145, 0.0], [1.18, 0.0], [1.375, 0.0], [1.295, 0.0], [1.16, 0.0], [1.22, 0.0],
                         [1.29, 0.0], [1.075, 0.0], [1.265, 0.0], [1.18, 0.0], [1.22, 0.0], [1.35, 0.0],
                         [1.16, 0.0], [1.11, 0.0], [1.105, 0.0], [1.13, 0.0], [1.045, 0.0], [1.155, 0.0],
                         [1.025, 0.0], [0.95, 0.0], [1.07, 0.0], [1.29, 0.0], [0.92, 0.0], [1.065, 0.0],
                         [1.265, 0.0], [1.05, 0.0], [1.105, 0.0], [1.185, 0.0], [1.14, 0.0], [1.23, 0.0],
                         [0.955, 0.0], [1.13, 0.0], [1.145, 0.0], [1.075, 0.0], [0.905, 0.0], [0.91, 0.0],
                         [0.855, 0.0], [0.775, 0.0], [0.955, 0.0], [1.03, 0.0], [0.875, 0.0], [0.885, 0.0],
                         [0.645, 0.0], [0.95, 0.0], [0.96, 0.0], [0.83, 0.0], [0.865, 0.0], [1.08, 0.0],
                         [1.05, 0.0], [1.115, 0.0], [0.995, 0.0], [0.77, 0.0], [0.995, 0.0], [0.875, 0.0],
                         [1.155, 0.0], [1.16, 0.0], [1.015, 0.0], [0.835, 0.0], [0.98, 0.0], [0.9, 0.0],
                         [1.185, 0.0], [0.99, 0.0], [1.135, 0.0], [1.08, 0.0], [1.025, 0.0], [1.1, 0.0],
                         [0.815, 0.0], [1.055, 0.0], [0.83, 0.0], [0.865, 0.0], [1.04, 0.0], [0.94, 0.0],
                         [0.895, 0.0], [1.025, 0.0]],
                        decimal=3)


def test_log_prob():
    data = ex4()
    X = add_bias(data['x'])
    theta = np.array([.01, .01, .01])
    g = LogisticGraph(theta)
    assert_almost_equal(theano.function([g.x], T.log(g.prob))(X)[:, 0],
                        [-0.2497, -0.2553, -0.2194, -0.2389, -0.2739, -0.2598, -0.2763, -0.2679, -0.2254, -0.2421,
                         -0.2727, -0.2587, -0.2432, -0.2936, -0.2486, -0.2679, -0.2587, -0.2305, -0.2727, -0.2848,
                         -0.2861, -0.2799, -0.3014, -0.2739, -0.3066, -0.3270, -0.2949, -0.2432, -0.3354, -0.2962,
                         -0.2486, -0.3001, -0.2861, -0.2668, -0.2775, -0.2564, -0.3256, -0.2799, -0.2763, -0.2936,
                         -0.3397, -0.3383, -0.3544, -0.3789, -0.3256, -0.3053, -0.3484, -0.3455, -0.4218, -0.3270,
                         -0.3242, -0.3619, -0.3514, -0.2924, -0.3001, -0.2836, -0.3146, -0.3805, -0.3146, -0.3484,
                         -0.2739, -0.2727, -0.3092, -0.3604, -0.3187, -0.3412, -0.2668, -0.3160, -0.2787, -0.2924,
                         -0.3066, -0.2873, -0.3665, -0.2988, -0.3619, -0.3514, -0.3027, -0.3298, -0.3426, -0.3066],
                        decimal=3)


def test_cost():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    assert_almost_equal(theano.function([g.x, g.y, g.l], g.cost)(X, y, 0.), 0.7785, decimal=3)
    g = LogisticGraph(np.array([-0.23201762, -6.826957, -13.900436]))
    assert_almost_equal(theano.function([g.x, g.y, g.l], g.cost)(X, y, 0.), 651.61406, decimal=3)


def test_cost_reg():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    assert_almost_equal(theano.function([g.x, g.y, g.l], g.cost)(X, y, 8000), 0.7885, decimal=3)


def test_grad():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    assert_almost_equal(theano.function([g.x, g.y, g.l], g.grad)(X, y, 0.), [0.2420, 6.8370, 13.9104], decimal=3)


def test_grad_reg():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    # assert_almost_equal(theano.function([g.x, g.y, g.l], g.grad)(X, y, 80.), [0.2420, 6.8470, 13.9204], decimal=3)


def test_hessian():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    # assert_almost_equal(theano.function([g.x, g.y, g.l], g.hessian)(X, y, 0.), [[0.0001, 0.0022, 0.0044],
    #                                                                             [0.0022, 0.0630, 0.1214],
    #                                                                             [0.0044, 0.1214, 0.2536]],
    #                     decimal=3)


def test_hessian_reg():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    # assert_almost_equal(theano.function([g.x, g.y, g.l], g.hessian)(X, y, 80.), [[0.1905, 7.1009, 12.7284],
    #                                                                              [7.1009, 284.236, 478.956],
    #                                                                              [12.7284, 478.956, 869.698]],
    #                     decimal=3)


def test_update():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.05, .05, .05]))
    # assert_almost_equal(theano.function([g.x, g.y, g.l], g.theta - T.nlinalg.matrix_inverse(g.hessian).dot(g.grad))(X, y, 0.),
    #                     20*np.array([-1.4166, 0.02035, 0.02791]), decimal=3)


def test_train():
    data = ex4()
    X = add_bias(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array([.01, .01, .01]))
    stats = {}
    theano_train(g, X, y, l=0., stats=stats)
    assert_almost_equal(g.theta.get_value(), [-16.3787, 0.1483, 0.1589], decimal=3)
    assert_equal(stats['iterations'], 5)


def test_train_reg():
    data = ex5()
    X = make_poly(data['x'])
    y = np.vstack([data['y'], 1 - data['y']]).T
    g = LogisticGraph(np.array(np.random.normal(scale=.001, size=28)))
    stats = {}
    theano_train(g, X, y, l=0., stats=stats)
    # assert_almost_equal(np.linalg.norm(g.theta.get_value()), 7173, decimal=0)
    # assert_equal(stats['iterations'], 13)

    g = LogisticGraph(np.array(np.random.normal(scale=.001, size=28)))
    stats = {}
    theano_train(g, X, y, l=1., stats=stats)
    # assert_almost_equal(np.linalg.norm(g.theta.get_value()), 4.240, decimal=3)
    assert_equal(stats['iterations'], 4)

    g = LogisticGraph(np.array(np.random.normal(scale=.001, size=28)))
    stats = {}
    theano_train(g, X, y, l=10., stats=stats)
    # assert_almost_equal(np.linalg.norm(g.theta.get_value()), 0.9384, decimal=3)
    assert_equal(stats['iterations'], 3)