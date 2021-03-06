from numpy import array, vstack
from numpy.linalg import norm
from numpy.ma.testutils import assert_array_equal, assert_equal
from numpy.testing.utils import assert_almost_equal

from mimir.data.mlclass import ex4, ex5, make_poly
from mimir.data.preprocessing import add_bias
from mimir.maxent.model import prob, act, h, cost, grad, hessian, log_prob, update, MaxentModel


def test_a_bin():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 2, 3]])

    assert_almost_equal(act(X, theta), [[54.598, 1.], [20.0855, 1.], [54.598, 1.],
                                        [2.7183, 1.], [403.429, 1.], [2.71828, 1.]],
                        decimal=3)


def test_a_mult():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1],
                   [2, 3, 1]])

    assert_almost_equal(act(X, theta), [[54.598, 2.718, 20.086, 1.],
                                        [20.086, 2.718, 2.718, 1.],
                                        [54.598, 2.718, 20.086, 1.],
                                        [2.718, 1., 7.389, 1.],
                                        [54.598, 20.086, 403.429, 1.],
                                        [2.718, 1., 7.389, 1.]],
                        decimal=3)


def test_p_bin():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 2, 3]])

    assert_almost_equal(prob(X, theta), [[0.9820, 0.01799], [0.9526, 0.04743], [0.9820, 0.01799],
                                         [0.7311, 0.2689], [0.9975, 0.002473], [0.7311, 0.2689]],
                        decimal=3)


def test_p_mult():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1],
                   [2, 3, 1]])
    assert_almost_equal(prob(X, theta), [[0.696, 0.0347, 0.256, 0.0128],
                                         [0.757, 0.1025, 0.1025, 0.0377],
                                         [0.696, 0.0347, 0.256, 0.0128],
                                         [0.225, 0.0826, 0.610, 0.0826],
                                         [0.114, 0.0419, 0.842, 0.00209],
                                         [0.225, 0.0826, 0.610, 0.0826]],
                        decimal=3)

    data = ex4()
    X = add_bias(data['x'])
    theta = array([[.01, 0.01, 0.01]])

    assert_almost_equal(prob(X, theta)[:, 0], [0.7790, 0.7747, 0.8030, 0.7875, 0.7604, 0.7712, 0.7586, 0.7649,
                                               0.7982, 0.7850, 0.7613, 0.7721, 0.7841, 0.7455, 0.7799, 0.7649,
                                               0.7721, 0.7941, 0.7613, 0.7521, 0.7512, 0.7558, 0.7398, 0.7604,
                                               0.7359, 0.7211, 0.7446, 0.7841, 0.7150, 0.7436, 0.7799, 0.7408,
                                               0.7512, 0.7658, 0.7577, 0.7738, 0.7221, 0.7558, 0.7586, 0.7455,
                                               0.7120, 0.7130, 0.7016, 0.6846, 0.7221, 0.7369, 0.7058, 0.7079,
                                               0.6559, 0.7211, 0.7231, 0.6964, 0.7037, 0.7465, 0.7408, 0.7530,
                                               0.7301, 0.6835, 0.7301, 0.7058, 0.7604, 0.7613, 0.7340, 0.6974,
                                               0.7271, 0.7110, 0.7658, 0.7291, 0.7568, 0.7465, 0.7359, 0.7503,
                                               0.6932, 0.7417, 0.6963, 0.7037, 0.7389, 0.7191, 0.7099, 0.7359],
                        decimal=3)


def test_h_bin():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 2, 3]])

    assert_array_equal(h(X, theta), [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])


def test_h_mult():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1],
                   [2, 3, 1]])

    assert_array_equal(h(X, theta), [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])


def test_cost_bin():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 2, 3]])
    y = array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])

    assert_almost_equal(cost(X, y, theta, l=0.), 1.952, decimal=3)


def test_cost_mult():
    X = array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1]])
    y = array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]])

    assert_almost_equal(cost(X, y, theta, l=0.), 1.793, decimal=3)

    data = ex4()
    X = add_bias(data['x'])
    theta = array([[.01, 0.01, 0.01]])
    y = vstack([data['y'], 1 - data['y']]).T

    assert_almost_equal(cost(X, y, theta, l=0.), 0.7785, decimal=3)
    assert_almost_equal(cost(X, y, theta, l=8000.), 0.7885, decimal=3)


def test_grad_bin():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 2, 3]])
    y = array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])

    assert_almost_equal(grad(X, y, theta, l=0.), [[0.2373, -0.0004, 0.4857]], decimal=3)


def test_grad_mult():
    X = array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1]])
    y = array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]])

    assert_almost_equal(grad(X, y, theta, l=0.), [[0.0797, 0.0121, 0.2502], [0.0315, 0.2863, -0.2623]], decimal=3)

    data = ex4()
    X = add_bias(data['x'])
    theta = array([[.01, 0.01, 0.01]])
    y = vstack([data['y'], 1 - data['y']]).T

    assert_almost_equal(grad(X, y, theta, l=0.), [[0.2420, 6.8370, 13.9104]], decimal=3)

    assert_almost_equal(grad(X, y, theta, l=80.), [[0.2420, 6.8470, 13.9204]], decimal=3)


def test_hessian_bin():
    X = array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 2, 3]])

    assert_almost_equal(hessian(X, theta, l=0.), [[0.0718, 0.0004, 0.0063],
                                            [0.0004, 0.0004, 0.0004],
                                            [0.0063, 0.0004, 0.0138]],
                        decimal=3)


def test_hessian_mult():
    X = array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1],
                   [2, 3, 1]])

    assert_almost_equal(hessian(X, theta, l=0.),
                        [[0.0839, 0.0196, 0.0521, -0.0080, -0.0009, -0.0048, -0.0712, -0.0186, -0.0457],
                         [0.0196, 0.0250, 0.0168, -0.0009, -0.0023, -0.0008, -0.0186, -0.0225, -0.0160],
                         [0.0521, 0.0168, 0.0827, -0.0048, -0.0008, -0.0178, -0.0457, -0.0160, -0.0587],
                         [-0.0080, -0.0010, -0.0048, 0.0323, 0.0141, 0.0123, -0.0230, -0.0131, -0.0074],
                         [-0.0009, -0.0023, -0.0008, 0.0141, 0.0454, 0.0067, -0.0131, -0.0415, -0.0059],
                         [-0.0048, -0.0008, -0.0178, 0.0123, 0.0067, 0.0276, -0.0074, -0.0059, -0.0091],
                         [-0.0712, -0.0186, -0.0457, -0.0229, -0.0131, -0.0074, 0.1043, 0.0330, 0.0539],
                         [-0.0186, -0.0225, -0.0160, -0.0131, -0.0415, -0.0059, 0.0330, 0.0691, 0.0222],
                         [-0.0457, -0.0160, -0.0587, -0.0074, -0.0059, -0.0091, 0.0539, 0.0221, 0.0693]],
                        decimal=3)

    X = array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0]])
    theta = array([[1, 0, 3],
                   [0, 2, 1]])

    assert_almost_equal(hessian(X, theta, l=0.),
                        [[0.1145, 0.0643, 0.0434, -0.0867, -0.0590, -0.0392],
                         [0.0643, 0.0802, 0.0335, -0.0590, -0.0730, -0.0319],
                         [0.0434, 0.0335, 0.0654, -0.0392, -0.0319, -0.0552],
                         [-0.0867, -0.0590, -0.0392, 0.1049, 0.0696, 0.0399],
                         [-0.0590, -0.0730, -0.0319, 0.0696, 0.0975, 0.0325],
                         [-0.0392, -0.0319, -0.0552, 0.0399, 0.0325, 0.0568]],
                        decimal=3)

    data = ex4()
    X = add_bias(data['x'])
    theta = array([[.01, 0.01, 0.01]])

    assert_almost_equal(hessian(X, theta, l=0.),
                        [[0.1905, 7.1009, 12.7284],
                         [7.1009, 283.236, 478.956],
                         [12.7284, 478.956, 868.698]],
                        decimal=3)

    assert_almost_equal(hessian(X, theta, l=80.),
                        [[0.1905, 7.1009, 12.7284],
                         [7.1009, 284.236, 478.956],
                         [12.7284, 478.956, 869.698]],
                        decimal=3)


def test_log_prob():
    data = ex4()
    X = add_bias(data['x'])
    theta = array([[.01, 0.01, 0.01]])

    assert_almost_equal(log_prob(X, theta)[:, 0],
                        [-0.2497, -0.2553, -0.2194, -0.2389, -0.2739, -0.2598, -0.2763,
                         -0.2679, -0.2254, -0.2421, -0.2727, -0.2587, -0.2432, -0.2936,
                         -0.2486, -0.2679, -0.2587, -0.2305, -0.2727, -0.2848, -0.2861,
                         -0.2799, -0.3014, -0.2739, -0.3066, -0.327, -0.2949, -0.2432,
                         -0.3354, -0.2962, -0.2486, -0.3001, -0.2861, -0.2668, -0.2775,
                         -0.2564, -0.3256, -0.2799, -0.2763, -0.2936, -0.3397, -0.3383,
                         -0.3544, -0.3789, -0.3256, -0.3053, -0.3484, -0.3455, -0.4218,
                         -0.327, -0.3242, -0.3619, -0.3514, -0.2924, -0.3001, -0.2836,
                         -0.3146, -0.3805, -0.3146, -0.3484, -0.2739, -0.2727, -0.3092,
                         -0.3604, -0.3187, -0.3412, -0.2668, -0.316, -0.2787, -0.2924,
                         -0.3066, -0.2873, -0.3665, -0.2988, -0.3619, -0.3514, -0.3027,
                         -0.3298, -0.3426, -0.3066],
                        decimal=3)


def test_update():
    data = ex4()
    X = add_bias(data['x'])
    theta = array([[.01, 0.01, 0.01]])
    y = vstack([data['y'], 1 - data['y']]).T

    assert_almost_equal(update(X, y, theta, rho=.05, l=1.), [[-0.5585, 0.0146, 0.0150]], decimal=3)


def test_train():
    data = ex4()
    X = add_bias(data['x'])
    y = vstack([data['y'], 1 - data['y']]).T
    m = MaxentModel(C=0., rho=1.).train(X, y, verbose=False)

    assert_almost_equal(m.theta, [[-16.3787, 0.1483, 0.1589]], decimal=3)
    assert_equal(m.stats['iterations'], 5)

    data = ex5()
    X = make_poly(data['x'])
    y = vstack([data['y'], 1 - data['y']]).T
    model = MaxentModel(rho=1., C=0.)
    model.train(X, y, verbose=False)

    assert_almost_equal(norm(model.theta), 7173, decimal=0)
    assert_equal(model.stats['iterations'], 13)

    model = MaxentModel(rho=1., C=1.)
    model.train(X, y, verbose=False)
    assert_almost_equal(norm(model.theta), 4.240, decimal=3)
    assert_equal(model.stats['iterations'], 4)

    model = MaxentModel(rho=1., C=10.)
    model.train(X, y, verbose=False)
    assert_almost_equal(norm(model.theta), 0.9384, decimal=3)
    assert_equal(model.stats['iterations'], 3)
