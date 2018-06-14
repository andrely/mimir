import logging

import numpy as np

from mimir.logistic.model import log_prob, a


def cost(X, y, theta, l=1.0):
    N, _ = X.shape
    p = log_prob(X, theta)
    return (sum(-y * p - (1. - y) * (log_prob(X, theta) - a(X, theta))) / float(N)) + l*sum(np.abs(theta[1:]))/(2.0*N)


def F(r, delta):
    if np.abs(r) <= delta:
        return .25
    else:
        return 1. / (2. + np.exp(np.abs(r) - delta) + np.exp(delta - np.abs(r)))


def train_bbr(X, y, C=1., theta=None, max_iter=1e4):
    N, P = X.shape

    if not theta:
        theta = np.linalg.inv(X.T.dot(X) - C*np.identity(P)).dot(X.T).dot(y)

    big_delta = np.ones(P) * .1

    iter = 0
    j = 0
    old_j = cost(X, y, theta, l=C)

    while abs(j - old_j) > .00001 and iter < max_iter:
        for i in range(P):
            r = y * X.dot(theta)
            delta = np.abs(X[:,i]) * big_delta[i]
            f = np.array([F(r_i, d_i) for r_i, d_i in zip(r, delta)])
            u = C*np.sum(f * (X[:,i]**2))
            l = C*np.sum(y * X[:,i] * ((1./(1. + np.exp(-r))) - 1.))

            if theta[i] > 0:
                g = l + 1.
            else:
                g = l - 1.

            z = -(g/u)

            if np.sign(theta[i] + z) == np.sign(theta[i]):
                p = z
            else:
                p = -theta[i]

            d = min(max(p, -big_delta[i]), big_delta[i])

            big_delta[i] = max(2. * np.abs(d), big_delta[i] / 2.)
            theta[i] += d

        old_j = j
        j = cost(X, y, theta, l=l)

        iter += 1

        if iter % 100 == 0:
            logging.info('iter {} cost {} delta {}'.format(iter, j, abs(j - old_j)))

    return theta
