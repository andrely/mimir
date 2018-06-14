import logging

from itertools import product


import numpy as np

from mimir.logistic.model import cost, make_theta, log_prob


def train_irls(X, y, theta=None, l=1., max_iter=100):
    N, P = X.shape

    if not theta:
        theta = make_theta(P)

    iter = 0
    j = 0
    old_j = cost(X, y, theta, l=l)

    while abs(j - old_j) > .00001 and iter < max_iter:
        p = np.exp(log_prob(X, theta))
        r = p * (1 - p)
        z = X.theta * r - (1 / r) * (p - y) + l / r

        xrx = np.zeros((P, P))

        for i, j in product(range(P), range(P)):
            xrx[i, j] = np.sum(X[:,i] * X[:,j] * r + l * (i == j))

        xrz =



        old_j = j
        j = cost(X, y, theta, l=l)

        iter += 1

        if iter % 100 == 0:
            logging.info('iter {} cost {} delta {}'.format(iter, j, abs(j - old_j)))

    return theta