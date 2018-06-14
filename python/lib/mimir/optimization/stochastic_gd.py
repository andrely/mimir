import sys

from math import ceil
from numpy.random.mtrand import choice


class MiniBatch():
    def __init__(self, X, y, size=10):
        self.X = X
        self.y = y
        self.size = size
        self.n = self.X.shape[0]
        self.n_batch = int(ceil(self.n / float(self.size)))

    def __iter__(self):
        idx = choice(self.n, size=self.n, replace=False)
        slices = ((self.size*i, min(self.size + self.size*i, self.n)) for i in xrange(self.n_batch))

        return ((self.X[idx[apply(range, s)]], self.y[idx[apply(range, s)]]) for s in slices)


def sgd(cost, grad, batch_iter, theta, rho=.05, max_iter=50, stats=None, tol=.0001, verbose=False):
    j = sys.float_info.max
    new_j = cost(theta)
    iter = 1

    while abs(j - new_j) > tol and iter <= max_iter:
        j = new_j

        if stats:
            iter_stats = stats.get('iterations', [])
            iter_stats.append((iter, j))
            stats['iterations'] = iter_stats

        if verbose:
            print('%d: %f' % (iter, j))

        for batch in batch_iter:
            theta -= rho*grad(theta, batch)

        new_j = cost(theta)

    return theta, new_j