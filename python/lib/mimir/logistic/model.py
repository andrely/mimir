import sys

from numpy import log, exp, diag, double, array, hstack, vstack
from numpy.linalg import inv
from numpy.ma import exp, log, zeros, outer
from numpy.random.mtrand import normal

import theano
import theano.tensor as T
import numpy as np

from mimir.maxent.model import log_sum_exp


def a(X, theta):
    return X.dot(theta)


def prob(X, theta):
    return exp(log_prob(X, theta))


def log_prob(X, theta):
    return -log_sum_exp(vstack([zeros(X.shape[0]), -X.dot(theta)]).T)


def cost(X, y, theta, l=1.0):
    N, _ = X.shape
    p = log_prob(X, theta)
    return (sum(-y * p - (1. - y) * (log_prob(X, theta) - a(X, theta))) / float(N)) + l*sum(theta[1:]**2)/(2.0*N)


def grad(X, y, theta, l=1.0):
    N, _ = X.shape
    err = prob(X, theta) - y
    err.shape = len(err), 1
    reg = zeros(theta.shape)
    reg[1:] = l*theta[1:] / N

    return sum(err * X + reg) / float(N)


def hessian(X, theta, l=1.0):
    N, P = X.shape

    result = zeros((len(theta), len(theta)))
    reg = diag([0] + ([l]*(P-1)))

    for i, (p, a_i) in enumerate(zip(log_prob(X, theta), a(X, theta))):
        result += exp(2*p - a_i) * outer(X[i, :], X[i, :])

    result += reg

    return result / float(N)


def update(X, y, theta, rho=.05, l=1.0):
    return theta - rho * inv(hessian(X, theta, l=l)).dot(grad(X, y, theta, l=l))


def make_theta(c):
    return normal(scale=.001, size=c)


def train(X, y, max_iter=50, rho=.05, l=1.0, stats=None, verbose=True):
    N, C = X.shape

    theta = make_theta(C)
    iter = 0
    j = 0
    new_j = cost(X, y, theta, l=l)

    while abs(j - new_j) > .00001 and iter < max_iter:
        if verbose:
            print(iter + 1, new_j)

        j = new_j
        theta = update(X, y, theta, rho=rho, l=l)
        new_j = cost(X, y, theta, l=l)
        iter += 1

    if stats is not None:
        stats['iterations'] = iter

    return theta


class LogisticModel():
    def __init__(self, C=1.0, rho=.05, max_iter=50):
        self.C = C
        self.rho = rho
        self.max_iter = max_iter
        self.theta = None
        self.stats = {}

    def train(self, X, y, verbose=True):
        self.theta = train(X, y, l=self.C, rho=self.rho, max_iter=self.max_iter, stats=self.stats, verbose=verbose)

        return self

    def predict(self, X):
        if not self.theta:
            raise ValueError

        return (log_prob(X, self.theta) > log(.5)).astype(double)

    def log_prob(self, X):
        return log_prob(X, self.theta)

    def replicate(self):
        return LogisticModel(C=self.C, rho=self.rho, max_iter=self.max_iter)


class LogisticGraph():
    def __init__(self, theta):
        self.x = T.dmatrix('x')
        self.y = T.dmatrix('y')
        self.n = self.x.shape[0]
        self.theta = theano.shared(theta)
        self.a = T.horizontal_stack((self.x.dot(self.theta)).reshape([self.n, 1]), T.zeros([self.n, 1]))
        self.prob = T.nnet.softmax(self.a)# T.exp(self.log_prob)
        self.l = T.dscalar('l')
        self.cost = -(T.sum(T.log(T.nnet.softmax(self.a))*self.y) / self.n) + self.l*T.sum(self.theta[2:]**2)/self.n
        self.grad = T.grad(self.cost, wrt=self.theta)
        self.hessian = theano.gradient.hessian(self.cost, self.theta)
        self.pred = self.prob > .5
        self.predict = theano.function(inputs=[self.x], outputs=[self.pred])


def theano_train(g, X, y, l=1.0, max_iter=50, stats=None, verbose=True):
    updates = [(g.theta, g.theta - T.nlinalg.matrix_inverse(g.hessian).dot(g.grad))]
    train = theano.function(inputs=[g.x, g.y, g.l],
                            outputs=[g.theta, g.cost],
                            updates=updates)

    j = sys.float_info.max

    iter = 1
    theta, new_j = train(X, y, l)

    if verbose:
        print(iter, new_j)

    while abs(j - new_j) > .00001 and iter < max_iter:
        j = new_j

        theta, new_j = train(X, y, l)
        iter += 1

        if verbose:
            print(iter, new_j)

    if stats is not None:
        stats['iterations'] = iter

    return g.theta.get_value()
