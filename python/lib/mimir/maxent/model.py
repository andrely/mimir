import logging
from collections import defaultdict
from itertools import izip

from numpy import exp, sum, argmax, unique, zeros, ones, abs, array, log, float, amax, outer
from numpy.random.mtrand import shuffle
from scipy.sparse.base import issparse
from scipy.sparse.sputils import isdense


def _create_feature_cache(X, y, encoder, c):
    classes = unique(y)
    feat_counts = zeros(len(encoder) + 1)
    feat_cache = defaultdict(int)

    for i in range(len(X)):
        for j in range(len(classes)):
            f = encoder(X[i], classes[j])

            f += [(len(encoder), c - sum([e[1] for e in f]))]

            if classes[j] == y[i]:
                for k, val in f:
                    feat_counts[k] += val

            feat_cache[(i, j)] = f

    return feat_cache, feat_counts


def train_iis(model, X, y, iterations, tol):
    data_counts = [sum([e[1] for e in f]) for f
                   in [model.encoder(x, yy) for x, yy in zip(X, y)]]
    c = max(data_counts)

    model.N = float(len(X))
    model.w = ones(len(model.encoder) + 1)

    feat_cache, feat_counts = _create_feature_cache(X, y, model.encoder, c)

    lepf_emp = log(feat_counts + 1)

    it = 0

    for it in range(iterations):
        epf_est = zeros(len(model.encoder) + 1)

        for i in range(len(X)):
            f = [feat_cache[(i, j)] for j in range(len(model.classes))]
            p = array([sum([model.w[l] * val for l, val in ff]) for ff in f])
            p = p - (max(p) + log(sum(exp(p - max(p)))))

            for j in range(len(model.classes)):
                for k, val in feat_cache[(i, j)]:
                    epf_est[k] += exp(p[j]) * val

        new_w = model.w + (1. / c) * (lepf_emp - log(epf_est + 1))

        if max(abs(new_w - model.w)) < tol:
            model.w = new_w
            break

        model.w = new_w

    logging.info("Training finished in %d iterations ..." % (it + 1))

    return model


def train_gd(model, X, y, iterations=100, tol=.0001, alpha=.1, C=1.0):
    model.K = len(unique(y))
    model.N, model.P = X.shape

    model._init_weights()

    it = 0

    for it in range(iterations):
        grad = zeros((model.K - 1, model.P))

        for i in range(model.N):
            x = zip(X[i,:].rows[0], X[i,:].data[0])
            l_p = model.log_prob(X[i,:])

            for k in range(model.K - 1):
                for w_j, val in x:
                    t = y[i,k] == model.classes[k]
                    grad[k, w_j] += (t - exp(l_p[k]))*val

        delta = alpha * grad / model.N

        new_w = model.w + delta
        new_w[:,1:] -= 2 * C * model.w[:,1:]

        if amax(abs(new_w - model.w)) < tol:
            model.w = new_w
            break

        model.w = new_w

    logging.info("Training finished in %d iterations ..." % (it + 1))

    return model


def train_sgd_bin(model, X, y, iterations=100, tol=.0001, alpha=.1, C=1.0):
    model.N, model.P = X.shape
    model.K = len(unique(y))
    model._init_weights()

    it = 0

    for it in range(iterations):
        indexes = range(model.N)
        shuffle(indexes)

        for i in indexes:
            p = model.log_prob(X[i,:])

            t = y[i,:] == model.classes[0:-1]
            grad = outer(t - exp(p)[0:-1], X[i,:])

            new_w = model.w + alpha * grad
            new_w[1:] -= alpha * 2 * C * model.w[1:] / model.N

            if amax(abs(new_w - model.w)) < tol:
                model.w = new_w
                break

            model.w = new_w

    logging.info("Training finished in %d iterations ..." % (it + 1))


    return model


class MaxEntModel(object):
    def __init__(self, encoder=None):
        self.encoder = encoder
        self.classes = None
        self.input_type = None

        self.w = None
        self.K = None
        self.N = None
        self.P = None

    def _init_weights(self):
        self.w = zeros((self.K - 1, self.P))

        return self

    def log_prob(self, x):
        if self.input_type == 'sparse':
            x = zip(x.rows[0], x.data[0])
            idx, vals = izip(*x)
            p = array([self.w[i, list(idx)].dot(vals) for i in range(self.K - 1)] + [0.])
        elif self.input_type == 'iter':
            if self.encoder:
                x = [self.encoder(x, y) for y in self.classes]
                p = array([sum([self.w[l] * val for l, val in ff]) for ff in x])
            else:
                idx, vals = izip(*x)
                p = array([self.w[i, list(idx)].dot(vals) for i in range(self.K - 1)] + [0.])
        elif self.input_type == 'dense':
            p = array(self.w.dot(x).tolist() + [0.])
        else:
            raise ValueError

        p = p - (max(p) + log(sum(exp(p - max(p)))))

        return p

    def predict(self, x):
        return self.classes[argmax(self.log_prob(x))]

    def fit(self, X, y, method='iis', iterations=100, tol=.0001, **args):
        self.classes = unique(y)

        if issparse(X):
            self.input_type = 'sparse'
        elif isdense(X):
            self.input_type = 'dense'
        elif hasattr(X, '__iter__'):
            self.input_type = 'iter'
        else:
            raise ValueError

        if method == 'iis':
            train_iis(self, X, y, iterations, tol)
        elif method == 'gd':
            train_gd(self, X, y, iterations, tol, **args)
        elif method == 'sgd':
            train_sgd_bin(self, X, y, iterations, tol, **args)
        else:
            raise ValueError("Unknown training method %s ..." % method)

        return self

