import logging
from collections import defaultdict

from numpy import exp, sum, argmax, unique, zeros, ones, abs, array, log


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


def train_iis(model, X, iterations, tol, y):
    data_counts = [sum([e[1] for e in f]) for f
                   in [model.encoder(x, yy) for x, yy in zip(X, y)]]
    c = max(data_counts)
    model.n = float(len(X))
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


class MaxEntModel(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.classes = None

        self.w = None
        self.n = None

    def prob(self, x, y):
        f = self.encoder(x, y)

        return exp(sum([self.w[i] * val for i, val in f]))

    def predict_prob(self, x, normalize=True):
        probs = [self.prob(x, cl) for cl in self.classes]

        if normalize:
            probs = probs / sum(probs)

        return probs

    def predict(self, x):
        return self.classes[argmax(self.predict_prob(x, normalize=False))]

    def prob_f(self, f):
        return exp(sum([self.w[i] * val for i, val in f]))

    def fit(self, X, y, iterations=100, tol=.0001):
        self.classes = unique(y)

        train_iis(self, X, iterations, tol, y)

        return self

