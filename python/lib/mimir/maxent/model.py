import logging
from collections import defaultdict

from numpy import exp, sum, argmax, unique, zeros, ones, abs, array, log


class MaxEntModel(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.classes = None

        self.w = None
        self.c = None
        self.n = None

    def prob(self, x, y):
        f = self.encoder(x, y)
        f += [(len(self.encoder), self.c - sum([e[1] for e in f]))]

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
        data_counts = [sum([e[1] for e in f]) for f
                       in [self.encoder(x, yy) for x, yy in zip(X, y)]]

        self.c = max(data_counts)
        self.n = float(len(X))

        self.w = ones(len(self.encoder) + 1)

        feat_cache, feat_counts = self._create_feature_cache(X, y)

        lepf_emp = log(feat_counts + 1)

        it = 0

        for it in range(iterations):
            epf_est = zeros(len(self.encoder) + 1)

            for i in range(len(X)):
                f = [feat_cache[(i, j)] for j in range(len(self.classes))]
                p = array([sum([self.w[l] * val for l, val in ff]) for ff in f])
                p = p - (max(p) + log(sum(exp(p - max(p)))))

                for j in range(len(self.classes)):
                    for k, val in feat_cache[(i, j)]:
                        epf_est[k] += exp(p[j])*val

            new_w = self.w + (1./self.c)*(lepf_emp - log(epf_est + 1))

            if max(abs(new_w - self.w)) < tol:
                self.w = new_w
                break

            self.w = new_w

        logging.info("Training finished in %d iterations ..." % (it + 1))

        return self

    def _create_feature_cache(self, X, y):
        feat_counts = zeros(len(self.encoder) + 1)
        feat_cache = defaultdict(int)

        for i in range(len(X)):
            for j in range(len(self.classes)):
                f = self.encoder(X[i], self.classes[j])

                f += [(len(self.encoder), self.c - sum([e[1] for e in f]))]

                if self.classes[j] == y[i]:
                    for k, val in f:
                        feat_counts[k] += val

                feat_cache[(i, j)] = f

        return feat_cache, feat_counts
