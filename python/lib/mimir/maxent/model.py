from collections import defaultdict

from numpy import exp, sum, product, argmax, unique, zeros, ones, abs


class MaxEntModel(object):
    def __init__(self, encoder, type='dual'):
        self.encoder = encoder
        self.classes = None
        self.type = type

        self.w = None
        self.feat_map = None
        self.feat_counts = None
        self.data_counts = None
        self.c = None
        self.n = None

    def prob(self, x, y):
        if self.type == 'primal':
            return product([self.w[i] ** val for i, val in self.encoder(x, y)])
        elif self.type == 'dual':
            return exp(sum([self.w[i] * val for i, val in self.encoder(x, y)]))

    def predict_prob(self, x, normalize=True):
        probs = [self.prob(x, cl) for cl in self.classes]

        if normalize:
            probs = probs / sum(probs)

        return probs

    def predict(self, x):
        return self.classes[argmax(self.predict_prob(x, normalize=False))]

    def prob_f(self, f):
        if self.type == 'primal':
            return product([self.w[i] ** val for i, val in f])
        elif self.type == 'dual':
            return exp(sum([self.w[i] * val for i, val in f]))

    def fit(self, X, y, iterations=100):
        self.classes = unique(y)
        self.feat_map = [self.encoder(x, yy) for x, yy in zip(X, y)]
        self.data_counts = [sum([e[1] for e in f]) for f in self.feat_map]

        self.c = max(self.data_counts)
        self.n = float(len(X))

        [f.append((len(self.encoder), self.c - self.data_counts[i])) for i, f in enumerate(self.feat_map)]

        self.w = ones(len(self.encoder) + 1)

        self.feat_counts = zeros(len(self.encoder) + 1)

        for f in self.feat_map:
            for i, val in f:
                self.feat_counts[i] += val

        epf_emp = self.feat_counts + 1

        # print epf_emp

        feat_cache = defaultdict(int)

        for i in range(len(X)):
            for j in range(len(self.classes)):
                f = self.encoder(X[i], self.classes[j])
                f += [(len(self.encoder), self.c - sum([e[1] for e in f]))]

                feat_cache[(i, j)] = f

        # print feat_cache.items()

        for _ in range(iterations):
            epf_est = ones(len(self.encoder) + 1)

            for i in range(len(X)):
                f = [feat_cache[(i, j)] for j in range(len(self.classes))]
                p = [self.prob_f(ff) for ff in f]
                p = p /sum(p)

                for j in range(len(self.classes)):
                    for k, val in feat_cache[(i, j)]:
                        epf_est[k] += p[j] * val

            # print epf_est

            new_w = self.w * (epf_emp / epf_est) ** (1./self.c)

            if max(abs(new_w - self.w)) < .0001:
                self.w = new_w
                break

            self.w = new_w

            # print self.w

        return self
