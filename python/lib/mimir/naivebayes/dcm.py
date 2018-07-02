import logging
from collections import Counter

import numpy as np
from scipy.sparse import issparse
from scipy.special import digamma, polygamma, gammaln, logsumexp


def estimate_dcm_newton(x, n_iter=100, tol=.001, trunc_val=1e-4):
    # Newton estimation linear in parameters as outlined in Minka (2003) and Huang (2005)
    i = 0

    # recommended initialization is the count mean
    alpha = np.mean(x, axis=0)

    # zero or negative alpha estimates are invalid but not automatically avoided by the optimization algorithm
    # if this happens at initialization or after update below we truncate the alpha value to a small positive value.
    # NOTE: all zero features are a case of degenerate input input and handled by the wrapped_dcm_estimate() function
    # which skips these features and sets their parameter to a small positive value.
    alpha[np.argwhere(alpha <= 0.).ravel()] = trunc_val

    for i in range(n_iter):
        # gradient calculation
        grad = np.sum(digamma(np.sum(alpha)) - digamma(np.sum(x, axis=1, keepdims=True) + np.sum(alpha)) +
                      digamma(x + alpha) - digamma(alpha), axis=0)

        # combined inverse hessian and newton update calculation
        q = np.sum(polygamma(1, x + alpha) - polygamma(1, alpha), axis=0)
        q[np.argwhere(q == 0.).ravel()] = trunc_val  # avoid division by zero on zero counts (see above)

        z = np.sum(polygamma(1, np.sum(alpha)) - polygamma(1, np.sum(x, axis=1) + np.sum(alpha)))

        b = np.sum(grad / q) / ((1 / z) + np.sum(1 / q))

        update = (grad - b) / q

        new_alpha = alpha - update
        new_alpha[np.argwhere(new_alpha <= 0.).ravel()] = trunc_val  # Truncate invalid alpha parameters (see above)

        if np.sum(np.abs(new_alpha - alpha)) < tol:
            alpha = new_alpha
            break

        alpha = new_alpha

    if i == n_iter - 1:
        logging.warning('DCM Newton estimation failed to converge at % iterations' % i)

    return i, alpha


def wrapped_dcm_estimate(x, trunc_value=1e-4):
    # removes all zero counts features and set the parameter for these to a small positive value. Then calls the
    # estimator on the remaining features.
    non_zero_idx = np.argwhere(np.sum(x, axis=0) != 0).ravel()
    inner_x = x[:, non_zero_idx]
    iters, inner_alpha = estimate_dcm_newton(inner_x)

    alpha = np.ones(x.shape[1]) * trunc_value

    for idx, a in zip(non_zero_idx, inner_alpha):
        alpha[idx] = a

    return iters, alpha


class DCMNaiveBayes(object):
    def __init__(self, n_iter=100, tol=.001, trunc_val=1e-4):
        self.alpha = None
        self.prior = None
        self.classes = None

        self.n_iter = n_iter
        self.tol = tol
        self.trunc_val = trunc_val

    def fit(self, X, y):
        p = X.shape[1]
        self.classes = np.unique(y)
        num_c = len(self.classes)

        class_counts = Counter(y)
        self.prior = np.array([np.log(class_counts[c]) - np.log(len(y)) for c in self.classes])
        self.alpha = np.zeros((num_c, p))

        for i, c in enumerate(self.classes):
            iters, a = wrapped_dcm_estimate(np.array(X[y == c].todense()), trunc_value=self.trunc_val)
            self.alpha[i] = a

        return self

    def predict_log_proba(self, X, normalized=False):
        log_probas = np.zeros((X.shape[0], self.alpha.shape[0]))

        alpha_tot = np.sum(self.alpha, axis=1)

        for j in range(X.shape[0]):
            # convert count matrix to float if necessary
            v = X[j].astype(np.float32)

            # convert to dense and flat array if necessary
            if issparse(v):
                v = np.array(v.todense())

            if len(v.shape) > 1:
                v = v[0]

            v_tot = np.sum(v)

            # factorial(v) is functionally the same as gamma(v + 1) and we can also handle fractional counts
            # NOTE the DCM normalization is not required and can be removed as an optimizationo. It is cancelled in
            # the class conditional normalization and included for completeness.
            norm = gammaln(np.sum(v) + 1) - np.sum(gammaln(v + 1))

            for i in range(self.alpha.shape[0]):
                a = self.alpha[i]
                a_tot = alpha_tot[i]

                log_probas[j, i] = gammaln(a_tot) - gammaln(v_tot + a_tot) + np.sum(gammaln(v + a) - gammaln(a))
                log_probas[j, i] += self.prior[i]

                if normalized:
                    log_probas[j, i] += norm

        if normalized:
            log_probas -= logsumexp(log_probas, axis=1, keepdims=True)

        return log_probas

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X, normalized=True))

    def predict(self, X):
        proba = self.predict_log_proba(X)

        pred = np.array([self.classes[i] for i in np.argmax(proba, axis=1)])

        return pred