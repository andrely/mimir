from numpy import zeros, sum, ones, hstack, exp, argmax, log, kron, diag, outer, max
from numpy.linalg import inv
from numpy.random.mtrand import normal


def act(X, theta):
    N, _ = X.shape
    a = hstack([exp(X.dot(theta.T)), ones((N, 1))])

    return a


def prob(X, theta):
    a = act(X, theta)
    z = sum(a, axis=1)
    z.shape = len(z), 1
    p = a / z

    return p


def log_sum_exp(X):
    m = max(X, axis=1)
    m.shape = len(m), 1

    return m.flatten() + log(sum(exp(X - m), axis=1))


def log_prob(X, theta):
    N, _ = X.shape
    a = hstack([X.dot(theta.T), zeros((N, 1))])
    z = log_sum_exp(a)
    z.shape = len(z), 1

    return a - z


def h(X, theta):
    N, _ = X.shape
    C, _ = theta.shape

    a = act(X, theta)
    r = zeros((N, C + 1))

    for i, j in enumerate(argmax(a, axis=1)):
        r[i, j] = 1

    return r


def cost(X, y, theta, l=1.0):
    N, _ = X.shape

    return (sum(-y * log(prob(X, theta))) / N) + l/(2*N)*sum(theta[:,1:]**2)


def grad(X, y, theta, l=1.0):
    N, _ = X.shape
    C, _ = theta.shape

    err = prob(X, theta) - y
    reg = (l/N) * hstack([zeros((C, 1)), theta[:,1:]])

    result = zeros(theta.shape)

    for i in range(C):
        e = err[:, i]
        e.shape = len(e), 1
        result[i,:] = (sum(e * X, axis=0) / float(N)) + reg[i,:]

    return result


def hessian(X, theta, l=1.0):
    N, _ = X.shape
    C, P = theta.shape

    mu = prob(X, theta)[:, 0:C]
    reg = (l/N) * ones(theta.size)

    for i in range(C):
        reg[i*P] = 0.

    reg = diag(reg)

    result = zeros((theta.size, theta.size))

    for i in xrange(N):
        result += kron(diag(mu[i, 0:C]) - outer(mu[i, 0:C], mu[i, 0:C]), outer(X[i,:], X[i, :]))

    return (result / float(N)) + reg


def update(X, y, theta, rho=.05, l=1.0):
    g = grad(X, y, theta, l=l)
    update = theta - rho * inv(hessian(X, theta, l=l)).dot(g.flatten())
    update.shape = theta.shape

    return update


def make_theta(c, p):
    theta = normal(scale=.001, size=c * p)
    theta.shape = c, p

    return theta


def train(X, y, max_iter=50, rho=.05, l=1.0, stats=None, verbose=True):
    N, P = X.shape
    _, C = y.shape

    theta = make_theta(C - 1, P)
    iter = 0
    j = 0
    new_j = cost(X, y, theta, l=l)

    while abs(j - new_j) > .00001 and iter < max_iter:
        if verbose:
            print iter + 1, new_j

        j = new_j
        theta = update(X, y, theta, rho=rho, l=l)
        new_j = cost(X, y, theta, l=l)
        iter += 1

    if stats is not None:
        stats['iterations'] = iter

    return theta


class MaxentModel():
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

        return h(X, self.theta)

    def log_prob(self, X):
        return log_prob(X, self.theta)

    def replicate(self):
        return MaxentModel(C=self.C, rho=self.rho, max_iter=self.max_iter)
