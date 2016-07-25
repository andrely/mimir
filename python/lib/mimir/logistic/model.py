from numpy import log, exp, diag, double
from numpy.linalg import inv
from numpy.ma import exp, log, zeros, outer
from numpy.random.mtrand import normal


def a(X, theta):
    return X.dot(theta)


def prob(X, theta):
    return 1. / (1. + exp(-a(X, theta)))


def log_prob(X,theta):
    return -log(1 + exp(-a(X, theta)))


def cost(X, y, theta, l=1.0):
    N, _ = X.shape
    p = log_prob(X, theta)
    return (sum(-y * p - (1 - y) * (log_prob(X, theta) - a(X, theta))) / float(N)) + l*sum(theta[1:]**2)/(2.0*N)


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
            print iter + 1, new_j

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
