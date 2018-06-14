from numpy.linalg import norm

from mimir.optimization.one_dim import line_search


def conjugate_gd_fr(cost, grad, theta, stats=None, max_iter=50, tol=.0001):
    iter = 1
    g = grad(theta)
    p = -g

    while norm(theta) > tol and iter <= max_iter:
        if stats:
            iter_stats = stats.get('iterations', [])
            iter_stats.append((iter, cost(theta)))
            stats['iterations'] = iter_stats

        iter += 1

        alpha = line_search(0., 1., lambda alpha: cost(theta + alpha*p))[0]
        theta += alpha*p
        g_prev = g
        g = grad(theta)
        beta = g.flatten().dot(g.flatten()) / g_prev.flatten().dot(g_prev.flatten())
        p = g + beta*p

    return theta, cost(theta)
