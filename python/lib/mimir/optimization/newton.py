from numpy.linalg import inv


def newton(cost, grad, hessian, theta, max_iter=50, tol=.00001, verbose=False, stats=None):
    i = 0
    j = 0
    new_j = cost(theta)

    while abs(j - new_j) > tol and i < max_iter:
        i += 1

        if verbose:
            print i, new_j

        j = new_j

        new_theta = theta.flatten() - inv(hessian(theta)).dot(grad(theta).flatten())
        new_theta.shape = theta.shape
        theta = new_theta

        new_j = cost(theta)

    if stats is not None:
        stats['iterations'] = i

    return theta, new_j
