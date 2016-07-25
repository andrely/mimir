from numpy import ones, hstack, unique, array, float


def add_bias(X):
    N, _ = X.shape

    return hstack([ones((N, 1)), X])


def binarize(X):
    classes = unique(X)

    return array([(X == cls).astype(float) for cls in classes]).T
