from numpy import array, zeros


def make_poly(X, n = 6):
    N = X.shape[0]
    P = (n*(n+1)//2) + n + 1

    result = zeros((N, P))

    for k in range(N):
        col = 0

        for i in range(n+1):
            for j in range(n+1):
                if (i + j) > 6:
                    continue

                result[k, col] = X[k, 0]**i * X[k, 1]**j

                col += 1

    return result


def ex4():
    return {'x': array([[55.5, 69.5], [41., 81.5], [53.5, 86.], [46., 84.], [41., 73.5],
                        [51.5, 69.], [51., 62.5], [42., 75.], [53.5, 83.], [57.5, 71.],
                        [42.5, 72.5], [41., 80.], [46., 82.], [46., 60.5], [49.5, 76.],
                        [41., 76.], [48.5, 72.5], [51.5, 82.5], [44.5, 70.5], [44., 66.],
                        [33., 76.5], [33.5, 78.5], [31.5, 72.], [33., 81.5], [42., 59.5],
                        [30., 64.], [61., 45.], [49., 79.], [26.5, 64.5], [34., 71.5],
                        [42., 83.5], [29.5, 74.5], [39.5, 70.], [51.5, 66.], [41.5, 71.5],
                        [42.5, 79.5], [35., 59.5], [38.5, 73.5], [32., 81.5], [46., 60.5],
                        [36.5, 53.], [36.5, 53.5], [24., 60.5], [19., 57.5], [34.5, 60.],
                        [37.5, 64.5], [35.5, 51.], [37., 50.5], [21.5, 42.], [35.5, 58.5],
                        [26.5, 68.5], [26.5, 55.5], [18.5, 67.], [40., 67.], [32.5, 71.5],
                        [39., 71.5], [43., 55.5], [22., 54.], [36., 62.5], [31., 55.5],
                        [38.5, 76.], [40., 75.], [37.5, 63.], [24.5, 58.], [30., 67.],
                        [33., 56.], [56.5, 61.], [41., 57.], [49.5, 63.], [34.5, 72.5],
                        [32.5, 69.], [36., 73.], [27., 53.5], [41., 63.5], [29.5, 52.5],
                        [20., 65.5], [38., 65.], [18.5, 74.5], [16., 72.5], [33.5, 68.]]),
            'y': array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}


def ex5():
    return {'x': array([[0.051267, 0.69956], [-0.092742, 0.68494], [-0.21371, 0.69225], [-0.375, 0.50219], [-0.51325, 0.46564],
                        [-0.52477, 0.2098], [-0.39804, 0.034357], [-0.30588, -0.19225], [0.016705, -0.40424], [0.13191, -0.51389],
                        [0.38537, -0.56506], [0.52938, -0.5212], [0.63882, -0.24342], [0.73675, -0.18494], [0.54666, 0.48757],
                        [0.322, 0.5826], [0.16647, 0.53874], [-0.046659, 0.81652], [-0.17339, 0.69956], [-0.47869, 0.63377],
                        [-0.60541, 0.59722], [-0.62846, 0.33406], [-0.59389, 0.005117], [-0.42108, -0.27266], [-0.11578, -0.39693],
                        [0.20104, -0.60161], [0.46601, -0.53582], [0.67339, -0.53582], [-0.13882, 0.54605], [-0.29435, 0.77997],
                        [-0.26555, 0.96272], [-0.16187, 0.8019], [-0.17339, 0.64839], [-0.28283, 0.47295], [-0.36348, 0.31213],
                        [-0.30012, 0.027047], [-0.23675, -0.21418], [-0.06394, -0.18494], [0.062788, -0.16301], [0.22984, -0.41155],
                        [0.2932, -0.2288], [0.48329, -0.18494], [0.64459, -0.14108], [0.46025, 0.012427], [0.6273, 0.15863],
                        [0.57546, 0.26827], [0.72523, 0.44371], [0.22408, 0.52412], [0.44297, 0.67032], [0.322, 0.69225],
                        [0.13767, 0.57529], [-0.0063364, 0.39985], [-0.092742, 0.55336], [-0.20795, 0.35599], [-0.20795, 0.17325],
                        [-0.43836, 0.21711], [-0.21947, -0.016813], [-0.13882, -0.27266], [0.18376, 0.93348], [0.22408, 0.77997],
                        [0.29896, 0.61915], [0.50634, 0.75804], [0.61578, 0.7288], [0.60426, 0.59722], [0.76555, 0.50219],
                        [0.92684, 0.3633], [0.82316, 0.27558], [0.96141, 0.085526], [0.93836, 0.012427], [0.86348, -0.082602],
                        [0.89804, -0.20687], [0.85196, -0.36769], [0.82892, -0.5212], [0.79435, -0.55775], [0.59274, -0.7405],
                        [0.51786, -0.5943], [0.46601, -0.41886], [0.35081, -0.57968], [0.28744, -0.76974], [0.085829, -0.75512],
                        [0.14919, -0.57968], [-0.13306, -0.4481], [-0.40956, -0.41155], [-0.39228, -0.25804], [-0.74366, -0.25804],
                        [-0.69758, 0.041667], [-0.75518, 0.2902], [-0.69758, 0.68494], [-0.4038, 0.70687], [-0.38076, 0.91886],
                        [-0.50749, 0.90424], [-0.54781, 0.70687], [0.10311, 0.77997], [0.057028, 0.91886], [-0.10426, 0.99196],
                        [-0.081221, 1.1089], [0.28744, 1.087], [0.39689, 0.82383], [0.63882, 0.88962], [0.82316, 0.66301],
                        [0.67339, 0.64108], [1.0709, 0.10015], [-0.046659, -0.57968], [-0.23675, -0.63816], [-0.15035, -0.36769],
                        [-0.49021, -0.3019], [-0.46717, -0.13377], [-0.28859, -0.060673], [-0.61118, -0.067982], [-0.66302, -0.21418],
                        [-0.59965, -0.41886], [-0.72638, -0.082602], [-0.83007, 0.31213], [-0.72062, 0.53874], [-0.59389, 0.49488],
                        [-0.48445, 0.99927], [-0.0063364, 0.99927]]),
            'y': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0])}
