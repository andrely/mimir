from nose.tools import assert_equal
from numpy.testing.utils import assert_almost_equal
from numpy.testing.utils import assert_equal as assert_np_equal
from sklearn.metrics.classification import accuracy_score

from mimir.data.iris import iris
from mimir.data.preprocessing import add_bias, binarize
from mimir.logistic.model import LogisticModel
from mimir.multiclass.ovr import OVRClassifier


def test_ovr():
    data = iris()
    x = add_bias(data['x'])
    y = binarize(data['y'])

    model = OVRClassifier(LogisticModel(rho=1.)).train(x, y, verbose=False)

    assert_equal(len(model.models), 3)

    assert_almost_equal(model.models[0].theta, [6.69036, -0.445019, 0.900008, -2.32352, -0.973446], decimal=3)
    assert_equal(model.models[0].stats['iterations'], 7)

    assert_almost_equal(model.models[1].theta, [5.58621, -0.17931, -2.12865, 0.696673, -1.27481], decimal=3)
    assert_equal(model.models[1].stats['iterations'], 4)

    assert_almost_equal(model.models[2].theta, [-14.4313, -0.394427, -0.51333, 2.93086, 2.41706], decimal=3)
    assert_equal(model.models[2].stats['iterations'], 7)

    assert_np_equal(model.predict(x),
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


def test_iris_benchmark():
    data = iris()
    x = add_bias(data['x'])
    y = binarize(data['y'])

    train_split = [12, 39, 23, 5, 3, 29, 49, 47, 21, 30, 34, 48, 20, 45, 31, 27, 17, 22,
                   41, 6, 40, 38, 42, 19, 26, 15, 35, 10, 46, 25, 0, 32, 1, 16, 4, 13,
                   24, 33, 43, 18, 81, 65, 62, 50, 93, 92, 53, 58, 87, 55, 70, 72, 83,
                   56, 52, 73, 78, 64, 68, 59, 74, 89, 67, 51, 66, 98, 90, 69, 95, 63,
                   82, 54, 86, 85, 96, 97, 79, 71, 94, 80, 142, 147, 125, 145, 119, 101,
                   141, 105, 129, 138, 122, 120, 139, 124, 134, 111, 148, 117, 132, 133,
                   104, 130, 128, 115, 127, 131, 136, 112, 107, 143, 149, 106, 109, 108,
                   102, 100, 126, 103, 146, 113]

    test_split = [2, 7, 8, 9, 11, 14, 28, 36, 37, 44, 57, 60, 61, 75, 76, 77, 84, 88,
                  91, 99, 110, 114, 116, 118, 121, 123, 135, 137, 140, 144]

    xTrain = x[train_split, :]
    yTrain = y[train_split, :]
    xTest = x[test_split, :]
    yTest = y[test_split, :]

    model = OVRClassifier(LogisticModel(rho=1.)).train(xTrain, yTrain, verbose=False)
    pred = binarize(model.predict(xTest))
    assert_almost_equal(accuracy_score(yTest, pred), 0.96667, decimal=3)
