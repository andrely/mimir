from numpy.ma import array
from numpy.ma.testutils import assert_equal

from mimir.data.preprocessing import add_bias, binarize


def test_add_bias():
    assert_equal(add_bias(array([[1., 2.], [3., 4.]])), [[1., 1., 2.], [1., 3., 4.]])


def test_binarize():
    assert_equal(binarize(array(['ba', 'foo', 'ba'])), [[1., 0.], [0., 1.], [1., 0.]])
