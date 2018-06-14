package com.aratools.mimir

import breeze.linalg.DenseVector

object Metrics {
  def accuracy(actual: DenseVector[Double], pred: DenseVector[Double]): Double = {
    var result = 0.0d

    for (i <- 0 until actual.length) {
      if (actual(i) == pred(i)) {
        result += 1
      }
    }

    result / actual.length
  }
}
