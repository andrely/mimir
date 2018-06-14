package com.aratools.mimir

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics._

object Numerics {
  def logSumExp(x: DenseMatrix[Double]): DenseVector[Double] = {
    val result = DenseVector.zeros[Double](x.rows)

    for (i <- 0 until x.rows) {
      val m = max(x(i, ::))
      result(i) = m + log(sum(exp(x(i, ::) - m)))
    }

    result
  }
}
