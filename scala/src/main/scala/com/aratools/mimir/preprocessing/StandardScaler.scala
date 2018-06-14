package com.aratools.mimir.preprocessing

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector}
import breeze.stats.{mean, stddev}
import com.aratools.mimir.{Model, Transformer}

class StandardScaler extends Transformer {
  var center:Option[DenseVector[Double]] = None
  var scale:Option[DenseVector[Double]] = None

  override def fit(X: DenseMatrix[Double]): Transformer = {
    center = Option(mean(X, Axis._0).inner)
    scale = Option(stddev(X, Axis._0).inner)

    this
  }

  override def transform(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val result = X.copy

    for (i <- 0 until X.rows) {
      result(i, ::) :-= center.get.t
      result(i, ::) :/= scale.get.t
    }

    result
  }

  override def replicate(): Transformer = {
    new StandardScaler()
  }
}

object StandardScaler {
  def apply(): StandardScaler = new StandardScaler()
}
