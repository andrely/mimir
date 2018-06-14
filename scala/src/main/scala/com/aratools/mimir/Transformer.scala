package com.aratools.mimir

import breeze.linalg.DenseMatrix

trait Transformer {
  def fit(X:DenseMatrix[Double]) : Transformer

  def transform(X:DenseMatrix[Double]) : DenseMatrix[Double]

  def replicate() : Transformer
}
