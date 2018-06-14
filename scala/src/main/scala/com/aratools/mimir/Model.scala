package com.aratools.mimir

import breeze.linalg.{DenseMatrix, DenseVector}

trait Model {
  def train(X: DenseMatrix[Double], y: Any, verbose: Boolean = true): Model

  def predict(X: DenseMatrix[Double]): Any

  def logProb(x: DenseMatrix[Double]): Any

  def replicate() : Model
}
