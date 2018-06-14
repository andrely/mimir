package com.aratools.mimir

import breeze.linalg.{DenseMatrix, DenseVector, argmax}

import scala.collection.immutable.IndexedSeq

class OVRClassifier(baseModel: Model) extends Model {
  var models: Option[IndexedSeq[Model]] = Option.empty

  override def train(X: DenseMatrix[Double], y: Any, verbose: Boolean): Model = {
    val yVal = y.asInstanceOf[DenseMatrix[Double]]
    val C = yVal.cols

    models = Option(
      for {
        i <- 0 until C
        model = baseModel.replicate()
      } yield {
        model.train(X, yVal(::, i), verbose = verbose)
      })

    this
  }

  override def logProb(x: DenseMatrix[Double]): Any = {
    val rows = for {m <- models.get} yield {m.logProb(x).asInstanceOf[DenseVector[Double]].toArray}
    DenseMatrix(rows:_*).t
  }

  override def replicate(): Model = {
    new OVRClassifier(baseModel = baseModel)
  }

  override def predict(X: DenseMatrix[Double]): Any = {
    val probs: DenseMatrix[Double] = logProb(X).asInstanceOf[DenseMatrix[Double]]
    val values: IndexedSeq[Double] = for {
      i <- 0 until X.rows
      row = probs(i, ::)
    } yield {
      argmax(row).toDouble
    }

    DenseVector(values:_*)
  }
}
