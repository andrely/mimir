package com.aratools.mimir.logistic

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{abs, exp, log}
import breeze.stats.distributions.Gaussian
import com.aratools.mimir.logistic.Optimization.{Newton, NewtonState, Optimizer}
import com.aratools.mimir.{Model, Numerics, Stats}


class LogisticModel(C: Double = 1.0, rho: Double = 0.05, maxIter: Int = 50) extends Model {
  var theta:Option[DenseVector[Double]] = None
  var stats:Option[Stats] = None

  override def train(X: DenseMatrix[Double], y: Any, verbose: Boolean = true): LogisticModel = {
    stats = Option(new Stats)
    theta = Option(LogisticModel.internalTrain(X, y.asInstanceOf[DenseVector[Double]],
      new Newton(NewtonState(C, rho)),
      maxIter, stats, verbose))

    this
  }

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    if (theta.isDefined) {
      (LogisticModel.logProb(X, theta.get) :> log(0.5)).toDenseVector.mapValues(_.compare(true).toDouble)
    }
    else throw new IllegalArgumentException
  }

  override def replicate(): Model = {
    new LogisticModel(C = C, rho = rho, maxIter = maxIter)
  }

  override def logProb(x: DenseMatrix[Double]): DenseVector[Double] = {
    if (theta.isDefined) {
      LogisticModel.logProb(x, theta.get)
    }
    else throw new IllegalArgumentException
  }
}

object LogisticModel {
  def a(x: DenseMatrix[Double], theta: DenseVector[Double]): DenseVector[Double] =
    x*theta

  def prob(x: DenseMatrix[Double], theta: DenseVector[Double]): DenseVector[Double] =
    exp(logProb(x, theta))

  def logProb(x: DenseMatrix[Double], theta: DenseVector[Double]): DenseVector[Double] = {
    val act = a(x, theta)
    val padded = DenseMatrix.zeros[Double](x.rows, 2)
    padded(::, 0) := act

    act - Numerics.logSumExp(padded)
  }

  def makeTheta(c: Int): DenseVector[Double] = {
    val normal = Gaussian(0, 0.001)

    normal.samplesVector(c)
  }

  def internalTrain(x: DenseMatrix[Double], y: DenseVector[Double], optimizer:Optimizer[NewtonState],
                    maxIter: Int = 50, stats: Option[Stats] = Option.empty,
                    verbose: Boolean = true): DenseVector[Double] = {
    var theta = makeTheta(x.cols)
    var iters = 0
    var oldCost = 0.0d
    var newCost = optimizer.cost(x, y, theta)

    while (abs(newCost - oldCost) > .00001 & iters < maxIter) {
      iters += 1

      if (verbose) {
        println("iter " + iters + ", cost " + newCost)
      }

      theta = optimizer.update(x, y, theta)
      oldCost = newCost
      newCost = optimizer.cost(x, y, theta)
    }

    if (stats.isDefined) {
      stats.get.iterations = iters
    }

    theta
  }
}
