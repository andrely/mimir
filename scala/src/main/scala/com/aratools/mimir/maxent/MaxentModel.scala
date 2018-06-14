package com.aratools.mimir.maxent

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, argmax, diag, inv, kron, max, reshape, sum}
import breeze.numerics._
import breeze.stats.distributions.Gaussian
import com.aratools.mimir.{Model, Numerics, Stats}

class MaxentModel(C: Double = 1.0, rho: Double = 0.05, maxIter: Int = 50) extends Model {
  var theta:Option[DenseMatrix[Double]] = None
  var stats:Option[Stats] = None

  override def train(X: DenseMatrix[Double], y: Any, verbose: Boolean): MaxentModel = {
    stats = Option(new Stats)
    theta = Option(MaxentModel.internalTrain(X, y.asInstanceOf[DenseMatrix[Double]],
      C, rho, maxIter, stats, verbose))

    this
  }

  override def logProb(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    MaxentModel.logProb(x, theta.get)
  }

  override def replicate(): Model = {
    new MaxentModel(C = C, rho = rho, maxIter = maxIter)
  }

  override def predict(X: DenseMatrix[Double]): Any = {
    MaxentModel.h(X, theta.get)
  }
}

object MaxentModel {
  def logProb(x: DenseMatrix[Double], theta: DenseMatrix[Double]): DenseMatrix[Double] = {
    val logProb = DenseMatrix.horzcat(x*theta.t, DenseMatrix.zeros[Double](x.rows, 1))
    val z = Numerics.logSumExp(logProb)

    for (i <- 0 until logProb.cols) {
      logProb(::, i) :-= z
    }

    logProb
  }

  def prob(x: DenseMatrix[Double], theta: DenseMatrix[Double]): DenseMatrix[Double] = {
    exp(logProb(x, theta))
  }

  def h(x: DenseMatrix[Double], theta: DenseMatrix[Double]): DenseVector[Double] = {
    val act = logProb(x, theta)
    val result = DenseVector.zeros[Double](act.rows)

    for ((j, i) <- argmax(act, Axis._1).valuesIterator.zipWithIndex) {
      result(i) = j.toDouble
    }

    result
  }

  def cost(x: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double],
           lambda:Double = 1.0): Double = {
    val N = x.rows
    -(sum(y :* log(prob(x, theta))) / N) + lambda * sum(theta(::, 1 until theta.cols) :^ 2.0) / (2 * N)
  }

  def grad(x: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double],
           lambda: Double = 1.0): DenseMatrix[Double] = {
    val C = y.cols
    val N:Double = x.rows
    val reg = (lambda/N) * DenseMatrix.horzcat(DenseMatrix.zeros[Double](theta.rows, 1), theta(::, 1 until theta.cols))
    val err = prob(x, theta) - y

    val result = DenseMatrix.zeros[Double](theta.rows, theta.cols)

    for (i <- 0 until (C-1)) {
      val e = err(::, i)
      val sum1 = sum(x(::, *) :* e, Axis._0) :/ N
      result(i, ::) := sum1 + reg(i, ::)
    }

    result
  }

  def hessian(x: DenseMatrix[Double], theta: DenseMatrix[Double], lambda: Double = 1.0): DenseMatrix[Double] = {
    val N = x.rows
    val C = theta.rows
    val P = theta.cols

    val mu = prob(x, theta)(::, 0 until C)
    val reg = lambda * diag(DenseVector.ones[Double](theta.size))
    for (i <- 0 until theta.size) { if (i % P == 0) reg(i, i) = 0.0 }
    val result = DenseMatrix.zeros[Double](theta.size, theta.size)

    for (i <- 0 until N) {
      result :+= kron(diag(mu(i, ::)) - mu(i, ::).t * mu(i, ::), x(i, ::).t * x(i, ::))
    }

    result + reg :/ N.toDouble
  }

  def update(x: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double],
             lambda: Double = 1.0, rho: Double = 0.05): DenseMatrix[Double] = {
    val u = inv(hessian(x, theta, lambda = lambda)) * DenseMatrix.create(theta.size, 1, grad(x, y, theta, lambda = lambda).t.toArray)
    theta - rho*DenseMatrix.create(theta.cols, theta.rows, u.toArray).t
  }

  def makeTheta(c: Int, p: Int): DenseMatrix[Double] = {
    val normal = Gaussian(0, 0.001)

    DenseMatrix.create(c, p, normal.sample(c*p).toArray)
  }

  def internalTrain(x: DenseMatrix[Double], y: DenseMatrix[Double], lambda: Double = 1.0, rho: Double = 0.05,
                    maxIter: Int = 50, stats: Option[Stats] = Option.empty,
                    verbose: Boolean = true): DenseMatrix[Double] = {
    var theta = makeTheta(y.cols - 1, x.cols)
    var iters = 0
    var oldCost = 0.0d
    var newCost = cost(x, y, theta, lambda = lambda)

    while (abs(newCost - oldCost) > .00001 & iters < maxIter) {
      iters += 1

      if (verbose) {
        println("iter " + iters + ", cost " + newCost)
      }

      theta = update(x, y, theta, lambda = lambda, rho = rho)
      oldCost = newCost
      newCost = cost(x, y, theta, lambda = lambda)
    }

    if (stats.isDefined) {
      stats.get.iterations = iters
    }

    theta
  }
}
