package com.aratools.mimir.logistic

import scala.math.{max, min, signum}
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, sum}
import breeze.numerics.{abs, exp}
import com.aratools.mimir.logistic.LogisticModel.{a, logProb, prob}

object Optimization {

  trait State

  trait Optimizer[T] {
    val state: T

    def cost(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): Double

    def update(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): DenseVector[Double]
  }

  case class NewtonState(lambda: Double = 1.0, rho: Double = 0.05) extends State

  class Newton(override val state: NewtonState) extends Optimizer[NewtonState] {
    def grad(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): DenseVector[Double] = {
      val lambda = state.lambda

      val N = x.rows
      val err = prob(x, theta) - y
      val result = DenseVector.zeros[Double](theta.length)
      val reg = DenseVector.tabulate(theta.length) { i => if (i == 0) 0.0d else lambda * theta(i) / N }

      for (i <- 0 until N) {
        result :+= err(i) * x(i, ::).t :+ reg
      }

      result / N.toDouble
    }

    def hessian(x: DenseMatrix[Double], theta: DenseVector[Double]): DenseMatrix[Double] = {
      val lambda = state.lambda

      val p = logProb(x, theta)
      val act = a(x, theta)
      val N = x.rows
      val result = DenseMatrix.zeros[Double](theta.length, theta.length)
      val reg = diag(DenseVector.tabulate(theta.length) { i => if (i == 0) 0.0d else lambda })

      for (i <- 0 until N) {
        result :+= exp(p(i) * 2.0d - act(i)) * (x(i, ::).t * x(i, ::))
      }

      result :+= reg

      result / N.toDouble
    }

    def update(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): DenseVector[Double] = {
      val lambda = state.lambda
      val rho = state.rho

      theta - rho * inv(hessian(x, theta)) * grad(x, y, theta)
    }

    def cost(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): Double = {
      val lambda = state.lambda

      val p = logProb(x, theta)
      val act = a(x, theta)
      -(sum(y :* p) + sum((1.0d - y) :* (p - act)) - lambda * sum(theta(1 to -1) :^ 2.0) / 2.0) / x.rows
    }
  }

  case class BBRState(C: Double = 1.0, theta: DenseVector[Double],
                      initBigTheta: Option[DenseVector[Double]]) extends State {
    private val length: Int = theta.length
    val bigDelta: DenseVector[Double] = initBigTheta.getOrElse(DenseVector.ones[Double](length) :* 0.1)
  }

  class BBR(override val state: BBRState) extends Optimizer[BBRState] {
    private def F(r: Double, delta: Double) = {
      if (scala.math.abs(r) <= delta) {
        .25
      }
      else {
        1 / (2 + scala.math.exp(scala.math.abs(r) - delta) + scala.math.exp(delta - scala.math.abs(r)))
      }
    }

    def update(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): DenseVector[Double] = {
      val bigDelta = state.bigDelta
      val C = state.C

      for (i <- 0 until theta.length) {
        val r = y :* (x * theta)
        val delta = abs(x(::, i)) * bigDelta(i)
        val f = DenseVector(r.data.zip(delta.data).map((F _).tupled))
        val u = C * sum(f :* (x(::, i) :^ 2.0))
        val l = C * sum(y :* x(::, i) :* (1.0 / (exp(r) + 1.0)))

        val g = if (theta(i) > 0) l + 1 else l - 1

        val z = -(g / u)

        val p = if (signum(theta(i) + z) == signum(theta(i))) z else -theta(i)
        val d = min(max(p, -bigDelta(i)), bigDelta(i))

        bigDelta(i) = max(2.0 * scala.math.abs(d), bigDelta(i) / 2.0)
        theta(i) += d
      }

      theta
    }

    override def cost(x: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): Double = ???
  }

}
