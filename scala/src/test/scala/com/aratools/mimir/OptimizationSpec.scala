package com.aratools.mimir

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Gaussian
import com.aratools.mimir.Data.mlclassEx4
import com.aratools.mimir.maxent.MaxentModel
import org.scalatest._
import com.aratools.mimir.Data.addBias
import com.aratools.mimir.preprocessing.StandardScaler
import com.aratools.mimir.logistic.Optimization.{Newton, NewtonState}

class OptimizationSpec extends FunSuite {
  val epsilon = 1e-3f

  test("testNewtonLogistic") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val initial = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val cost = (theta: DenseVector[Double]) => opt.cost(x, y, theta)
    val grad = (theta: DenseVector[Double]) => opt.grad(x, y, theta)
    val hessian = (theta: DenseVector[Double]) => opt.hessian(x, theta)

    val theta = Optimization.newtonVector(cost, grad, hessian, initial, verbose = false)

    val result = sum(abs(theta - DenseVector(-16.3787, 0.1483, 0.1589))) / 3

    assert(result < epsilon)
  }

  test("testNewtonMaxent") {
    val (xRaw, yBin) = mlclassEx4()
    val x = addBias(xRaw)
    val y = DenseMatrix.horzcat(yBin.toDenseMatrix.t, 1.0d - yBin.toDenseMatrix.t)
    val initial = DenseMatrix((0.01d, 0.01d, 0.01d))
    val cost = (theta: DenseMatrix[Double]) => MaxentModel.cost(x, y, theta, lambda = 0.0)
    val grad = (theta: DenseMatrix[Double]) => MaxentModel.grad(x, y, theta, lambda = 0.0)
    val hessian = (theta: DenseMatrix[Double]) => MaxentModel.hessian(x, theta, lambda = 0.0)

    val theta = Optimization.newtonMatrix(cost, grad, hessian, initial, verbose = false)

    val result = sum(abs(theta - DenseMatrix((-16.3787, 0.1483, 0.1589)))) / 3

    assert(result < epsilon)
  }

  test("testNewtonMaxentIris") {
    val data = Data.iris()
    val x = Data.addBias(data._1)
    val y = Data.binarize(data._2)

    val initial = new DenseMatrix(rows = 2, cols = 5, data = Gaussian(mu = 0.0, sigma = 0.001).sample(10).toArray)
    val cost = (theta: DenseMatrix[Double]) => MaxentModel.cost(x, y, theta, lambda = 1.0)
    val grad = (theta: DenseMatrix[Double]) => MaxentModel.grad(x, y, theta, lambda = 1.0)
    val hessian = (theta: DenseMatrix[Double]) => MaxentModel.hessian(x, theta, lambda = 1.0)

    val theta = Optimization.newtonMatrix(cost, grad, hessian, initial, verbose = false)

    val result = sum(abs(theta - DenseMatrix(
      (17.8988, -0.783738, 1.24289, -3.87904, -1.65902),
      (11.7486, 0.260549, -0.33588, -1.83314, -2.06362)))) / 10

    assert(result < epsilon)
  }

  test("testSteepestLogistic") {
    val (xRaw, y) = mlclassEx4()
    val scaler = StandardScaler().fit(xRaw)
    val x = addBias(scaler.transform(xRaw))
    val initial = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(1.0))

    val cost = (theta: DenseVector[Double]) => opt.cost(x, y, theta)
    val grad = (theta: DenseVector[Double]) => opt.grad(x, y, theta)

    val theta = Optimization.steepestVector(cost, grad, initial, rho = 0.5, maxIter = 500, verbose = false)

    val result = sum(abs(theta - DenseVector(-0.0254469, 1.14114, 1.21333))) / 3

    assert(result < 0.1)
  }

  test("testSteepestIris") {
    val data = Data.iris()
    val scaler = StandardScaler().fit(data._1)
    val x = addBias(scaler.transform(data._1))
    val y = Data.binarize(data._2)

    val initial = new DenseMatrix(rows = 2, cols = 5, data = Gaussian(mu = 0.0, sigma = 0.001).sample(10).toArray)
    val cost = (theta: DenseMatrix[Double]) => MaxentModel.cost(x, y, theta, lambda = 1.0)
    val grad = (theta: DenseMatrix[Double]) => MaxentModel.grad(x, y, theta, lambda = 1.0)

    val theta = Optimization.steepestMatrix(cost, grad, initial, rho = 0.5, maxIter = 500, verbose = false)

    val result = sum(abs(theta - DenseMatrix(
      (0.0425072, -1.76158, 1.40147, -2.77042, -2.63817),
      (2.06606, -0.0531037, -0.120857, -1.19605, -2.26611)))) / 10

    assert(result < epsilon)
  }

  test("sign") {
    assert(scala.math.abs(Optimization.sign(0.8, 0.4) - 0.8) < epsilon)
    assert(scala.math.abs(Optimization.sign(0.4, -0.8) - -0.4) < epsilon)
  }

  test("bracket") {
    val data = mlclassEx4()
    val x = addBias(data._1)
    val y = data._2
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val p = -opt.grad(x, y, theta)
    val c = (alpha:Double) => opt.cost(x, y, theta + alpha*p)

    val (ax, bx, cx, fa, fb, fc) = Optimization.bracket(0.0, 1.0, c)

    assert(scala.math.abs(ax - -1.61803) < epsilon)
    assert(scala.math.abs(bx - 0) < epsilon)
    assert(scala.math.abs(cx - 1) < epsilon)
    assert(scala.math.abs(fa - 880.894) < epsilon)
    assert(scala.math.abs(fb - 0.778512) < epsilon)
    assert(scala.math.abs(fc - 651.614) < epsilon)
  }

  test("goldenSection") {
    val data = mlclassEx4()
    val x = addBias(data._1)
    val y = data._2
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val p = -opt.grad(x, y, theta)
    val c = (alpha:Double) => opt.cost(x, y, theta + alpha*p)

    val (min, fmin) = Optimization.goldenSection((-1.61803, 0, 1, 880.894, 0.778512, 651.614), c)

    assert(scala.math.abs(min - 0.000739624) < epsilon)
    assert(scala.math.abs(fmin - 0.685489) < epsilon)
  }

  test("lineSearch") {
    val data = mlclassEx4()
    val x = addBias(data._1)
    val y = data._2
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val p = -opt.grad(x, y, theta)
    val c = (alpha:Double) => opt.cost(x, y, theta + alpha*p)

    val (min, fmin) = Optimization.lineSearch(0.0, 1.0, c)

    assert(scala.math.abs(min - 0.000739624) < epsilon)
    assert(scala.math.abs(fmin - 0.685489) < epsilon)
  }
}
