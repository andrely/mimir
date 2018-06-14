package com.aratools.mimir

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.numerics._

object Optimization {
  def newtonMatrix(cost: DenseMatrix[Double] => Double,
                   grad: DenseMatrix[Double] => DenseMatrix[Double],
                   hessian: DenseMatrix[Double] => DenseMatrix[Double],
                   theta: DenseMatrix[Double],
                   maxIter: Int = 50, verbose: Boolean = true,
                   stats: Option[Stats] = Option.empty): DenseMatrix[Double] = {
    var iters = 0
    var oldCost = 0.0d
    var newCost = cost(theta)
    var mutableTheta = theta

    while (abs(newCost - oldCost) > .00001 & iters < maxIter) {
      iters += 1

      if (verbose) {
        println("iter " + iters + ", cost " + newCost)
      }

      val u = inv(hessian(mutableTheta)) * DenseMatrix.create(mutableTheta.size, 1, grad(mutableTheta).t.toArray)
      mutableTheta = mutableTheta - DenseMatrix.create(mutableTheta.cols, mutableTheta.rows, u.toArray).t
      oldCost = newCost
      newCost = cost(mutableTheta)
    }

    if (stats.isDefined) {
      stats.get.iterations = iters
    }

    mutableTheta
  }

  def newtonVector(cost: DenseVector[Double] => Double,
                   grad: DenseVector[Double] => DenseVector[Double],
                   hessian: DenseVector[Double] => DenseMatrix[Double],
                   theta: DenseVector[Double],
                   maxIter: Int = 50, verbose: Boolean = true,
                   stats: Option[Stats] = Option.empty): DenseVector[Double] = {
    val p = theta.length
    val wrappedCost = (theta: DenseMatrix[Double]) => cost(theta.toDenseVector)
    val wrappedGrad = (theta: DenseMatrix[Double]) => DenseMatrix.create(rows = 1, cols = p,
      data = grad(theta.toDenseVector).toArray)
    val wrappedHessian = (theta: DenseMatrix[Double]) => hessian(theta.toDenseVector)
    val wrappedTheta = DenseMatrix.create(rows = 1, cols = theta.length, data = theta.toArray)

    newtonMatrix(wrappedCost, wrappedGrad, wrappedHessian, wrappedTheta,
      maxIter = maxIter, verbose = verbose, stats = stats).toDenseVector
  }

  def newton(cost: Any, grad: Any, hessian: Any, theta: Any,
             maxIter: Int = 50, verbose: Boolean = true, stats: Option[Stats] = Option.empty) = {
    (cost, grad, hessian, theta) match {
      case (
        cost: (DenseVector[Double] => Double),
        grad: (DenseVector[Double] => DenseVector[Double]),
        hessian: (DenseVector[Double] => DenseMatrix[Double]),
        theta: DenseVector[Double]
        ) => newtonVector(cost, grad, hessian, theta, maxIter = maxIter, verbose = verbose, stats = stats)
      case (
        cost: (DenseMatrix[Double] => Double),
        grad: (DenseMatrix[Double] => DenseMatrix[Double]),
        hessian: (DenseMatrix[Double] => DenseMatrix[Double]),
        theta: DenseMatrix[Double]
        ) => newtonMatrix(cost, grad, hessian, theta, maxIter = maxIter, verbose = verbose, stats = stats)
      case _ => throw new IllegalArgumentException
    }
  }

  def steepestMatrix(cost: DenseMatrix[Double] => Double,
                   grad: DenseMatrix[Double] => DenseMatrix[Double],
                   theta: DenseMatrix[Double],
                   maxIter: Int = 50, rho: Double = 0.05, verbose: Boolean = true,
                   stats: Option[Stats] = Option.empty): DenseMatrix[Double] = {
    var iters = 0
    var oldCost = 0.0d
    var newCost = cost(theta)
    var mutableTheta = theta

    while (abs(newCost - oldCost) > .00001 & iters < maxIter) {
      iters += 1

      if (verbose) {
        println("iter " + iters + ", cost " + newCost)
      }

      mutableTheta -= rho*grad(mutableTheta)
      oldCost = newCost
      newCost = cost(mutableTheta)
    }

    if (stats.isDefined) {
      stats.get.iterations = iters
    }

    mutableTheta
  }

  def steepestVector(cost: DenseVector[Double] => Double,
                     grad: DenseVector[Double] => DenseVector[Double],
                     theta: DenseVector[Double],
                     maxIter: Int = 50, rho: Double = 0.05, verbose: Boolean = true,
                     stats: Option[Stats] = Option.empty): DenseVector[Double] = {
    val p = theta.length
    val wrappedCost = (theta: DenseMatrix[Double]) => cost(theta.toDenseVector)
    val wrappedGrad = (theta: DenseMatrix[Double]) => DenseMatrix.create(rows = 1, cols = p,
      data = grad(theta.toDenseVector).toArray)
    val wrappedTheta = DenseMatrix.create(rows = 1, cols = theta.length, data = theta.toArray)

    steepestMatrix(wrappedCost, wrappedGrad, wrappedTheta,
      maxIter = maxIter, rho = rho, verbose = verbose, stats = stats).toDenseVector
  }

  def sign(a:Double, b:Double): Double = {
    if (b == 0.0) {
      a
    }
    else {
      a*b/scala.math.abs(b)
    }
  }

  val GOLD = 1.618034
  val TINY = 1e-20
  val GLIMIT = 100

  def bracket(a:Double, b:Double, func:Double => Double): Tuple6[Double, Double, Double, Double, Double, Double] = {
    var ax = a
    var bx = b
    var fa = func(a)
    var fb = func(b)

    if (fb > fa) {
      val xtmp = ax
      ax = bx
      bx = xtmp

      val ftmp = fa
      fa = fb
      fb = ftmp
    }

    var cx = bx + GOLD*(bx - ax)
    var fc = func(cx)

    while (fb > fc) {
      var r = (bx - ax)*(fb - fc)
      var q = (bx - cx)*(fb - fa)
      var u = bx - ((bx - cx)*q - (bx - ax)*r) / 2.0*sign(math.max(math.abs(q - r), TINY), q - r)
      var ulim = bx + GLIMIT*(cx - bx)
      var fu = NaN

      if ((bx - u)*(u - cx) > 0) {
        fu = func(u)

        if (fu < fc) {
          ax = bx
          bx = u
          fa = fb
          fb = fu

          return (ax, bx, cx, fa, fb, fc)
        }
        else if (fu > fb) {
          cx = u
          fc = fu

          return (ax, bx, cx, fa, fb, fc)
        }

        u = cx + GOLD*(cx - bx)
        fu = func(u)
      }
      else if ((cx - u)*(u - ulim) > 0) {
        fu = func(u)

        if (fu < fc) {
          bx = cx
          cx = u
          u = u + GOLD*(u - cx)
          fb = fc
          fc = fu
          fu = func(u)
        }
      }
      else if ((u - ulim)*(ulim - cx) > 0) {
        u = ulim
        fu = func(u)
      }
      else {
        u = cx * GOLD*(cx - bx)
        fu = func(u)
      }

      ax = bx
      bx = cx
      cx = u
      fa = fb
      fb = fc
      fc = fu
    }

    if (ax > cx) {
      (cx, bx, ax, fc, fb, fa)
    }
    else {
      (ax, bx, cx, fa, fb, fc)
    }
  }

  val R = .61803399
  val C = 1 - R
  val TOL = 3.0e-8

  def goldenSection(bracketing:Tuple6[Double, Double, Double, Double, Double, Double],
                    func:Double => Double): Tuple2[Double, Double] = {
    var (ax, bx, cx, fa, fb, fc) = bracketing
    var iter = 1
    var x0 = ax
    var x3 = cx
    var x1 = NaN
    var x2 = NaN

    if (math.abs(cx - bx) > math.abs(bx - ax)) {
      x1 = bx
      x2 = bx - C*(cx - bx)
    }
    else {
      x2 = bx
      x1 = bx - C*(bx - ax)
    }

    var f1 = func(x1)
    var f2 = func(x2)

    while (math.abs(x3 - x0) > TOL*math.abs(x2 - x1) && iter < 50) {
      iter += 1
      if (f2 < f1) {
        x0 = x1
        x1 = x2
        x2 = R*x2 + C*x3
        f1 = f2
        f2 = func(x2)
      }
      else {
        x3 = x2
        x2 = x1
        x1 = R*x1 + C*x0
        f2 = f1
        f1 = func(x1)
      }
    }

    if (f1 < f2) {
      (x1, f1)
    }
    else {
      (x2, f2)
    }
  }

  def nanGuard(x:Double): Double = {
    if (x.isNaN) {
      Double.MaxValue
    }
    else {
      x
    }
  }

  def lineSearch(x: Double, y: Double, func: Double => Double): Tuple2[Double, Double] = {
    val innerFunc = (x:Double) => nanGuard(func(x))

    Optimization.goldenSection(Optimization.bracket(x, y, innerFunc), innerFunc)
  }
}
