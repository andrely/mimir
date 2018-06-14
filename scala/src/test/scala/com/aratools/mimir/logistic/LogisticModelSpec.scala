package com.aratools.mimir.logistic

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum}
import breeze.numerics.abs
import com.aratools.mimir.Data.{addBias, makePoly, mlclassEx4, mlclassEx5}
import com.aratools.mimir.Stats
import com.aratools.mimir.logistic.LogisticModel.{logProb, prob}
import com.aratools.mimir.logistic.Optimization.{Newton, NewtonState}
import org.scalatest._

class LogisticModelSpec extends FunSuite {
  val epsilon = 1e-3f

  test("prob") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)
    val result = sum(abs(prob(x, theta) - DenseVector(
      0.7790d, 0.7747d, 0.8030d, 0.7875d, 0.7604d, 0.7712d, 0.7586d, 0.7649d, 0.7982d, 0.7850d,
      0.7613d, 0.7721d, 0.7841d, 0.7455d, 0.7799d, 0.7649d, 0.7721d, 0.7941d, 0.7613d, 0.7521d,
      0.7512d, 0.7558d, 0.7398d, 0.7604d, 0.7359d, 0.7211d, 0.7446d, 0.7841d, 0.7150d, 0.7436d,
      0.7799d, 0.7408d, 0.7512d, 0.7658d, 0.7577d, 0.7738d, 0.7221d, 0.7558d, 0.7586d, 0.7455d,
      0.7120d, 0.7130d, 0.7016d, 0.6846d, 0.7221d, 0.7369d, 0.7058d, 0.7079d, 0.6559d, 0.7211d,
      0.7231d, 0.6964d, 0.7037d, 0.7465d, 0.7408d, 0.7530d, 0.7301d, 0.6835d, 0.7301d, 0.7058d,
      0.7604d, 0.7613d, 0.7340d, 0.6974d, 0.7271d, 0.7110d, 0.7658d, 0.7291d, 0.7568d, 0.7465d,
      0.7359d, 0.7503d, 0.6932d, 0.7417d, 0.6963d, 0.7037d, 0.7389d, 0.7191d, 0.7099d, 0.7359d)))
    assert(result / 80 < epsilon)
  }

  test("logProb") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)
    val result = sum(abs(logProb(x, theta) - DenseVector(
      -0.2497, -0.2553, -0.2194, -0.2389, -0.2739, -0.2598, -0.2763,
      -0.2679, -0.2254, -0.2421, -0.2727, -0.2587, -0.2432, -0.2936,
      -0.2486, -0.2679, -0.2587, -0.2305, -0.2727, -0.2848, -0.2861,
      -0.2799, -0.3014, -0.2739, -0.3066, -0.327, -0.2949, -0.2432,
      -0.3354, -0.2962, -0.2486, -0.3001, -0.2861, -0.2668, -0.2775,
      -0.2564, -0.3256, -0.2799, -0.2763, -0.2936, -0.3397, -0.3383,
      -0.3544, -0.3789, -0.3256, -0.3053, -0.3484, -0.3455, -0.4218,
      -0.327, -0.3242, -0.3619, -0.3514, -0.2924, -0.3001, -0.2836,
      -0.3146, -0.3805, -0.3146, -0.3484, -0.2739, -0.2727, -0.3092,
      -0.3604, -0.3187, -0.3412, -0.2668, -0.316, -0.2787, -0.2924,
      -0.3066, -0.2873, -0.3665, -0.2988, -0.3619, -0.3514, -0.3027,
      -0.3298, -0.3426, -0.3066)))
    assert(result / 80 < epsilon)
  }

  test("cost") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val result = abs(opt.cost(x, y, theta) - 0.7785d)
    assert(result < epsilon)
  }

  test("cost-reg") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(8000.0))

    val result = abs(opt.cost(x, y, theta) - 0.7885d)
    assert(result < epsilon)
  }

  test("grad") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val result = sum(abs(opt.grad(x, y, theta) - DenseVector(0.2420d, 6.8370d, 13.9104d))) / 3
    assert(result < epsilon)
  }

  test("grad-reg") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(80.0))

    val result = sum(abs(opt.grad(x, y, theta) - DenseVector(0.2420, 6.8470, 13.9204))) / 3
    assert(result < epsilon)
  }

  test("hessian") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0))

    val result = sum(abs(opt.hessian(x, theta) - DenseMatrix(
      (0.1905d, 7.1009d, 12.7284d),
      (7.1009d, 283.236d, 478.956d),
      (12.7284d, 478.956d, 868.698d)))) / 9
    assert(result < epsilon)
  }

  test("hessian-reg") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(80.0))

    val result = sum(abs(opt.hessian(x, theta) - DenseMatrix(
      (0.1905d, 7.1009d, 12.7284d),
      (7.1009d, 284.236d, 478.956d),
      (12.7284d, 478.956d, 869.698d)))) / 9
    assert(result < epsilon)
  }

  test("update") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseVector(0.01d, 0.01d, 0.01d)

    val opt = new Newton(NewtonState(0.0, 0.05))

    val result = sum(abs(opt.update(x, y, theta) - DenseVector(-0.5585d, 0.0146d, 0.0150d))) / 3
    assert(result < epsilon)
  }

  test("train") {
    {
      val (xRaw, y) = mlclassEx4()
      val x = addBias(xRaw)
      val model = new LogisticModel(C = 0.0, rho = 1.0).train(x, y, verbose = false)
      val result = sum(abs(model.theta.get - DenseVector(-16.3787d, 0.1483d, 0.1589d))) / 3
      assert(result < epsilon)
      assert(model.stats.get.iterations == 5)
    }

    {
      val (xOrig, y) = mlclassEx5()
      val x = makePoly(xOrig)
      val model = new LogisticModel(C = 0.0, rho = 1.0).train(x, y, verbose = false)
      val result = abs(norm(model.theta.get) - 7173)
      assert(result < 1.0)
      assert(model.stats.get.iterations == 13)
    }

    {
      val (xOrig, y) = mlclassEx5()
      val x = makePoly(xOrig)
      val model = new LogisticModel(C = 1.0, rho = 1.0).train(x, y, verbose = false)
      val result = abs(norm(model.theta.get) - 4.240)
      assert(result < epsilon)
      assert(model.stats.get.iterations == 4)
    }

    {
      val (xOrig, y) = mlclassEx5()
      val x = makePoly(xOrig)
      val model = new LogisticModel(C = 10.0, rho = 1.0).train(x, y, verbose = false)
      val stats = new Stats
      val result = abs(norm(model.theta.get) - 0.9384)
      assert(result < epsilon)
      assert(model.stats.get.iterations == 3)
    }
  }
}
