package com.aratools.mimir.maxent

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum}
import breeze.numerics.abs
import com.aratools.mimir.{Data, Metrics, OVRClassifier}
import com.aratools.mimir.Data._
import com.aratools.mimir.logistic.LogisticModel
import com.aratools.mimir.logistic.LogisticModel.{prob => _, _}
import com.aratools.mimir.maxent.MaxentModel._
import org.scalatest._

class MaxentModelSpec extends FunSuite {

  val epsilon = 1e-3f

  test("logProb") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseMatrix((0.01d, 0.01d, 0.01d))
    val result = sum(abs(MaxentModel.logProb(x, theta)(::, 0) - DenseVector(
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
      -0.3298, -0.3426, -0.3066))) / 80
    assert(result < epsilon)
  }

  test("prob") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseMatrix((0.01d, 0.01d, 0.01d))
    val result = sum(abs(MaxentModel.prob(x, theta)(::, 0) - DenseVector(
      0.7790, 0.7747, 0.8030, 0.7875, 0.7604, 0.7712, 0.7586, 0.7649,
      0.7982, 0.7850, 0.7613, 0.7721, 0.7841, 0.7455, 0.7799, 0.7649,
      0.7721, 0.7941, 0.7613, 0.7521, 0.7512, 0.7558, 0.7398, 0.7604,
      0.7359, 0.7211, 0.7446, 0.7841, 0.7150, 0.7436, 0.7799, 0.7408,
      0.7512, 0.7658, 0.7577, 0.7738, 0.7221, 0.7558, 0.7586, 0.7455,
      0.7120, 0.7130, 0.7016, 0.6846, 0.7221, 0.7369, 0.7058, 0.7079,
      0.6559, 0.7211, 0.7231, 0.6964, 0.7037, 0.7465, 0.7408, 0.7530,
      0.7301, 0.6835, 0.7301, 0.7058, 0.7604, 0.7613, 0.7340, 0.6974,
      0.7271, 0.7110, 0.7658, 0.7291, 0.7568, 0.7465, 0.7359, 0.7503,
      0.6932, 0.7417, 0.6963, 0.7037, 0.7389, 0.7191, 0.7099, 0.7359))) / 80
    assert(result < epsilon)
  }

  test("p bin") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 0.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix((1.0, 2.0, 3.0))
    val result = sum(abs(prob(x, theta) - DenseMatrix((0.9820, 0.01799), (0.9526, 0.04743), (0.9820, 0.01799),
      (0.7311, 0.2689), (0.9975, 0.002473), (0.7311, 0.2689)))) / 12
    assert(result < epsilon)
  }

  test("p mult") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 0.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix(
      (1.0, 0.0, 3.0),
      (0.0, 2.0, 1.0),
      (2.0, 3.0, 1.0))
    val result = sum(abs(prob(x, theta) - DenseMatrix(
      (0.696, 0.0347, 0.256, 0.0128),
      (0.757, 0.1025, 0.1025, 0.0377),
      (0.696, 0.0347, 0.256, 0.0128),
      (0.225, 0.0826, 0.610, 0.0826),
      (0.114, 0.0419, 0.842, 0.00209),
      (0.225, 0.0826, 0.610, 0.0826)))) / 24
    assert(result < epsilon)
  }

  test("cost bin") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 0.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix((1.0, 2.0, 3.0))
    val y = DenseMatrix((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0))

    assert(abs(cost(x, y, theta, lambda = 0.0) - 1.952) < epsilon)
  }

  test("cost mult") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 1.0, 0.0),
      (0.0, 1.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix((1.0, 0.0, 3.0), (0.0, 2.0, 1.0))
    val y = DenseMatrix((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert(abs(cost(x, y, theta, lambda = 0.0) - 1.793) < epsilon)
  }

  test("cost") {
    val (xRaw, yRaw) = mlclassEx4()
    val x = addBias(xRaw)
    val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
    val theta = DenseMatrix((0.01d, 0.01d, 0.01d))

    var result = abs(MaxentModel.cost(x, y, theta, lambda = 0.0) - 0.7785)
    assert(result < epsilon)

    result = abs(MaxentModel.cost(x, y, theta, lambda = 8000.0) - 0.7885)
    assert(result < epsilon)
  }

  test("grad bin") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 0.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix((1.0, 2.0, 3.0))
    val y = DenseMatrix((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0))

    val result = sum(abs(grad(x, y, theta, lambda = 0.0) - DenseMatrix((0.2373, -0.0004, 0.4857))))

    assert(result < epsilon)
  }

  test("grad mult") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 1.0, 0.0),
      (0.0, 1.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix((1.0, 0.0, 3.0), (0.0, 2.0, 1.0))
    val y = DenseMatrix((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert(abs(sum(grad(x, y, theta, lambda = 0.0) - DenseMatrix((0.0797, 0.0121, 0.2502), (0.0315, 0.2863, -0.2623)))) < epsilon)

    assert(abs(sum(grad(x, y, theta, lambda = 10.0) - DenseMatrix((0.0797473, 0.0121058, 5.25024), (0.0315305, 3.6196, 1.40437)))) < epsilon)
  }

  test("grad") {
    val (xRaw, yRaw) = mlclassEx4()
    val x = addBias(xRaw)
    val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
    val theta = DenseMatrix((0.01d, 0.01d, 0.01d))
    var result = sum(abs(grad(x, y, theta, lambda = 0.0) - DenseMatrix(
      (0.2420, 6.8370, 13.9104)))) / 3
    assert(result < epsilon)
    result = sum(abs(grad(x, y, theta, lambda = 80.0) - DenseMatrix(
      (0.2420, 6.8470, 13.9204)))) / 3
    assert(result < epsilon)
  }

  test("hessian bin") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 0.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    val theta = DenseMatrix((1.0, 2.0, 3.0))

    val result = sum(abs(hessian(x, theta, lambda = 0.0) - DenseMatrix((0.0718, 0.0004, 0.0063),
      (0.0004, 0.0004, 0.0004),
      (0.0063, 0.0004, 0.0138))))

    assert(result < epsilon)
  }

  test("hessian mult") {
    val x = DenseMatrix(
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (1.0, 1.0, 0.0),
      (0.0, 1.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 0.0))
    var theta = DenseMatrix(
      (1.0, 0.0, 3.0),
      (0.0, 2.0, 1.0),
      (2.0, 3.0, 1.0))

    var result = sum(abs(hessian(x, theta, lambda = 0.0) - DenseMatrix(
      (0.0839, 0.0196, 0.0521, -0.0080, -0.0009, -0.0048, -0.0712, -0.0186, -0.0457),
      (0.0196, 0.0250, 0.0168, -0.0009, -0.0023, -0.0008, -0.0186, -0.0225, -0.0160),
      (0.0521, 0.0168, 0.0827, -0.0048, -0.0008, -0.0178, -0.0457, -0.0160, -0.0587),
      (-0.0080, -0.0010, -0.0048, 0.0323, 0.0141, 0.0123, -0.0230, -0.0131, -0.0074),
      (-0.0009, -0.0023, -0.0008, 0.0141, 0.0454, 0.0067, -0.0131, -0.0415, -0.0059),
      (-0.0048, -0.0008, -0.0178, 0.0123, 0.0067, 0.0276, -0.0074, -0.0059, -0.0091),
      (-0.0712, -0.0186, -0.0457, -0.0229, -0.0131, -0.0074, 0.1043, 0.0330, 0.0539),
      (-0.0186, -0.0225, -0.0160, -0.0131, -0.0415, -0.0059, 0.0330, 0.0691, 0.0222),
      (-0.0457, -0.0160, -0.0587, -0.0074, -0.0059, -0.0091, 0.0539, 0.0221, 0.0693)))) / (9.0 * 9.0)

    assert(result < epsilon)

    result = sum(abs(hessian(x, theta, lambda = 10.0) - DenseMatrix(
      (0.0838768, 0.0196201, 0.0520671, -0.00804248, -0.000927776, -0.00482031, -0.0712059, -0.0186349, -0.0457268),
      (0.0196201, 1.69175, 0.0168284, -0.000927776, -0.00234534, -0.000796224, -0.0186349, -0.0224882, -0.0159926),
      (0.0520671, 0.0168284, 1.74937, -0.00482031, -0.000796224, -0.0177566, -0.0457268, -0.0159926, -0.0586631),
      (-0.00804248, -0.000927776, -0.00482031, 0.0322636, 0.0140566, 0.0122723, -0.0229474, -0.0130659, -0.00736372),
      (-0.000927776, -0.00234534, -0.000796224, 0.0140566, 1.71203, 0.00669415, -0.0130659, -0.0415384, -0.00588334),
      (-0.00482031, -0.000796224, -0.0177566, 0.0122723, 0.00669415, 1.69427, -0.00736372, -0.00588334, -0.00911446),
      (-0.0712059, -0.0186349, -0.0457268, -0.0229474, -0.0130659, -0.00736372, 0.104364, 0.0329657,0.053928),
      (-0.0186349, -0.0224882, -0.0159926, -0.0130659, -0.0415384, -0.00588334, 0.0329657, 1.73581, 0.0221688),
      (-0.0457268, -0.0159926, -0.0586631, -0.00736372, -0.00588334, -0.00911446, 0.053928, 0.0221688, 1.73593)))) / (9.0 * 9.0)

    assert(result < epsilon)

    theta = DenseMatrix((1.0, 0.0, 3.0), (0.0, 2.0, 1.0))

    result = sum(abs(hessian(x, theta, lambda = 0.0) - DenseMatrix((0.1145, 0.0643, 0.0434, -0.0867, -0.0590, -0.0392),
      (0.0643, 0.0802, 0.0335, -0.0590, -0.0730, -0.0319),
      (0.0434, 0.0335, 0.0654, -0.0392, -0.0319, -0.0552),
      (-0.0867, -0.0590, -0.0392, 0.1049, 0.0696, 0.0399),
      (-0.0590, -0.0730, -0.0319, 0.0696, 0.0975, 0.0325),
      (-0.0392, -0.0319, -0.0552, 0.0399, 0.0325, 0.0568)))) / (6.0 * 6.0)

    assert(result < epsilon)
  }

  test ("hessian") {
    val (xRaw, y) = mlclassEx4()
    val x = addBias(xRaw)
    val theta = DenseMatrix((0.01d, 0.01d, 0.01d))
    var result = sum(abs(MaxentModel.hessian(x, theta, lambda = 0.0) - DenseMatrix(
      (0.1905, 7.1009, 12.7284),
      (7.1009, 283.236, 478.956),
      (12.7284, 478.956, 868.698)))) / 9
    assert(result < epsilon)

    result = sum(abs(MaxentModel.hessian(x, theta, lambda = 80.0) - DenseMatrix(
      (0.1905, 7.1009, 12.7284),
      (7.1009, 284.236, 478.956),
      (12.7284, 478.956, 869.698)))) / 9

    assert(result < epsilon)
  }

  test ("update") {
    val (xRaw, yRaw) = mlclassEx4()
    val x = addBias(xRaw)
    val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
    val theta = DenseMatrix((0.01d, 0.01d, 0.01d))

    val result = sum(abs(MaxentModel.update(x, y, theta, lambda = 1.0, rho = 0.05) -
      DenseMatrix((-0.5585, 0.0146, 0.0150)))) / 3
    assert(result < epsilon)
  }

  test ("train") {
    {
      val (xRaw, yRaw) = mlclassEx4()
      val x = addBias(xRaw)
      val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
      val result = sum(abs(MaxentModel.internalTrain(x, y, lambda = 0.0, rho = 1.0, verbose = false) -
        DenseMatrix((-16.3787, 0.1483, 0.1589)))) / 3
      assert(result < epsilon)

    }
    {
      val (xRaw, yRaw) = mlclassEx5()
      val x = makePoly(xRaw)
      val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
      val model = new MaxentModel(C = 0.0, rho = 1.0).train(x, y, verbose = false)
      val result = abs(norm(model.theta.get.toDenseVector) - 7173)
      assert(result < 1.0)
      assert(model.stats.get.iterations == 13)
    }
    {
      val (xRaw, yRaw) = mlclassEx5()
      val x = makePoly(xRaw)
      val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
      val model = new MaxentModel(C = 1.0, rho = 1.0).train(x, y, verbose = false)
      val result = abs(norm(model.theta.get.toDenseVector) - 4.2400)
      assert(result < 1.0)
      assert(model.stats.get.iterations == 4)
    }
    {
      val (xRaw, yRaw) = mlclassEx5()
      val x = makePoly(xRaw)
      val y = DenseMatrix.horzcat(yRaw.toDenseMatrix.t, 1.0d - yRaw.toDenseMatrix.t)
      val model = new MaxentModel(C = 10.0, rho = 1.0).train(x, y, verbose = false)
      val result = abs(norm(model.theta.get.toDenseVector) - 0.9384)
      assert(result < 1.0)
      assert(model.stats.get.iterations == 3)
    }

    {
      val data = Data.iris()
      val x = Data.addBias(data._1)
      val y = Data.binarize(data._2)

      val model = new MaxentModel(rho = 1.0)
      model.train(x, y, verbose = false)
     assert(sum(abs(model.theta.get - DenseMatrix(
       (17.898793677952817, -0.7837392000252522, 1.242888374876643, -3.879039544159735, -1.6590203490205193),
       (11.748576317948881, 0.26054944160198223, -0.3358800265778448, -1.8331435276744747, -2.063617173979995)))) < epsilon)

      assert(sum(abs(model.predict(x).asInstanceOf[DenseVector[Double]] - DenseVector(
        0.0d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0,
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0))) < epsilon)
    }
  }

  test("irisBenchmark") {
    val data = Data.iris()
    val x = Data.addBias(data._1)
    val y = Data.binarize(data._2)
    val trainSplit = Seq(12, 39, 23, 5, 3, 29, 49, 47, 21, 30, 34, 48, 20, 45, 31, 27, 17, 22,
      41, 6, 40, 38, 42, 19, 26, 15, 35, 10, 46, 25, 0, 32, 1, 16, 4, 13,
      24, 33, 43, 18, 81, 65, 62, 50, 93, 92, 53, 58, 87, 55, 70, 72, 83,
      56, 52, 73, 78, 64, 68, 59, 74, 89, 67, 51, 66, 98, 90, 69, 95, 63,
      82, 54, 86, 85, 96, 97, 79, 71, 94, 80, 142, 147, 125, 145, 119, 101,
      141, 105, 129, 138, 122, 120, 139, 124, 134, 111, 148, 117, 132, 133,
      104, 130, 128, 115, 127, 131, 136, 112, 107, 143, 149, 106, 109, 108,
      102, 100, 126, 103, 146, 113)
    val testSplit = Seq(2, 7, 8, 9, 11, 14, 28, 36, 37, 44, 57, 60, 61, 75, 76, 77, 84, 88,
      91, 99, 110, 114, 116, 118, 121, 123, 135, 137, 140, 144)
    val xTrain = x(trainSplit,::).toDenseMatrix
    val yTrain = y(trainSplit,::).toDenseMatrix
    val xTest = x(testSplit,::).toDenseMatrix
    val yTest = y(testSplit,::).toDenseMatrix
    val model = new MaxentModel(rho = 1.0)
    model.train(xTrain, yTrain, verbose = false)
    val pred = model.predict(xTest).asInstanceOf[DenseVector[Double]]
    assert(abs(Metrics.accuracy(Data.factorize(yTest), pred) - 0.966667) < epsilon)
  }
}
