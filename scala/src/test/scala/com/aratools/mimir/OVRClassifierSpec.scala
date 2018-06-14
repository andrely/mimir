package com.aratools.mimir

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.abs
import com.aratools.mimir.logistic.LogisticModel
import org.scalatest.FunSuite

class OVRClassifierSpec extends FunSuite {
  val epsilon = 1e-3f

  test("testTrain") {
    val data = Data.iris()
    val x = Data.addBias(data._1)
    val y = Data.binarize(data._2)

    val model = new OVRClassifier(new LogisticModel(rho = 1.0))
    model.train(x, y, verbose = false)
    assert(model.models.isDefined)
    assert(model.models.get.size == 3)
    assert(sum(abs(model.models.get(0).asInstanceOf[LogisticModel].theta.get - DenseVector(6.69036, -0.445019, 0.900008, -2.32352, -0.973446))) < epsilon)
    assert(sum(abs(model.models.get(1).asInstanceOf[LogisticModel].theta.get - DenseVector(5.58621, -0.17931, -2.12865, 0.696673, -1.27481))) < epsilon)
    assert(sum(abs(model.models.get(2).asInstanceOf[LogisticModel].theta.get - DenseVector(-14.4313, -0.394427, -0.51333, 2.93086, 2.41706))) < epsilon)

    assert(sum(abs(model.predict(x).asInstanceOf[DenseVector[Double]] - DenseVector(
      0.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))) < epsilon)
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
    val model = new OVRClassifier(new LogisticModel(rho = 1.0))
    model.train(xTrain, yTrain, verbose = false)
    val pred = model.predict(xTest).asInstanceOf[DenseVector[Double]]
    assert(abs(Metrics.accuracy(Data.factorize(yTest), pred) - 0.966667) < epsilon)
  }
}
