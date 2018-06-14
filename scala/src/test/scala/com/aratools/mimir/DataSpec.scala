package com.aratools.mimir

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.abs
import org.scalatest.FunSuite


class DataSpec extends FunSuite {
  val epsilon = 1e-3f

  test("testAddBias") {
    val result = sum(abs(Data.addBias(DenseMatrix((1.0, 2.0), (3.0, 4.0))) - DenseMatrix((1.0, 1.0, 2.0), (1.0 , 3.0, 4.0))))
    assert(result < epsilon)
  }

  test("binarize") {
    val result = sum(abs(Data.binarize(DenseVector("ba", "foo", "ba")) - DenseMatrix((1.0d, 0.0d), (0.0d, 1.0d), (1.0d, 0.0d))))
    assert(result < epsilon)
  }
}
