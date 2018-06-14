import breeze.linalg._
import breeze.numerics.log
import com.aratools.mimir.{Data, Stats}
import com.aratools.mimir.logistic.LogisticModel._
import com.aratools.mimir.Data.mlclassEx4

val (x, y) = Data.iris()

val theta = DenseMatrix((0.01d, 0.01d, 0.01d, 0.01d), (0.02d, 0.02d, 0.02d, 0.02d))

val logProb = DenseMatrix.zeros[Double](x.rows, theta.rows + 1)
logProb(::, 0 to theta.rows) := x*theta.t


