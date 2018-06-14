(in-package :mimir-test)

(def-suite :data :in :mimir-all)

(test (add-bias :suite :data)
  (is (mimir::array-almost-= (add-bias #2A((1.0 2.0) (3.0 4.0)))
			     #2A((1.0 1.0 2.0) (1.0 3.0 4.0)))))

(test (binarize :suite :data)
  (is (mimir::array-almost-= (binarize #("ba" "foo" "ba"))
			     #2A((1.0 0.0)
				 (0.0 1.0)
				 (1.0 0.0)))))
