(in-package :mimir-test)

(def-suite :ovr :in :mimir-all)

(test (train :suite :ovr)
  (let* ((data (iris))
	 (x (add-bias (getf data :x)))
	 (y (binarize (getf data :y)))
	 (model (train (mimir/ovr::make-ovr-classifier (make-logistic-model :rho 1.0 :c 1.0))
		       x y :verbose nil)))
    (is (= (length (mimir/ovr::models model)) 3))
    (is (mimir::array-almost-= (mimir/logistic::theta (elt (mimir/ovr::models model) 0))
			       #(6.69036 -0.445019 0.900008 -2.32352 -0.973446)))
    (is (= (getf (mimir/logistic::stats (elt (mimir/ovr::models model) 0))
		 :iterations)
	   7))
    (is (mimir::array-almost-= (mimir/logistic::theta (elt (mimir/ovr::models model) 1))
			       #(5.58621 -0.17931 -2.12865 0.696673 -1.27481)))
    (is (= (getf (mimir/logistic::stats (elt (mimir/ovr::models model) 1))
		 :iterations)
	   4))
    (is (mimir::array-almost-= (mimir/logistic::theta (elt (mimir/ovr::models model) 2))
			       #(-14.4313 -0.394427 -0.51333 2.93086 2.41706)))
    (is (= (getf (mimir/logistic::stats (elt (mimir/ovr::models model) 2))
		 :iterations)
	   7))
    (is (mimir::array-almost-= (predict model x)
			       #(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0
				 1.0 1.0 2.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 2.0 1.0
				 1.0 1.0 1.0 1.0 1.0 2.0 1.0 1.0 1.0 1.0 1.0 2.0 1.0 2.0 1.0 1.0 1.0 1.0
				 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0 1.0 2.0
				 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0
				 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0
				 2.0 2.0 2.0 2.0 2.0 2.0)))))

(test (iris-benchmark :suite :ovr)
  (let* ((data (iris))
	 (x (add-bias (getf data :x)))
	 (y (binarize (getf data :y)))
	 (train-split #(12 39 23 5 3 29 49 47 21 30 34 48 20 45 31 27 17 22
			41 6 40 38 42 19 26 15 35 10 46 25 0 32 1 16 4 13
			24 33 43 18 81 65 62 50 93 92 53 58 87 55 70 72 83
			56 52 73 78 64 68 59 74 89 67 51 66 98 90 69 95 63
			82 54 86 85 96 97 79 71 94 80 142 147 125 145 119 101
			141 105 129 138 122 120 139 124 134 111 148 117 132 133
			104 130 128 115 127 131 136 112 107 143 149 106 109 108
			102 100 126 103 146 113))
	 (test-split #(2 7 8 9 11 14 28 36 37 44 57 60 61 75 76 77 84 88
		       91 99 110 114 116 118 121 123 135 137 140 144))
	 (x-train (aslice x train-split :all))
	 (y-train (aslice y train-split :all))
	 (x-test (aslice x test-split :all))
	 (y-test (aslice y test-split :all))
	 (model (train (mimir/ovr::make-ovr-classifier (make-logistic-model :rho 1.0 :c 1.0))
		       x-train y-train :verbose nil))
	 (pred (predict model x-test)))
    (is (mimir::almost-= (mimir/metrics:accuracy (factorize y-test) pred) 0.966667))))
