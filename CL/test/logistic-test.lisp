(in-package :mimir-test)

(def-suite :logistic :in :mimir-all)

(test (prob :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (theta #(0.01 0.01 0.01)))
    (is (mimir::array-almost-= (mimir/logistic::prob x theta)
			       #(0.7790 0.7747 0.8030 0.7875 0.7604
				 0.7712 0.7586 0.7649 0.7982 0.7850
				 0.7613 0.7721 0.7841 0.7455 0.7799
				 0.7649 0.7721 0.7941 0.7613 0.7521
				 0.7512 0.7558 0.7398 0.7604 0.7359
				 0.7211 0.7446 0.7841 0.7150 0.7436
				 0.7799 0.7408 0.7512 0.7658 0.7577
				 0.7738 0.7221 0.7558 0.7586 0.7455
				 0.7120 0.7130 0.7016 0.6846 0.7221
				 0.7369 0.7058 0.7079 0.6559 0.7211
				 0.7231 0.6964 0.7037 0.7465 0.7408
				 0.7530 0.7301 0.6835 0.7301 0.7058
				 0.7604 0.7613 0.7340 0.6974 0.7271
				 0.7110 0.7658 0.7291 0.7568 0.7465
				 0.7359 0.7503 0.6932 0.7417 0.6963
				 0.7037 0.7389 0.7191 0.7099 0.7359)))))

(test (log-prob :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (theta #(0.01 0.01 0.01)))
    (is (mimir::array-almost-=
	 (mimir/logistic::log-prob-internal x theta)
	 #(-0.2497 -0.2553 -0.2194 -0.2389 -0.2739 -0.2598 -0.2763 -0.2679 -0.2254 -0.2421
	   -0.2727 -0.2587 -0.2432 -0.2936 -0.2486 -0.2679 -0.2587 -0.2305 -0.2727 -0.2848
	   -0.2861 -0.2799 -0.3014 -0.2739 -0.3066 -0.327 -0.2949 -0.2432 -0.3354 -0.2962
	   -0.2486 -0.3001 -0.2861 -0.2668 -0.2775 -0.2564 -0.3256 -0.2799 -0.2763 -0.2936
	   -0.3397 -0.3383 -0.3544 -0.3789 -0.3256 -0.3053 -0.3484 -0.3455 -0.4218 -0.327
	   -0.3242 -0.3619 -0.3514 -0.2924 -0.3001 -0.2836 -0.3146 -0.3805 -0.3146 -0.3484
	   -0.2739 -0.2727 -0.3092 -0.3604 -0.3187 -0.3412 -0.2668 -0.316 -0.2787 -0.2924
	   -0.3066 -0.2873 -0.3665 -0.2988 -0.3619 -0.3514 -0.3027 -0.3298 -0.3426 -0.3066)))))

(test (cost :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (y (getf data :y))
	 (theta #(0.01 0.01 0.01)))
    (is (mimir::almost-= (mimir/logistic::cost x y theta :l 0.0) 0.7785))
    (is (mimir::almost-= (mimir/logistic::cost x y theta :l 8000.0) 0.7885))
    (is (mimir::almost-= (mimir/logistic::cost x y #(-0.23201762 -6.826957 -13.900436) :l 0.0) 651.61406))))

(test (grad :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (y (getf data :y))
	 (theta #(0.01 0.01 0.01)))
    (is (mimir::array-almost-= (mimir/logistic::grad x y theta :l 0.0) #(0.2420 6.8370 13.9104)))
    (is (mimir::array-almost-= (mimir/logistic::grad x y theta :l 80.0) #(0.2420 6.8470 13.9204)))))

(test (hessian :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (theta #(0.01 0.01 0.01)))
    (is (mimir::array-almost-= (mimir/logistic::hessian x theta :l 0.0)
			       #2A ((0.1905 7.1009 12.7284)
				    (7.1009 283.236 478.956)
				    (12.7284 478.956 868.698))))
    (is (mimir::array-almost-= (mimir/logistic::hessian x theta :l 80.0)
			       #2A ((0.1905 7.1009 12.7284)
				    (7.1009 284.236 478.956)
				    (12.7284 478.956 869.698))))))

(test (update :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (y (getf data :y))
	 (theta #(0.01 0.01 0.01)))
    (is (mimir::array-almost-= (mimir/logistic::update x y theta :l 0.0) #(-0.5585 0.0146 0.0150)))))

(test (train :suite :logistic)
  (let* ((data (mimir/data::mlclass-ex4))
	 (x (add-bias (getf data :x)))
	 (y (getf data :y))
	 (model (train (make-logistic-model :c 0.0 :rho 1.0) x y :verbose nil))
	 (theta (mimir/logistic::theta model))
	 (iterations (getf (mimir/logistic::stats model) :iterations)))
    (is (mimir::array-almost-= theta #(-16.3787 0.1483 0.1589)))
    (is (= iterations 5)))
  (let* ((data (mlclass-ex5))
	 (x (make-poly (getf data :x)))
	 (y (getf data :y)))
    (let* ((model (train (make-logistic-model :c 0.0 :rho 1.0) x y :verbose nil))
	   (theta (mimir/logistic::theta model))
	   (iterations (getf (mimir/logistic::stats model) :iterations)))
      (is (mimir::almost-= (norm theta) 7173 :epsilon 1.))
      (is (= iterations 13)))
    (let* ((model (train (make-logistic-model :c 1.0 :rho 1.0) x y :verbose nil))
	   (theta (mimir/logistic::theta model))
	   (iterations (getf (mimir/logistic::stats model) :iterations)))
      (is (mimir::almost-= (norm theta) 4.240))
      (is (= iterations 4)))
    (let* ((model (train (make-logistic-model :c 10.0 :rho 1.0) x y :verbose nil))
	   (theta (mimir/logistic::theta model))
	   (iterations (getf (mimir/logistic::stats model) :iterations)))
      (is (mimir::almost-= (norm theta) 0.9384))
      (is (= iterations 3)))))
