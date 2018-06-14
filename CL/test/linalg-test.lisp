(in-package :mimir-test)

(def-suite :linalg :in :mimir-all)

(test (aslice :suite :linalg)
  (is (mimir::array-almost-=
       (aslice #2A((1.0 2.0 3.0) (4.0 5.0 6.0)) :all :all)
       #2A((1.0 2.0 3.0) (4.0 5.0 6.0))))
  (is (mimir::array-almost-=
       (aslice #2A((1.0 2.0 3.0) (4.0 5.0 6.0)) #(1) :all)
       #2A((4.0 5.0 6.0))))
  (is (mimir::array-almost-=
       (aslice #2A((1.0 2.0 3.0) (4.0 5.0 6.0)) :all #(1 2))
       #2A((2.0 3.0) (5.0 6.0))))
  (is (mimir::array-almost-=
       (aslice #2A((1.0 2.0 3.0) (4.0 5.0 6.0)) #(0) #(0 1))
       #2A((1.0 2.0)))))

(test (a+ :suite :linalg)
  (is (mimir::array-almost-= (a+ #(1.0 2.0 3.0) 1.5)
			     #(2.5 3.5 4.5)))
  (is (mimir::array-almost-= (a+ 0.3 #(0.0 -1.0 1.0 3.0))
			     #(0.3 -0.7 1.3 3.3))))
(test (m-v-prod :suite :linalg)
  (is (equalp (m-v-prod #2A((1 2) (3 4)) #(5 6))
              #(17 39)))
  (is (equalp (m-v-prod #2A((0 1 0 0) (1 0 1 0)) #(1 2 3 4))
              #(2 4))))

(test (permutations :suite :linalg)
  (is (= (length (mimir/linalg::permutations 3)) 3))
  (is (equalp (mimir/linalg::permutations 3) '((0) (1) (2))))
  (is (= (length (mimir/linalg::permutations 3 2)) 6))
  (is (equalp (mimir/linalg::permutations 3 2) '((0 0) (0 1) (1 0) (1 1) (2 0) (2 1))))
  (is (= (length (mimir/linalg::permutations 3 2 3)) 18))
  (is (equalp (mimir/linalg::permutations 3 2 3) '((0 0 0) (0 0 1) (0 0 2) (0 1 0) (0 1 1) (0 1 2)
                                               (1 0 0) (1 0 1) (1 0 2) (1 1 0) (1 1 1) (1 1 2)
                                               (2 0 0) (2 0 1) (2 0 2) (2 1 0) (2 1 1) (2 1 2)))))

(test (argwhere :suite :linalg)
  (is (equalp (argwhere #(0.0 2.0 1.0 4.0 3.0 5.0) (lambda (x) (> x 1.0)))
              #(1 3 4 5)))
  (is (equalp (argwhere #2A((0.0 2.0 1.0) (4.0 3.0 5.0)) (lambda (x) (> x 1.0)))
              #2A((0 1) (1 0) (1 1) (1 2)))))

(test (m-v-prod :suite :linalg)
  (is (equalp (m-v-prod #2A((0.0 1.0 0.0 0.0) (1.0 0.0 1.0 0.0))
                        #(1 2 3 4))
              #(2 4))))

(test (m-m-prod :suite :linalg)
  (is (equalp (m-m-prod #2A((0.0 1.0 0.0 0.0) (1.0 0.0 1.0 0.0))
                        #2A((1 2) (2 3) (3 4) (4 5)))
              #2A((2 3) (4 6)))))