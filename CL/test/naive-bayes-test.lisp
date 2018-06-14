(in-package :mimir-test)

(def-suite :naive-bayes :in :mimir-all)

(defparameter *package-path*
    (directory-namestring
     (asdf:component-pathname (asdf:find-system :mimir-test))))

(defparameter *xwin-train-file* (merge-pathnames (cl-fad:pathname-as-directory "../data/xwindows.train.svmlight") *package-path*))

(defparameter *xwin-test-file* (merge-pathnames (cl-fad:pathname-as-directory "../data/xwindows.test.svmlight") *package-path*))

(test (xwin-benchmark :suite :naive-bayes)
  (let* ((data (with-open-file (s *xwin-train-file* :direction :input)
                 (mimir/data:read-svmlight s)))
         (x-train (first data))
         (y-train (mimir/data:binarize (second data)))
         (test-data (with-open-file (s *xwin-test-file* :direction :input)
                      (mimir/data:read-svmlight s)))
         (x-test (first test-data))
         (y-test (mimir/data:binarize (second test-data)))
         (model (train (mimir/naive-bayes:make-nbb-model) x-train y-train :verbose nil)))
    (is (mimir::almost-= (mimir/metrics:accuracy (m-col y-test 1) (predict model x-test)) 0.8133))))
