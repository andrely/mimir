(in-package :cl-user)

(defpackage :mimir/naive-bayes
  (:use :common-lisp :mimir :mimir/linalg :mimir/stat)
  (:export :nbb-model :make-nbb-model))

(in-package :mimir/naive-bayes)

(defclass nbb-model ()
  ((alpha            :accessor alpha            :initarg :alpha            :initform 1.0)
   (num-classes      :accessor num-classes      :initarg :num-classes      :initform nil)
   (num-features     :accessor num-features     :initarg :num-features     :initform nil)
   (class-counts     :accessor class-counts     :initarg :class-counts     :initform nil)
   (feature-counts   :accessor feature-counts   :initarg :feature-counts   :initform nil)
   (class-log-prob   :accessor class-log-prob   :initarg :class-log-prob   :initform nil)
   (feature-log-prob :accessor feature-log-prob :initarg :feature-log-prob :initform nil)))

(defun make-nbb-model (&key (alpha 1.0))
  (make-instance 'nbb-model :alpha alpha))

(defmethod train ((model nbb-model) x y &key (verbose t))
  (with-slots (alpha num-classes num-features class-counts feature-counts class-log-prob feature-log-prob) model
    (setf num-classes (second (shape y)))
    (setf num-features (second (shape x)))
    (when verbose
      (format t "Collecting counts~%"))
    (setf class-counts (a-sum y :axis :col))
    (let ((class-indices (loop for c from 0 below num-classes
                               collect (argwhere (m-col y c) (lambda (e) (= e alpha))))))
      (setf feature-counts (make-array (list num-classes num-features)
                                       :initial-contents (loop for idx in class-indices
                                                               collect (a-sum (aslice x idx :all)
                                                                              :axis :col)))))
    (when verbose
      (format t "Populating parameters~%"))
    (setf feature-log-prob (make-array (list num-classes num-features)
                                       :initial-contents (loop for i from 0 below num-classes
                                                               collect (loop for j from 0 below num-features
                                                                             collect (- (log (+ (aref feature-counts i j) alpha))
                                                                                        (log (+ (aref class-counts i) (* 2 alpha))))))))
    
    (setf class-log-prob (a- (a-log class-counts) (log (a-sum class-counts))))

    model))

(defmethod log-prob ((model nbb-model) x &key)
  (with-slots (feature-log-prob class-log-prob) model
    (let* ((neg-log-prob (a-log (a- 1.0 (a-exp feature-log-prob))))
           (neg-log-prob-sum (a-sum neg-log-prob :axis :row))
           (log-prob (m-m-prod x (tr (a- feature-log-prob neg-log-prob)))))
      (loop for i from 0 below (first (shape log-prob))
            do (loop for j from 0 below (second (shape log-prob))
                     do (setf (aref log-prob i j) (+ (aref log-prob i j)
                                                     (aref neg-log-prob-sum j)
                                                     (aref class-log-prob j)))))
      log-prob)))

(defmethod predict ((model nbb-model) x &key)
  (let* ((log-prob (log-prob model x)))
    (make-array (array-dimension log-prob 0)
                :initial-contents (loop for i from 0 below (first (shape log-prob))
                                        for row = (coerce (m-row log-prob i) 'list)
                                        collect (position-if #'(lambda (x) (= x (apply #'max row))) row)))))

(defparameter *xwin-fn* #P"/Users/stinky/Documents/Work/mimir/data/xwindows.train.svmlight")

(defparameter *data* (with-open-file (s *xwin-fn* :direction :input) (mimir/data::read-svmlight s)))

(defparameter *test-data* (with-open-file (s #P"/Users/stinky/Documents/Work/mimir/data/xwindows.test.svmlight" :direction :input) (mimir/data::read-svmlight s)))

(defparameter *x-train* (first *data*))

(defparameter *y-train* (mimir/data:binarize (second *data*)))

(defparameter *x-test* (first *test-data*))

(defparameter *y-test* (mimir/data:binarize (second *test-data*)))

(defparameter *class-counts* (a-sum *y-train* :axis :col))
(defparameter *feat-counts* (second (shape *x-train*)))

(defparameter *num-classes* (array-dimension *y-train* 1))

(defparameter *class-indices* (loop for c from 0 below *num-classes*
                                    collect (mimir/linalg:argwhere (mimir/linalg:m-col *y-train* c) (lambda (x) (= x 1.0)))))

(defparameter *feature-counts* (make-array (list *num-classes* 600)
                                           :initial-contents (loop for idx in *class-indices*
                                                                   collect (mimir/linalg:a-sum (mimir/linalg:aslice *x-train* idx :all) :axis :col))))

(defparameter *feature-log-prob* (make-array (list *num-classes* *feat-counts*)
                                             :initial-contents (loop for i from 0 below *num-classes*
                                                                     collect (loop for j from 0 below *feat-counts*
                                                                                   collect (- (log (1+ (aref *feature-counts* i j)))
                                                                                              (log (+ (aref *class-counts* i) 2)))))))

(defparameter *class-log-prob* (a- (a-log *class-counts*) (log (a-sum *class-counts*))))

(defparameter *neg-log-prob* (a-log (a- 1.0 (a-exp *feature-log-prob*))))

(defparameter *neg-log-prob-sum* (a-sum *neg-log-prob* :axis :row))

(defparameter *log-prob* (m-m-prod *x-train* (tr (a- *feature-log-prob* *neg-log-prob*))))

(loop for i from 0 below (first (shape *log-prob*))
      do (loop for j from 0 below (second (shape *log-prob*))
               do (setf (aref *log-prob* i j) (+ (aref *log-prob* i j)
                                                 (aref *neg-log-prob-sum* j)
                                                 (aref *class-log-prob* j)))))

(defparameter *pred* (make-array (array-dimension *log-prob* 0) :initial-contents (loop for i from 0 below (first (shape *log-prob*))
                                                                                        for row = (coerce (m-row *log-prob* i) 'list)
                                                                                        collect (position-if #'(lambda (x) (= x (apply #'max row))) row))))
