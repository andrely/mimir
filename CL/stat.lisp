(in-package :cl-user)

(defpackage :mimir/stat
  (:use :common-lisp :mimir :mimir/linalg)
  (:export :normal-deviate :mean :var :sd))

(in-package :mimir/stat)

(defparameter *prev-deviate* nil)

(defun normal-deviate ()
  (if (not (null *prev-deviate*))
      (let ((val *prev-deviate*))
	(setf *prev-deviate* nil)
	val)
      (let* (v1 v2 rsq)
	(loop
	   do (setf v1 (- (* 2.0 (random 1.0)) 1.0))
	   do (setf v2 (- (* 2.0 (random 1.0)) 1.0))
	   do (setf rsq (+ (* v1 v1) (* v2 v2))) 
	   until (and (>= 1.0 rsq) (> rsq 0.0)))
	(let* ((fac (sqrt (/ (* -2.0 (log rsq)) rsq))))
	  (setf *prev-deviate* (* v1 fac))
	  (* v2 fac)))))

(defgeneric mean (a &key))

(defmethod mean ((m simple-array) &key)
  (let* ((n (array-dimension m 0))
	 (p (array-dimension m 1))
	 (result (make-array p :initial-element 0.0d0)))
    (loop for i from 0 below n
       do (loop for j from 0 below p
	     do (incf (aref result j) (aref m i j))))
    (a/ result n)))

(defgeneric var (a &key))

(defmethod var ((m simple-array) &key)
  (let* ((n (array-dimension m 0))
	 (p (array-dimension m 1))
	 (mu (mean m))
	 (result (make-array p :initial-element 0.0d0)))
    (loop for i from 0 below n
       do (setf result
		(a+ result
		    (a-expt (a- (m-row m i) mu) 2))))
    (a/ result (1- n))))

(defgeneric sd (a &key))

(defmethod sd ((m simple-array) &key)
  (a-expt (var m) 0.5))
