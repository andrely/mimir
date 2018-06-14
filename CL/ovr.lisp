(in-package :cl-user)

(defpackage :mimir/ovr
  (:use :common-lisp :mimir :mimir/linalg)
  (:export :make-ovr-classifier))

(in-package :mimir/ovr)

(defclass ovr-classifier ()
  ((base-model :accessor base-model :initarg :base-model :initform nil)
   (models     :accessor models     :initform nil)))

(defun make-ovr-classifier (base-model)
  (make-instance 'ovr-classifier :base-model base-model))

(defmethod train ((model ovr-classifier) x y &key (verbose t))
  (let* ((num-classes (array-dimension y 1)))
    (setf (models model)
	  (loop for i from 0 below num-classes
	     collect (train (replicate (base-model model)) x (m-col y i) :verbose verbose)))
    model))

(defmethod predict ((model ovr-classifier) x &key)
  (let* ((all-log-probs (loop for model in (models model)
			   collect (log-prob model x)))
	 (n (array-dimension x 0)))
    
    (make-array (list n)
		:initial-contents (loop for i from 0 below n
				     for log-probs = (loop for col in all-log-probs
							collect (aref col i))
				     collect (position (apply #'max log-probs) log-probs)))))
