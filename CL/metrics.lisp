(in-package :cl-user)

(defpackage :mimir/metrics
  (:use :common-lisp :mimir)
  (:export :accuracy))

(in-package :mimir/metrics)

(defun accuracy (true pred)
  (/ (loop for i from 0 below (length true)
	sum (if (equalp (elt true i) (elt pred i)) 1.0 0.0))
     (length true)))
