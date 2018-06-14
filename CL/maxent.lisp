(in-package :cl-user)

(defpackage :mimir/maxent
  (:use :common-lisp :mimir :mimir/linalg :mimir/stat)
  (:export :maxent-model :make-maxent-model))

(in-package :mimir/maxent)

(defun log-prob-internal (x theta)
  (let* ((a (join-col (m-m-prod x (tr theta))
		      (make-array (array-dimension x 0) :initial-element 0.0)))
	 (lsm (log-sum-exp a)))
    (a- a lsm)))

(defun prob (x theta)
  (a-exp (log-prob-internal x theta)))

(defun cost (x y theta &key (l 1.0))
  (let* ((n (array-dimension x 0))
	 (reg (a/ (a* l (a-sum (a-expt (aslice theta
					       :all (loop for i from 1 below (array-dimension theta 1)
						       collect i))
				      2)))
		  (* 2 n))))
    (+ (/ (a-sum (a* -1.0 y (log-prob-internal x theta))) n) reg)))

(defun grad (x y theta &key (l 1.0))
  (let* ((err (a- (a-exp (log-prob-internal x theta)) y))
	 (n (array-dimension x 0))
	 (p (array-dimension theta 1))
	 (c (array-dimension theta 0))
	 (reg (let ((result (copy-array theta)))
		(loop for i from 0 below (array-dimension result 0)
		   do (setf (aref result i 0) 0.0d0))
		(a* (/ l n) result)))
	 (result (make-array (array-dimensions theta) :initial-element 0.0)))
    (loop for i from 0 below n
       do (loop for j from 0 below p
	     do (loop for k from 0 below c
		   do (incf (aref result k j)
			    (+ (* (aref err i k) (aref x i j)) (aref reg k j))))))
    (a/ result n)))

(defun hessian (x theta &key (l 1.0))
  (let* ((n (array-dimension x 0))
	 (c (array-dimension theta 0))
	 (p (array-dimension theta 1))
	 (mu (let ((result (make-array (list n c) :initial-element 0.))
		   (mu (a-exp (log-prob-internal x theta))))
	       (loop for i from 0 below n
		  do (loop for j from 0 below c
			do (setf (aref result i j)
				 (aref mu i j))))
	       result))
	 (reg (diag (a* (/ l n)
			(let ((result (make-array (array-total-size theta) :initial-element 1.0)))
			  (loop for i from 0 below (length result)
			     if (= (mod i p) 0)
			     do (setf (aref result i) 0.0))
			  result))))
	 (result (make-array (list (array-total-size theta) (array-total-size theta)) :initial-element 0.)))
    (loop for i from 0 below n
       do (setf result
		(a+ result
		     (kronecker-prod (a- (diag (m-row mu i)) (outer (m-row mu i) (m-row mu i)))
				     (outer (m-row x i) (m-row x i))))))
    (a+ (a/ result n) reg)))

(defun update (x y theta &key (l 1.0) (rho 0.05))
  (let* ((inv-hess (inverse (hessian x theta :l l)))
	 (grad (grad x y theta :l l)))
    (a- theta (a* (reshape (m-v-prod inv-hess (flatten grad))
			   (array-dimensions theta))
		  rho))))

(defun make-theta (c p)
  (make-array (list c p) :initial-contents (loop for i from 0 below c
					      collect (loop for j from 0 below p
							 collect (* (normal-deviate) .001d0)))
	      :element-type 'double-float))

(defparameter *stats* (list :type :maxent))

(defun train-internal (x y &key (l 1.0) (stats *stats*) (verbose t) (rho 0.05)
			     (max-iter 50))
  (let* ((p (array-dimension x 1))
	 (c (array-dimension y 1))
	 (theta (make-theta (1- c) p))
	 (iter 0)
	 (cost (cost x y theta :l l))
	 (old-cost 0.0))
    (loop until (or (< (abs (- cost old-cost)) .00001)
		    (> iter max-iter))
       do (incf iter)
       do (when verbose
	    (format t "iter ~a, cost ~a~%" iter cost))
       do (setf theta (update x y theta :l l :rho rho))
       do (setf old-cost cost)
       do (setf cost (cost x y theta :l l)))
    (when stats
      (if (getf stats :iterations)
	  (setf (getf stats :iterations) iter)
	  (nconc stats (list :iterations iter))))    
    theta))

(defclass maxent-model ()
  ((c        :accessor c        :initarg :c        :initform 1.0)
   (rho      :accessor rho      :initarg :rho      :initform 0.05)
   (stats    :accessor stats    :initarg :stats    :initform nil)
   (max-iter :accessor max-iter :initarg :max-iter :initform 50)
   (theta    :accessor theta    :initarg :theta    :initform nil)))

(defun make-maxent-model (&key (c 1.0) (rho 0.05) (max-iter 50))
  (make-instance 'maxent-model :c c :rho rho :max-iter max-iter))

(defmethod train ((model maxent-model) x y &key (verbose t))
  (let* ((stats (list :type :maxent))
	 (theta (train-internal x y :l (c model) :stats stats :rho (rho model)
				:max-iter (max-iter model) :verbose verbose)))
    (setf (theta model) theta)
    (setf (stats model) stats)

    model))

(defmethod predict ((model maxent-model) x &key)
  (let* ((n (array-dimension x 0))
	 (probs (log-prob-internal x (theta model)))
	 (c (array-dimension probs 1)))
    (make-array n
		:initial-contents (loop for i from 0 below n
				     collect (let ((max most-negative-double-float)
						   (idx 0))
					       (loop for j from 0 below c
						  if (> (aref probs i j) max)
						  do (setf max (aref probs i j)) and
						  do (setf idx j))
					       idx)))))

(defmethod log-prob ((model maxent-model) x &key)
  (log-prob-internal x (theta model)))

(defmethod replicate ((model maxent-model) &key)
  (make-maxent-model :c (c model) :rho (rho model) :max-iter (max-iter model)))
