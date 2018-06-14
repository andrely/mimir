(in-package :cl-user)

(defpackage :mimir/logistic
  (:use :common-lisp :mimir :mimir/linalg :mimir/stat)
  (:export :logistic-model :make-logistic-model))

(in-package :mimir/logistic)

(defparameter *stats* (list :type :logistic))


(defun a (x theta)
  (m-v-prod x theta))

(defun prob (x theta)
  (a-exp (log-prob-internal x theta)))

(defun log-prob-internal (x theta)
  (let* ((a (a x theta)))
    (a- a (log-sum-exp (join-col a (make-array (array-total-size a) :initial-element 0.0d0))))))

(defun cost (x y theta &key (l 1.0))
  (let* ((h (log-prob-internal x theta))
	 (a (a x theta))
	 (n (array-dimension x 0))
	 (reg (/ (* l (loop for i from 1 below (length theta)
			 sum (* (aref theta i) (aref theta i))))
		 2.0)))
    (/ (+ (loop for h-i across h
	     for a-i across a
	     for y-i across y
	     summing (- (+ (* y-i h-i)
			   (* (- 1.0 y-i) (- h-i a-i)))))
	  reg)
       (float n))))

(defun grad (x y theta &key (l 1.0))
  (let* ((h (prob x theta))
	 (n (array-dimension x 0))
	 (p (array-dimension x 1))
	 (err (a- h y))
	 (reg (a/ (a* theta l) n))
	 (result (make-array (array-dimensions theta) :initial-element 0.0)))
    (setf (aref reg 0) 0.0)
    
    (loop for i from 0 below n
	  do (loop for j from 0 below p
		do (setf (aref result j)
			 (+ (aref result j)
			    (* (aref err i) (aref x i j))
			    (aref reg j)))))
    (a/ result n)))

(defun hessian (x theta &key (l 1.0))
  (let* ((h (log-prob-internal x theta))
	 (a (a x theta))
	 (n (array-dimension x 0))
	 (p (array-dimension x 1))
	 (result (make-array (list p p) :initial-element 0.0))
	 (reg (diag (make-array (length theta) :initial-element l))))
    (setf (aref reg 0 0) 0.0)

    (loop for i from 0 below n
       do (setf result (a+ result (a* (outer (m-row x i)
					     (m-row x i))
					(exp (- (* 2.0 (aref h i)) (aref a i)))))))
    (setf result (a+ result reg))
    
    (a/ result n)))

(defun update (x y theta &key (l 1.0) (rho 0.05))
  (let* ((inv-hess (inverse (hessian x theta :l l)))
	 (grad (grad x y theta :l l)))
    (a- theta (a* (m-v-prod inv-hess grad) rho))))

(defun make-theta (p)
  (make-array p :initial-contents (loop for i from 0 below p
				     collect (* (normal-deviate) .001d0))))

(defun train-internal (x y &key (l 1.0) (stats *stats*) (verbose t) (rho 0.05)
			     (max-iter 50))
  (let* ((p (array-dimension x 1))
	 (theta (make-theta p))
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

(defclass logistic-model ()
  ((c        :accessor c        :initarg :c        :initform 1.0)
   (rho      :accessor rho      :initarg :rho      :initform 0.05)
   (stats    :accessor stats    :initarg :stats    :initform nil)
   (max-iter :accessor max-iter :initarg :max-iter :initform 50)
   (theta    :accessor theta    :initarg :theta    :initform nil)))

(defun make-logistic-model (&key (c 1.0) (rho 0.05) (max-iter 50))
  (make-instance 'logistic-model :c c :rho rho :max-iter max-iter))

(defmethod train ((model logistic-model) x y &key (verbose t))
  (let* ((stats (list :type :logistic))
	 (theta (train-internal x y :l (c model) :stats stats :rho (rho model)
				:max-iter (max-iter model) :verbose verbose)))
    (setf (theta model) theta)
    (setf (stats model) stats)

    model))

(defmethod predict ((model logistic-model) x &key)
  (let* ((probs (log-prob-internal x (theta model)))
	 (cutoff (log 0.5)))
    (make-array (array-dimension x 0)
		:initial-contents (loop for p across probs
				       collect (if (> p cutoff) 1.0 0.0)))))

(defmethod log-prob ((model logistic-model) x &key)
  (log-prob-internal x (theta model)))

(defmethod replicate ((model logistic-model) &key)
  (make-logistic-model :c (c model) :rho (rho model) :max-iter (max-iter model)))
