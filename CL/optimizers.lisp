(in-package :cl-user)

(defpackage :mimir/optimizers
  (:use :common-lisp :mimir :mimir/linalg)
  (:export
   :newton :steepest :cgd-fr :sgd
   :line-search  :mini-batch))

(in-package :mimir/optimizers)

(defun newton (cost grad hessian theta &key (max-iter 50) (tol 0.0001d0) (verbose nil) (stats nil))
  (let* ((iter 1)
	 (j (funcall cost theta))
	 (old-j most-positive-double-float)
	 (iter-stats nil))
    (loop until (or (< (abs (- j old-j)) tol)
		    (>= iter max-iter))
       for inv-h = (inverse (funcall hessian theta))
       for g = (funcall grad theta)
       do (setf theta (a- theta (reshape (m-v-prod inv-h (flatten g))
					 (array-dimensions theta))))
       do (setf old-j j)
       do (setf j (funcall cost theta))
       when verbose do (format t "iter ~a, cost ~a~%" iter j)
       when stats do (push (list iter j) iter-stats)
       do (incf iter))
    
    (when stats
      (if (getf stats :iterations)
	  (setf (getf stats :iterations) (nreverse iter-stats))
	  (nconc stats (list :iterations (nreverse iter-stats)))))    
    theta))

(defun steepest (cost grad theta &key (rho 0.05) (max-iter 50) (tol 0.0001) (verbose nil) (stats nil))
  (let* ((iter 1)
	 (j (funcall cost theta))
	 (old-j most-positive-double-float)
	 (iter-stats nil))
    (loop until (or (< (abs (- j old-j)) tol)
		    (>= iter max-iter))
       for g = (funcall grad theta)
       do (setf theta (a- theta (a* rho g)))
       do (setf old-j j)
       do (setf j (funcall cost theta))
       when verbose do (format t "iter ~a, cost ~a~%" iter j)
       when stats do (push (list iter j) iter-stats)
       do (incf iter))
    
    (when stats
      (if (getf stats :iterations)
	  (setf (getf stats :iterations) (nreverse iter-stats))
	  (nconc stats (list :iterations (nreverse iter-stats)))))    
    theta))

(defconstant +gold+ 1.618034)
(defconstant +tiny+ 1e-20)
(defconstant +glimit+ 100)

(defun sign (a b)
  (if (= b 0.)
      a
      (/ (* a b) (abs b))))

(defun overflow-guard (func val)
  (handler-case (funcall func val)
    (floating-point-overflow () most-positive-double-float)))

(defun bracket (a b func)
  (let* ((ax a)
	 (bx b)
	 (fa (funcall func a))
	 (fb (funcall func b)))
    (when (> fb fa)
      (rotatef ax bx)
      (rotatef fa fb))
    (let* ((cx (+ bx (* +gold+ (- bx ax))))
	   (fc (funcall func cx))
	   fu)
      (loop while (> fb fc)
	 for r = (* (- bx ax) (- fb fc))
	 for q = (* (- bx cx) (- fb fa))
	 for u = (- bx (/ (- (* (- bx cx) q) (* (- bx ax) r))
			  (* 2.0 (sign (max (abs (- q r)) +tiny+) (- q r)))))
	 for ulim = (+ bx (* +glimit+ (- cx bx)))
	 do (cond ((> (* (- bx u) (- u cx)) 0.0)
		   (progn
		     (setf fu (funcall func u))
		     (cond ((< fu fc)
			    (progn
			      (setf ax bx
				    bx u
				    fa fb
				    fb fu)
			      (return-from bracket (list ax bx cx fa fb fc))))
			   ((> fu fb)
			    (progn
			      (setf cx u
				    fc fu)
			      (return-from bracket (list ax bx cx fa fb fc)))))
		     (setf u (+ cx (* +gold+ (- cx bx))))
		     (setf u (funcall func u))))
		  ((> (* (- cx u) (- u ulim)) 0.0)
		   (setf fu (funcall func u))
		   (when (< fu fc)
		     (let* ((new-u (+ u (* +gold+ (- u cx)))))
		       (setf bx cx
			     cx u
			     u new-u)
		       (setf fb fc
			     fc fu
			     fu (funcall func u)))))
		  ((> (* (- u ulim) (- ulim cx)) 0.0)
		   (setf u ulim
			 fu (funcall func u)))
		  (t
		   (setf u (+ cx (* +gold+ (- cx bx)))
			 fu (funcall func u))))
	 do (setf ax bx
		  bx cx
		  cx u)
	 do (setf fa fb
		  fb fc
		  fc fu))
      (if (> ax cx)
	  (list cx bx ax fc fb fa)
	  (list ax bx cx fa fb fc)))))

(defconstant +r+ 0.61803399)
(defconstant +c+ (- 1.0 +r+))
(defconstant +tol+ 3.0e-8)

(defun golden-section (bracket func)
  (destructuring-bind (ax bx cx fa fb fc) bracket
    (declare (ignore fa fb fc))
    (let* ((iter 0)
	   (x0 ax)
	   (x3 cx)
	   x1 x2)
      (if (> (abs (- cx bx)) (abs (- bx ax)))
	  (setf x1 bx
		x2 (- bx (* +c+ (- cx bx))))
	  (setf x2 bx
		x1 (- bx (* +c+ (- bx ax)))))
      (let* ((f1 (funcall func x1))
	     (f2 (funcall func x2)))
	(loop
	   while (and (> (abs (- x3 x0))
			 (* +tol+ (abs (- x2 x1))))
		      (< iter 50))
	   do (incf iter)
	   if (< f2 f1)
	   do (let* ((x2-new (+ (* +r+ x2) (* +c+ x3))))
		(setf x0 x1
		      x1 x2
		      x2 x2-new)
		(setf f1 f2
		      f2 (funcall func x2)))
	   else
	   do (let* ((x1-new (+ (* +r+ x1) (* +c+ x0))))
		(setf x3 x2
		      x2 x1
		      x1 x1-new)
		(setf f2 f1
		      f1 (funcall func x1))))
	(if (< f1 f2)
	    (values x1 f1)
	    (values x2 f2))))))

(defun line-search (x y func)
  (labels ((inner-func (val)
	     (overflow-guard func val)))
    (golden-section (bracket x y #'inner-func) #'inner-func)))

(defun cgd-fr (cost grad theta &key (max-iter 50) (tol 1.0d-5) (verbose nil) (stats nil))
  (let* ((iter 1)
	 (j (funcall cost theta))
	 (iter-stats nil)
         (g (funcall grad theta))
         (p (a- 0.0 g))
         g-prev beta)
    (loop until (or (< (norm g) tol)
		    (>= iter max-iter))
       for alpha = (line-search 0.0 1.0
                                #'(lambda (alpha)
                                    (funcall cost (a+ theta (a* alpha p)))))
       do (setf theta (a+ theta (a* alpha p)))
       do (setf g-prev g)
       do (setf g (funcall grad theta))
       do (setf beta (/ (dot (flatten g) (flatten g))
                        (dot (flatten g-prev) (flatten g-prev))))
       do (setf p (a+ g (a* beta p)))
       do (setf j (funcall cost theta))
       when verbose do (format t "iter ~a, cost ~a~%" iter j)
       when stats do (push (list iter j) iter-stats)
       do (incf iter))
    
    (when stats
      (if (getf stats :iterations)
	  (setf (getf stats :iterations) (nreverse iter-stats))
	  (nconc stats (list :iterations (nreverse iter-stats)))))    
    theta))

(defun mini-batch (x y &key (batch-size 10))
  #'(lambda ()
      (let* ((n (array-dimension x 0))
             (idx (make-array n :initial-contents (loop for i from 0 below n collect i)))
             (batch-num (ceiling (/ n batch-size)))
             (batch-idx (loop for i from 0 to (1- batch-num)
                              collect (list (* i batch-size) (min (* (1+ i) batch-size) n)))))
        (cl-variates:shuffle-elements! idx)
        (loop for (start end) in batch-idx
              for subscript = (loop for i from start below end collect (aref idx i))
              collect (list (aslice x subscript :all)
                            (aslice y subscript))))))

(defun sgd (cost grad batch-func theta &key (rho 0.05) (max-iter 50) (tol 0.0001) (verbose nil) (stats nil))
  (let* ((iter 1)
	 (j (funcall cost theta))
	 (old-j most-positive-double-float)
	 (iter-stats nil))
    (loop until (or (< (abs (- j old-j)) tol)
		    (>= iter max-iter))
       do (loop for batch in (funcall batch-func)
                do (setf theta (a- theta (a* rho (funcall grad theta batch)))))
       do (setf old-j j)
       do (setf j (funcall cost theta))
       when verbose do (format t "iter ~a, cost ~a~%" iter j)
       when stats do (push (list iter j) iter-stats)
       do (incf iter))
    
    (when stats
      (if (getf stats :iterations)
	  (setf (getf stats :iterations) (nreverse iter-stats))
	  (nconc stats (list :iterations (nreverse iter-stats)))))    
    theta))