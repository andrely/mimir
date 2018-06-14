(in-package :cl-user)

(defpackage :mimir/linalg
  (:use :common-lisp :mimir)
  (:export :copy-array :diag :flatten :reshape :tr
           :shape :argwhere
	   :aslice :a+ :a- :a* :a/
	   :m-row :m-col :join-col
	   :a-exp :a-log :a-expt
	   :v-max :v-min :a-sum
	   :m-v-prod :m-m-prod
	   :norm :dot :outer :kronecker-prod :inverse
	   :log-sum-exp))

(in-package :mimir/linalg)

(defgeneric shape (a &key))

(defmethod shape ((v simple-vector) &key)
  (array-dimensions v))

(defmethod shape ((m simple-array) &key)
  (array-dimensions m))

(defun permutations (&rest dims)
  (cond ((= (length dims) 1)
         (loop for i from 0 below (first dims)
               collect (list i)))
        (t (loop for i from 0 below (first dims)
                 appending (loop for val in (apply #'permutations (rest dims))
                                 collect (cons i val))))))

(defgeneric argwhere (a pred &key))

(defmethod argwhere ((a simple-array) pred &key)
  (let ((cols (length (array-dimensions a)))
        (result (loop for idx in (apply #'permutations (array-dimensions a))
                      for elt = (apply #'aref (cons a idx))
                      if (funcall pred elt)
                      collect idx)))
    (make-array (list (length result) cols) :initial-contents result)))

(defmethod argwhere ((a simple-vector) pred &key)
  (let ((result (loop for elt across a
                      for i from 0
                      if (funcall pred elt)
                      collect i)))
    (make-array (list (length result)) :initial-contents result)))

(defun copy-array (array &key
                   (element-type (array-element-type array))
                   (fill-pointer (and (array-has-fill-pointer-p array)
                                      (fill-pointer array)))
                   (adjustable (adjustable-array-p array)))
  "Returns an undisplaced copy of ARRAY, with same fill-pointer and
adjustability (if any) as the original, unless overridden by the keyword
arguments."
  (let* ((dimensions (array-dimensions array))
         (new-array (make-array dimensions
                                :element-type element-type
                                :adjustable adjustable
                                :fill-pointer fill-pointer)))
    (dotimes (i (array-total-size array))
      (setf (row-major-aref new-array i)
            (row-major-aref array i)))
    new-array))

(defun diag (v)
  (let ((result (make-array (list (array-total-size v) (array-total-size v)) :initial-element 0.)))
    (loop for e across v
	 for i from 0
       do (setf (aref result i i) e))
    result))

(defun flatten (m)
  (when (= (array-rank m) 1)
    (return-from flatten m))
  (let* ((k (array-dimension m 0))
	 (l (array-dimension m 1))
	 (result (make-array (* k l)
			     :initial-element 0.0)))
    (loop for i from 0 below k
       do (loop for j from 0 below l
	     do (setf (aref result (+ j (* i l))) (aref m i j))))
    result))

(defgeneric reshape (a dimensions &key))

(defmethod reshape ((v simple-vector) dimensions &key)
  (assert (= (array-total-size v) (apply #'* dimensions)))
  (ecase (length dimensions)
    (1 v)
    (2 (let* ((result (make-array dimensions :initial-element 0.0))
	      (rows (first dimensions))
	      (cols (second dimensions)))
	 (loop for i from 0 below rows
	    do (loop for j from 0 below cols
		  do (setf (aref result i j) (aref v (+ j (* i cols))))))
	 result))))

(defmethod reshape ((m simple-array) dimensions &key)
  (ecase (length dimensions)
    (1 (let ((result (make-array dimensions :initial-element 0.0))
	     (rows (array-dimension m 0))
	     (cols (array-dimension m 1)))
	 (loop for i from 0 below rows
	    do (loop for j from 0 below cols
		  do (setf (aref result (+ j (* i cols))) (aref m i j ))))
	 result))
    (2 (reshape (flatten m) dimensions))))

(defun m-row (m idx)
  (let* ((len (array-dimension m 1))
	 (result (make-array len :initial-element 0.)))
    (loop for i from 0 below len
	 do (setf (aref result i)
		  (aref m idx i)))
    result))

(defun m-col (m idx)
  (let* ((len (array-dimension m 0))
	 (result (make-array len :initial-element 0.)))
    (loop for i from 0 below len
	 do (setf (aref result i)
		  (aref m i idx)))
    result))

(defgeneric join-col (a1 a2 &key))

(defmethod join-col ((v1 simple-vector) (v2 simple-vector) &key)
  (assert (= (length v1) (length v2)))
  (let* ((result (make-array (list (length v1) 2) :initial-element 0.0)))
    (loop for i from 0 below (length v1)
       do (setf (aref result i 0) (aref v1 i))
       do (setf (aref result i 1) (aref v2 i)))
    result))

(defmethod join-col ((m simple-array) (v simple-array) &key)
  (let* ((k (array-dimension m 0))
	 (l (array-dimension m 1))
	 (result (make-array (list k (1+ l)) :initial-element 0.0)))
    (assert (= k (length v)))
    (loop for i from 0 below k
       do (setf (aref result i l) (aref v i))
       do (loop for j from 0 below l
	     do (setf (aref result i j) (aref m i j))))
    result))

(defun m-m-elt-op (m1 m2 op)
  (assert (= (array-dimension m1 0) (array-dimension m2 0)))
  (assert (= (array-dimension m1 1) (array-dimension m2 1)))
  (let* ((n (array-dimension m1 0))
	 (p (array-dimension m1 1))
	 (result (make-array (list n p) :initial-element 0.)))
    (loop for i from 0 below n
       do (loop for j from 0 below p
	     do (setf (aref result i j) (funcall op (aref m1 i j) (aref m2 i j)))))
    result))

(defun m-elt-op (m op arg)
  (let* ((n (array-dimension m 0))
	 (p (array-dimension m 1))
	 (result (make-array (list n p) :initial-element 0.)))
    (loop for i from 0 below n
       do (loop for j from 0 below p
	     do (setf (aref result i j) (funcall op (aref m i j) arg))))
    result))

(defun elt-m-op (m op arg)
  (let* ((n (array-dimension m 0))
	 (p (array-dimension m 1))
	 (result (make-array (list n p) :initial-element 0.)))
    (loop for i from 0 below n
       do (loop for j from 0 below p
	     do (setf (aref result i j) (funcall op arg (aref m i j)))))
    result))

(defun m-op (m op)
  (let* ((n (array-dimension m 0))
	 (p (array-dimension m 1))
	 (result (make-array (list n p) :initial-element 0.)))
    (loop for i from 0 below n
       do (loop for j from 0 below p
	     do (setf (aref result i j) (funcall op (aref m i j)))))
    result))

(defgeneric aslice (m &rest subscripts))

(defmethod aslice ((v simple-vector) &rest subscripts)
  (assert (= (length subscripts) 1))
  (let* ((subscript (first subscripts))
         (result (make-array (length subscript) :initial-element 0.0d0)))
    (loop for i from 0 below (length subscript)
          for s in subscript
          do (setf (aref result i) (aref v s)))
    result))

(defmethod aslice ((m simple-array) &rest subscripts)
  (if (= (length subscripts) 1)
    (aslice m (first subscripts) :all)
    (let* ((sub-1 (first subscripts))
           (sub-2 (second subscripts))
           (result-size (list (if (eql sub-1 :all) (array-dimension m 0) (length sub-1))
                              (if (eql sub-2 :all) (array-dimension m 1) (length sub-2))))
           (result (make-array result-size :initial-element 0.0)))
      (loop for i from 0 below (first result-size)
            for k = (if (eql sub-1 :all) i (elt sub-1 i))
            do (loop for j from 0 below (second result-size)
                     for l = (if (eql sub-2 :all) j (elt sub-2 j))
                     do (setf (aref result i j) (aref m k l))))
      result)))

(defgeneric apply-bin-op (op x y &key))

(defmethod apply-bin-op (op (x vector) (y number) &key)
  (let ((result (make-array (length x) :initial-element 0.)))
    (loop for i from 0 below (length x)
       do (setf (aref result i) (funcall op (aref x i) y)))
    result))

(defmethod apply-bin-op (op (x number) (y vector) &key)
  (let ((result (make-array (length y) :initial-element 0.)))
    (loop for i from 0 below (length y)
       do (setf (aref result i) (funcall op x (aref y i))))
    result))

(defmethod apply-bin-op (op (x vector) (y vector) &key)
  (assert (= (length x) (length y)))
  (let ((result (make-array (length x) :initial-element 0.)))
    (loop for i from 0 below (length x)
       do (setf (aref result i) (funcall op (aref x i) (aref y i))))
    result))

(defmethod apply-bin-op (op (x number) (y number) &key)
  (funcall op x y))

(defmethod apply-bin-op (op (x array) (y array) &key)
  (m-m-elt-op x y op))

(defmethod apply-bin-op (op (x array) (y number) &key)
  (m-elt-op x op y))

(defmethod apply-bin-op (op (x number) (y array) &key)
  (elt-m-op y op x))

(defmethod apply-bin-op (op (x array) (y vector) &key)
  (assert (= (array-dimension x 0) (length y)))
  (let* ((result (make-array (array-dimensions x) :initial-element 0.0)))
    (loop for i from 0 below (array-dimension x 0)
       do (loop for j from 0 below (array-dimension x 1)
	     do (setf (aref result i j)
		      (funcall op (aref x i j) (aref y i)))))
    result))

(defun a+-bin (x y)
  (apply-bin-op #'+ x y))

(defun a+ (&rest args)
  (reduce #'a+-bin args))

(defun a--bin (x y)
  (apply-bin-op #'- x y))

(defun a- (&rest args)
  (reduce #'a--bin args))

(defun a*-bin (x y)
  (apply-bin-op #'* x y))

(defun a* (&rest args)
  (reduce #'a*-bin args))

(defgeneric a/-bin (x y &key))

(defmethod a/-bin (x y &key)
  (apply-bin-op #'/ x y))

(defmethod a/-bin ((x simple-array) (y simple-vector) &key)
  (assert (= (array-dimension x 0) (length y)))
  (let* ((result (make-array (array-dimensions x) :initial-element 0.0)))
    (loop for i from 0 below (array-dimension x 0)
       do (loop for j from 0 below (array-dimension x 1)
	     do (setf (aref result i j)
		      (/ (aref x i j)
			 (aref y i)))))
    result))

(defun a/ (&rest args)
  (reduce #'a/-bin args))

(defgeneric a-exp (a &key))

(defmethod a-exp ((v simple-vector) &key)
  (let ((result (make-array (length v) :initial-element 0.)))
    (loop for i from 0 below (length v)
       do (setf (aref result i)
		(exp (aref v i))))
    result))

(defmethod a-exp ((m simple-array) &key)
  (m-op m #'exp ))

(defgeneric a-log (a &key))

(defmethod a-log ((v simple-vector) &key)
  (let ((result (make-array (length v) :initial-element 0.)))
    (loop for i from 0 below (length v)
       do (setf (aref result i)
		(log (aref v i))))
    result))

(defmethod a-log ((m simple-array) &key)
  (m-op m #'log))

(defgeneric a-expt (a pow &key))

(defmethod a-expt ((v simple-vector) pow &key)
  (let ((result (make-array (length v) :initial-element 0.)))
    (loop for i from 0 below (length v)
       do (setf (aref result i)
		(expt (aref v i) pow)))
    result))

(defmethod a-expt ((m simple-array) pow &key)
  (m-elt-op m #'expt pow))

(defun v-max (u)
  (let ((max least-negative-double-float))
    (loop for i across u
       if (> i max)
       do (setf max i))
    max))

(defun v-min (u)
  (let ((min most-positive-double-float))
    (loop for i across u
       if (< i min)
       do (setf min i))
    min))

(defgeneric a-sum (a &key))

(defmethod a-sum ((v simple-vector) &key)
  (loop for i across v
     summing i))

(defmethod a-sum ((m simple-array) &key (axis :all))
  (let* ((k (array-dimension m 0))
	 (l (array-dimension m 1)))
    (cond ((eql axis :all)
	   (loop for i from 0 below k
	      sum (loop for j from 0 below l
		     sum (aref m i j))))
	  ((eql axis :row)
	   (let* ((result (make-array k :initial-element 0.0)))
	     (loop for i from 0 below k
		do (loop for j from 0 below l
		      do (incf (aref result i) (aref m i j))))
	     result))
	  ((eql axis :col)
	   (let* ((result (make-array l :initial-element 0.0)))
	     (loop for i from 0 below k
		do (loop for j from 0 below l
		      do (incf (aref result j) (aref m i j))))
	     result))
	  (t (error "invalid axis")))))

(defgeneric m-v-prod (m v &key))

(defmethod m-v-prod ((m simple-array) (v simple-array) &key)
  (assert (= (length (array-dimensions m)) 2))
  (assert (= (length (array-dimensions v)) 1))
  (let* ((n (first (shape m)))
	 (result (make-array n :element-type 'float :initial-element 0.0)))
    (loop for i from 0 below n
          do (setf (aref result i) (dot (m-row m i) v)))
    result))

(defgeneric m-m-prod (m1 m2 &key))

(defmethod m-m-prod ((m1 simple-array) (m2 simple-array) &key)
  (let* ((m (array-dimension m1 0))
	 (n (array-dimension m1 1))
	 (k (array-dimension m2 0))
	 (l (array-dimension m2 1))
	 (result (make-array (list m l) :initial-element 0.0)))
    (assert (= n k))
    (loop for i from 0 below m
          do (loop for j from 0 below l
                   do (setf (aref result i j)
                            (loop for h from 0 below k
                                  sum (* (aref m1 i h)
                                         (aref m2 h j))))))
    result))

(defun tr (m)
  (let* ((k (array-dimension m 0))
	 (l (array-dimension m 1))
	 (result (make-array (list l k) :initial-element 0.0)))
    (loop for i from 0 below k
       do (loop for j from 0 below l
	     do (setf (aref result j i) (aref m i j))))
    result))

(defgeneric norm (u &key))

(defmethod norm ((u simple-vector) &key)
  (sqrt (loop for i across u
	   sum (* i i))))

(defmethod norm ((m simple-array) &key)
  (sqrt (loop for i from 0 below (array-dimension m 0)
              sum (loop for j from 0 below (array-dimension m 1)
                        sum (expt (aref m i j) 2)))))

(defun dot (u v)
  (loop for i across u
     for j across v
     summing (* i j)))

(defun outer (u v)
  (let* ((result (make-array (list (array-total-size u) (array-total-size v)) :initial-element 0.)))
    (loop for i from 0
       for ui across u
       do (loop for j from 0
	     for vj across v
	     do (setf (aref result i j) (* ui vj))))
    result))

(defun kronecker-prod (m1 m2)
  (let* ((r1 (array-dimension m1 0))
	 (r2 (array-dimension m2 0))
	 (c1 (array-dimension m1 1))
	 (c2 (array-dimension m2 1))
	 (result (make-array (list (* r1 r2) (* c1 c2)) :initial-element 0.)))
    (loop for i1 from 0 below r1
       do (loop for i2 from 0 below r2
	     do (loop for j1 from 0 below c1
		   do (loop for j2 from 0 below c2
			 do (setf (aref result (+ (* i1 r2) i2) (+ (* j1 c2) j2))
				  (* (aref m1 i1 j1) (aref m2 i2 j2)))))))
    result))

(defun gauss-jordan (a b)
  (let* (icol irow dum pivinv
	 (n (array-dimension a 0))
	 (m (array-dimension b 1))
	 (indxc (make-array n :initial-element 0))
	 (indxr (make-array n :initial-element 0))
	 (ipiv (make-array n :initial-element 0)))
    (loop for i from 0 below n
       for big = 0.0
       do (loop for j from 0 below n
	     do (if (/= (aref ipiv j) 1)
		    (loop for k from 0 below n
		       do (if (= (aref ipiv k) 0)
			      (if (>= (abs (aref a j k)) big)
				  (progn
				    (setf big (abs (aref a j k)))
				    (setf irow j)
				    (setf icol k)))))))
       do (incf (aref ipiv icol))
       do (if (/= irow icol)
	      (progn
		(loop for l from 0 below n
		   do (rotatef (aref a irow l) (aref a icol l)))
		(loop for l from 0 below m
		   do (rotatef (aref b irow l) (aref b icol l))  )))
       do (setf (aref indxr i) irow)
       do (setf (aref indxc i) icol)
       do (if (= (aref a icol icol) 0.0)
	      (error "singular matrix"))
       do (setf pivinv (/ 1.0 (aref a icol icol)))
       do (setf (aref a icol icol) 1.0)
       do (loop for l from 0 below n
	     do (setf (aref a icol l)
		      (* (aref a icol l) pivinv)))
       do (loop for l from 0 below m
	     do (setf (aref b icol l)
		      (* (aref b icol l)
			 pivinv)))
       do (loop for ll from 0 below n
	     do (if (/= icol ll)
		    (progn
		      (setf dum (aref a ll icol))
		      (setf (aref a ll icol) 0.0)
		      (loop for l from 0 below n
			 do (setf (aref a ll l)
				  (- (aref a ll l) (* (aref a icol l) dum))))
		      (loop for l from 0 below m
			 do (setf (aref b ll l)
				  (- (aref b ll l) (* (aref b icol l) dum)))))))
       do (loop for l from (1- n) downto 0
	     do (if (/= (aref indxr l) (aref indxc l))
		    (loop for k from 0 below n
		       do (rotatef (aref a k (aref indxr l))
				   (aref a k (aref indxc l)))))))))

(defun inverse (m)
  (let ((a (copy-array m)))
    (gauss-jordan a #2A())
    a))

(defun log-sum-exp (m)
  (let* ((k (array-dimension m 0))
	 (l (array-dimension m 1))
	 (result (make-array k :initial-element 0.0)))
    (loop for i from 0 below k
       for row = (loop for j from 0 below l collect (aref m i j))
       for v-row = (make-array l :initial-contents row)
       for max = (apply #'max row)
       do (setf (aref result i)
		(+ max (log (a-sum (a-exp (a- v-row max)))))))
    result))
