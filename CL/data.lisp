(in-package :cl-user)

(defpackage :mimir/data
  (:use :common-lisp :mimir :mimir/stat :mimir/sparse)
  (:export :standard-scaler :make-poly :add-bias :binarize :factorize
           :read-svmlight
	   :mlclass-ex4 :mlclass-ex5 :iris :newsgroups))

(in-package :mimir/data)

(defclass standard-scaler ()
  ((center :initform nil)
   (scale :initform nil)))

(defmethod train ((model standard-scaler) (x simple-array) y &key)
  (setf (slot-value model 'center) (mean x))
  (setf (slot-value model 'scale) (sd x))
  model)

(defmethod predict ((model standard-scaler) (x simple-array) &key)
  (let* ((n (array-dimension x 0))
	 (p (array-dimension x 1))
	 (result (make-array (array-dimensions x) :initial-element 0.0d0)))
    (with-slots (center scale) model
      (loop for i from 0 below n
	 do (loop for j from 0 below p
	       do (setf (aref result i j) (/ (- (aref x i j) (aref center j))
					     (aref scale j))))))
    result))

(defun make-poly (x &key (n 6))
  (let* ((p (+ (/ (* (+ n 1) n) 2) n 1))
	 (m (array-dimension x 0))
	 (result (make-array (list m p) :initial-element 0.0)))
    (loop for k from 0 below m
       do (loop for i from 0 to n
	     with idx = 0
	     do (loop for j from 0 to n
		   if (<= (+ i j) n)
		   do (setf (aref result k idx)
			    (* (expt (aref x k 0) i)
			       (expt (aref x k 1) j)))
		   and do (incf idx))))
    result))

(defun add-bias (x)
  (let* ((arr-type (array-element-type x))
         (n (array-dimension x 0))
	 (p (array-dimension x 1))
         (result (make-array (list n (1+ p)) :initial-element (coerce 0 arr-type) :element-type arr-type)))
    (loop for i from 0 below n
       do (loop for j from 0 below (1+ p)
	     if (= j 0)
             do (setf (aref result i j) (coerce 1 arr-type))
	     else do (setf (aref result i j) (aref x i (1- j)))))
    result))

(defun binarize (x)
  (let* ((predicate (if (stringp (elt x 0)) #'string-lessp #'<))
	 (classes (sort (remove-duplicates x :test #'equal) predicate))
	 (num-classes (length classes))
	 (n (array-dimension x 0))
         (result (make-array (list n num-classes) :initial-element 0.0d0)))
    (loop for item across x
       for i from 0
       do (setf (aref result i (position item classes :test #'equal)) 1.0))
    result))

(defun factorize (x)
  (make-array (array-dimension x 0)
	      :initial-contents (loop for i from 0 below (array-dimension x 0)
				   for row = (loop for j from 0 below (array-dimension x 1)
						collect (aref x i j))
				   collect (position (apply #'max row) row))))

(defun read-svmlight (in)
  (let (lbls vals (max-col-idx 0))
    (loop for line = (read-line in nil)
          while line
          do (let* ((tokens (split-sequence:split-sequence #\space line))
                    (label (parse-integer (first tokens)))
                    (elts (loop for token in (rest tokens)
                                for (index val) = (split-sequence:split-sequence #\: token)
                                do (setf index (parse-integer index))
                                do (if (> index max-col-idx)
                                     (setf max-col-idx index))
                                collect (cons index
                                              (let ((*read-eval* nil))
                                                (read-from-string val))))))
               (push label lbls)
               (push elts vals)))
    (setf vals (nreverse vals))
    (setf lbls (nreverse lbls))
    (list (make-lil-matrix (list (length vals) (1+ max-col-idx)) vals)
          (make-array (length lbls) :element-type 'fixnum :initial-contents lbls))))

(defun mlclass-ex4 ()
  (let ((x (make-array '(80 2)
                       :initial-contents '((55.5 69.5) (41.0 81.5) (53.5 86.0) (46.0 84.0) (41.0 73.5)
                                           (51.5 69.0) (51.0 62.5) (42.0 75.0) (53.5 83.0) (57.5 71.0)
                                           (42.5 72.5) (41.0 80.0) (46.0 82.0) (46.0 60.5) (49.5 76.0)
                                           (41.0 76.0) (48.5 72.5) (51.5 82.5) (44.5 70.5) (44.0 66.0)
                                           (33.0 76.5) (33.5 78.5) (31.5 72.0) (33.0 81.5) (42.0 59.5)
                                           (30.0 64.0) (61.0 45.0) (49.0 79.0) (26.5 64.5) (34.0 71.5)
                                           (42.0 83.5) (29.5 74.5) (39.5 70.0) (51.5 66.0) (41.5 71.5)
                                           (42.5 79.5) (35.0 59.5) (38.5 73.5) (32.0 81.5) (46.0 60.5)
                                           (36.5 53.0) (36.5 53.5) (24.0 60.5) (19.0 57.5) (34.5 60.0)
                                           (37.5 64.5) (35.5 51.0) (37.0 50.5) (21.5 42.0) (35.5 58.5)
                                           (26.5 68.5) (26.5 55.5) (18.5 67.0) (40.0 67.0) (32.5 71.5)
                                           (39.0 71.5) (43.0 55.5) (22.0 54.0) (36.0 62.5) (31.0 55.5)
                                           (38.5 76.0) (40.0 75.0) (37.5 63.0) (24.5 58.0) (30.0 67.0)
                                           (33.0 56.0) (56.5 61.0) (41.0 57.0) (49.5 63.0) (34.5 72.5)
                                           (32.5 69.0) (36.0 73.0) (27.0 53.5) (41.0 63.5) (29.5 52.5)
                                           (20.0 65.5) (38.0 65.0) (18.5 74.5) (16.0 72.5) (33.5 68.0))
                       :element-type 'single-float))
	(y #(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	     1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
	     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)))
    (list :x x :y y)))

(defun mlclass-ex5 ()
  (let ((x #2A((0.051267 0.69956) (-0.092742 0.68494) (-0.21371 0.69225) (-0.375 0.50219) (-0.51325 0.46564)
	       (-0.52477 0.2098) (-0.39804 0.034357) (-0.30588 -0.19225) (0.016705 -0.40424) (0.13191 -0.51389)
	       (0.38537 -0.56506) (0.52938 -0.5212) (0.63882 -0.24342) (0.73675 -0.18494) (0.54666 0.48757)
	       (0.322 0.5826) (0.16647 0.53874) (-0.046659 0.81652) (-0.17339 0.69956) (-0.47869 0.63377)
	       (-0.60541 0.59722) (-0.62846 0.33406) (-0.59389 0.005117) (-0.42108 -0.27266) (-0.11578 -0.39693)
	       (0.20104 -0.60161) (0.46601 -0.53582) (0.67339 -0.53582) (-0.13882 0.54605) (-0.29435 0.77997)
	       (-0.26555 0.96272) (-0.16187 0.8019) (-0.17339 0.64839) (-0.28283 0.47295) (-0.36348 0.31213)
	       (-0.30012 0.027047) (-0.23675 -0.21418) (-0.06394 -0.18494) (0.062788 -0.16301) (0.22984 -0.41155)
	       (0.2932 -0.2288) (0.48329 -0.18494) (0.64459 -0.14108) (0.46025 0.012427) (0.6273 0.15863)
	       (0.57546 0.26827) (0.72523 0.44371) (0.22408 0.52412) (0.44297 0.67032) (0.322 0.69225)
	       (0.13767 0.57529) (-0.0063364 0.39985) (-0.092742 0.55336) (-0.20795 0.35599) (-0.20795 0.17325)
	       (-0.43836 0.21711) (-0.21947 -0.016813) (-0.13882 -0.27266) (0.18376 0.93348) (0.22408 0.77997)
	       (0.29896 0.61915) (0.50634 0.75804) (0.61578 0.7288) (0.60426 0.59722) (0.76555 0.50219)
	       (0.92684 0.3633) (0.82316 0.27558) (0.96141 0.085526) (0.93836 0.012427) (0.86348 -0.082602)
	       (0.89804 -0.20687) (0.85196 -0.36769) (0.82892 -0.5212) (0.79435 -0.55775) (0.59274 -0.7405)
	       (0.51786 -0.5943) (0.46601 -0.41886) (0.35081 -0.57968) (0.28744 -0.76974) (0.085829 -0.75512)
	       (0.14919 -0.57968) (-0.13306 -0.4481) (-0.40956 -0.41155) (-0.39228 -0.25804) (-0.74366 -0.25804)
	       (-0.69758 0.041667) (-0.75518 0.2902) (-0.69758 0.68494) (-0.4038 0.70687) (-0.38076 0.91886)
	       (-0.50749 0.90424) (-0.54781 0.70687) (0.10311 0.77997) (0.057028 0.91886) (-0.10426 0.99196)
	       (-0.081221 1.1089) (0.28744 1.087) (0.39689 0.82383) (0.63882 0.88962) (0.82316 0.66301)
	       (0.67339 0.64108) (1.0709 0.10015) (-0.046659 -0.57968) (-0.23675 -0.63816) (-0.15035 -0.36769)
	       (-0.49021 -0.3019) (-0.46717 -0.13377) (-0.28859 -0.060673) (-0.61118 -0.067982) (-0.66302 -0.21418)
	       (-0.59965 -0.41886) (-0.72638 -0.082602) (-0.83007 0.31213) (-0.72062 0.53874) (-0.59389 0.49488)
	       (-0.48445 0.99927) (-0.0063364 0.99927)))
	(y #(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	     1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	     1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
	     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
	     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
	     0.0 0)))
    (list :x x :y y)))

(defun iris ()
  (let* ((x #2A((5.1 3.5 1.4 0.2) (4.9 3. 1.4 0.2) (4.7 3.2 1.3 0.2) (4.6 3.1 1.5 0.2) (5. 3.6 1.4 0.2)
		(5.4 3.9 1.7 0.4) (4.6 3.4 1.4 0.3) (5. 3.4 1.5 0.2) (4.4 2.9 1.4 0.2) (4.9 3.1 1.5 0.1)
		(5.4 3.7 1.5 0.2) (4.8 3.4 1.6 0.2) (4.8 3. 1.4 0.1) (4.3 3. 1.1 0.1) (5.8 4. 1.2 0.2)
		(5.7 4.4 1.5 0.4) (5.4 3.9 1.3 0.4) (5.1 3.5 1.4 0.3) (5.7 3.8 1.7 0.3) (5.1 3.8 1.5 0.3)
		(5.4 3.4 1.7 0.2) (5.1 3.7 1.5 0.4) (4.6 3.6 1. 0.2) (5.1 3.3 1.7 0.5) (4.8 3.4 1.9 0.2)
		(5. 3. 1.6 0.2) (5. 3.4 1.6 0.4) (5.2 3.5 1.5 0.2) (5.2 3.4 1.4 0.2) (4.7 3.2 1.6 0.2)
		(4.8 3.1 1.6 0.2) (5.4 3.4 1.5 0.4) (5.2 4.1 1.5 0.1) (5.5 4.2 1.4 0.2) (4.9 3.1 1.5 0.2)
		(5. 3.2 1.2 0.2) (5.5 3.5 1.3 0.2) (4.9 3.6 1.4 0.1) (4.4 3. 1.3 0.2) (5.1 3.4 1.5 0.2)
		(5. 3.5 1.3 0.3) (4.5 2.3 1.3 0.3) (4.4 3.2 1.3 0.2) (5. 3.5 1.6 0.6) (5.1 3.8 1.9 0.4)
		(4.8 3. 1.4 0.3) (5.1 3.8 1.6 0.2) (4.6 3.2 1.4 0.2) (5.3 3.7 1.5 0.2) (5. 3.3 1.4 0.2)
		(7. 3.2 4.7 1.4) (6.4 3.2 4.5 1.5) (6.9 3.1 4.9 1.5) (5.5 2.3 4. 1.3) (6.5 2.8 4.6 1.5)
		(5.7 2.8 4.5 1.3) (6.3 3.3 4.7 1.6) (4.9 2.4 3.3 1.) (6.6 2.9 4.6 1.3) (5.2 2.7 3.9 1.4)
		(5. 2. 3.5 1.) (5.9 3. 4.2 1.5) (6. 2.2 4. 1.) (6.1 2.9 4.7 1.4) (5.6 2.9 3.6 1.3)
		(6.7 3.1 4.4 1.4) (5.6 3. 4.5 1.5) (5.8 2.7 4.1 1.) (6.2 2.2 4.5 1.5) (5.6 2.5 3.9 1.1)
		(5.9 3.2 4.8 1.8) (6.1 2.8 4. 1.3) (6.3 2.5 4.9 1.5) (6.1 2.8 4.7 1.2) (6.4 2.9 4.3 1.3)
		(6.6 3. 4.4 1.4) (6.8 2.8 4.8 1.4) (6.7 3. 5. 1.7) (6. 2.9 4.5 1.5) (5.7 2.6 3.5 1.)
		(5.5 2.4 3.8 1.1) (5.5 2.4 3.7 1.) (5.8 2.7 3.9 1.2) (6. 2.7 5.1 1.6) (5.4 3. 4.5 1.5)
		(6. 3.4 4.5 1.6) (6.7 3.1 4.7 1.5) (6.3 2.3 4.4 1.3) (5.6 3. 4.1 1.3) (5.5 2.5 4. 1.3)
		(5.5 2.6 4.4 1.2) (6.1 3. 4.6 1.4) (5.8 2.6 4. 1.2) (5. 2.3 3.3 1.) (5.6 2.7 4.2 1.3)
		(5.7 3. 4.2 1.2) (5.7 2.9 4.2 1.3) (6.2 2.9 4.3 1.3) (5.1 2.5 3. 1.1) (5.7 2.8 4.1 1.3)
		(6.3 3.3 6. 2.5) (5.8 2.7 5.1 1.9) (7.1 3. 5.9 2.1) (6.3 2.9 5.6 1.8) (6.5 3. 5.8 2.2)
		(7.6 3. 6.6 2.1) (4.9 2.5 4.5 1.7) (7.3 2.9 6.3 1.8) (6.7 2.5 5.8 1.8) (7.2 3.6 6.1 2.5)
		(6.5 3.2 5.1 2.) (6.4 2.7 5.3 1.9) (6.8 3. 5.5 2.1) (5.7 2.5 5. 2.) (5.8 2.8 5.1 2.4)
		(6.4 3.2 5.3 2.3) (6.5 3. 5.5 1.8) (7.7 3.8 6.7 2.2) (7.7 2.6 6.9 2.3) (6. 2.2 5. 1.5)
		(6.9 3.2 5.7 2.3) (5.6 2.8 4.9 2.) (7.7 2.8 6.7 2.) (6.3 2.7 4.9 1.8) (6.7 3.3 5.7 2.1)
		(7.2 3.2 6. 1.8) (6.2 2.8 4.8 1.8) (6.1 3. 4.9 1.8) (6.4 2.8 5.6 2.1) (7.2 3. 5.8 1.6)
		(7.4 2.8 6.1 1.9) (7.9 3.8 6.4 2.) (6.4 2.8 5.6 2.2) (6.3 2.8 5.1 1.5) (6.1 2.6 5.6 1.4)
		(7.7 3. 6.1 2.3) (6.3 3.4 5.6 2.4) (6.4 3.1 5.5 1.8) (6. 3. 4.8 1.8) (6.9 3.1 5.4 2.1)
		(6.7 3.1 5.6 2.4) (6.9 3.1 5.1 2.3) (5.8 2.7 5.1 1.9) (6.8 3.2 5.9 2.3) (6.7 3.3 5.7 2.5)
		(6.7 3. 5.2 2.3) (6.3 2.5 5. 1.9) (6.5 3. 5.2 2.) (6.2 3.4 5.4 2.3) (5.9 3. 5.1 1.8)))
	 (y #("setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
	      "setosa" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
	      "versicolor" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica" "virginica" "virginica" "virginica" "virginica"
	      "virginica")))
    (list :x x :y y)))

(defun filename-from-path (p)
  (multiple-value-bind (type directories file flag)
                                   (uiop/pathname:split-unix-namestring-directory-components (uiop/pathname:unix-namestring p))
                                 (declare (ignore type file flag))
                                 (first (last directories))))

(defun file-content (p)
  (with-open-file (s p)
    (loop for line = (read-line s nil nil)
          while line
          collect line)))

(defun clean-article (lines)
  (remove-if #'(lambda (line)
                 (cl-ppcre:scan "(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\\||^>)" line))
             (subseq lines
                     (1+ (position "" lines :test #'equal))
                     (position-if #'(lambda (line)
                                      (string-equal "--" (string-trim '(#\Space #\Tab) line)))
                                  lines :from-end t))))

(defun newsgroups (path)
  (let* ((groups (loop for p in (directory (uiop/pathname:merge-pathnames* path uiop/pathname:*wild-directory*))
                       collect (filename-from-path p)))
         x y)
    (loop for g in groups
          for gpath = (uiop/pathname:merge-pathnames* (uiop/pathname:make-pathname* :directory (list :relative g)) path)
          do (loop for f in (directory (merge-pathnames gpath uiop/pathname:*wild-directory*))
                   do (push (clean-article (file-content f)) x)
                   do (push g y)))
    (list :x x :y y)))
