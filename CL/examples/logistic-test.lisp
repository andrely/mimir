(ql:quickload :mimir)

(defpackage :script/local (:use :common-lisp :mimir/linalg :mimir/stat))

(in-package :script/local)

(defparameter *iris* (mimir/data:iris))
(defparameter *x* (getf *iris* :x))
(defparameter *y* (m-col (mimir/data:binarize (getf *iris* :y)) 0))

(defparameter *w* (make-array 4 :initial-contents (loop for i from 0 below 4
                                                        collect (* (normal-deviate) .001d0))
                              :element-type 'double-float))
(defparameter *w0* (* (normal-deviate) .001d0))


(defun a (x w w0)
  (let ((p (a+ (m-v-prod x w) w0)))
    (setf p (a- p (log-sum-exp (join-col p (make-array (array-total-size p) :initial-element 0.0d0)))))
    (a-exp p)))


(defun cost (x y w w0)
  (let ((p (a x w w0)))
    (a-sum (a- (a+ (a* y (a-log p))
                   (a* (a- 1.0 y) (a-log (a- 1.0 p))))))))

(defun iteration (x y w w0)
  (let (p g0 g h0 h)
    (setf p (a+ (m-v-prod x w) w0))
    (setf p (a- p (log-sum-exp (join-col p (make-array (array-total-size p) :initial-element 0.0d0)))))
    (setf p (a-exp p))

    (setf g0 (a-sum (mimir/linalg:a- p y)))
    (setf g (m-v-prod (tr x) (a- p y)))
    (setf h0 (a-sum (a* p (a- 1 p))))
    (setf h (m-m-prod (m-m-prod (tr x) (diag (a* p (a- 1 p)))) x))

    (list (- w0 (* .01 (/ g0 h0)))
          (a- w (a* .01 (m-v-prod (inverse h) g))))))

(defparameter *result*
  (loop for i from 0 below 100 with w = *w* with w0 = *w0* do (let ((r (iteration *x* *y* w w0)))
                                                                (setf w0 (first r))
                                                                (setf w (second r)))
        when (= (mod i 10) 0) do (pprint (cost *x* *y* w w0))
        finally (return (list w0 w))))

(format t "~%~a~%" (mimir/metrics:accuracy *y* (loop for i across (a *x* (second *result*) (first *result*)) collect (if (> i 0.5) 1.0 0.0))))

#+CCL
(ccl:quit)
#+sbcl
(sb-ext:quit)
