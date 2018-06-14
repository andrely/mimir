(in-package :cl-user)

(defpackage :mimir/sparse
  (:use :common-lisp :mimir/linalg)
  (:export :make-lil-matrix))

(in-package :mimir/sparse)

(defclass lil-matrix ()
  ((rows  :initform nil)
   (dimensions :initform nil :accessor dimensions)))

(defmethod print-object ((obj lil-matrix) out)
  (print-unreadable-object (obj out :type t)
    (format out "~s x ~s" (first (slot-value obj 'dimensions)) (second (slot-value obj 'dimensions)))))

(defun make-lil-matrix (shape &optional (rows nil))
  (let* ((inst (make-instance 'lil-matrix))
         (n-rows (first shape))
         (n-cols (second shape)))
    (setf (slot-value inst 'dimensions) (list n-rows n-cols))
    (setf (slot-value inst 'rows) (make-array n-rows))

    (when (and rows (/= (length rows) n-rows))
      (error "Row argument must match shape"))

    (if rows
      (loop for i from 0
            for r in rows
            do (setf (aref (slot-value inst 'rows) i) r))
      (loop for i from 0 below n-rows
            do (setf (aref (slot-value inst 'rows) i) (list))))

    inst))

(defmethod shape ((m lil-matrix) &key)
  (dimensions m))

(defmethod aslice ((m lil-matrix) &rest subscripts)
  (if (= (length subscripts) 1)
    (aslice m (first subscripts) :all)
    (with-slots (rows) m
      (let* ((sub-1 (first subscripts))
             (sub-2 (second subscripts))
             (min-col (if (eql sub-2 :all) 0 (v-min sub-2)))
             (max-col (if (eql sub-2 :all) (second (shape m)) (1+ (v-max sub-2))))
             (selected-rows (cond ((eql sub-1 :all) rows)
                                  (t (make-array (length sub-1)
                                                 :initial-contents (loop for i across sub-1
                                                                         collect (aref rows i)))))))
        
        (make-lil-matrix (list (length selected-rows) (- max-col min-col))
                         (loop for row across selected-rows
                               for i from 0
                                collect (loop for (index . value) in row
                                             when (and (>= index min-col) (< index max-col))
                                             collect (cons (- index min-col) value))))))))

(defmethod a-sum ((m lil-matrix) &key (axis :all))
  (with-slots (rows dimensions) m
    (cond ((eql axis :all)
           (loop for row across rows
                 summing (loop for (index . val) in row
                               summing val)))
          ((eql axis :row)
           (let ((result (make-array (first dimensions) :initial-element 0)))
             (loop for row across rows
                   for i from 0
                   do (setf (aref result i)
                            (loop for (index . val) in row
                                  summing val)))
             result))
          ((eql axis :col)
           (let ((result (make-array (second dimensions) :initial-element 0)))
             (loop for row across rows
                   do (loop for (index . val) in row
                            do (incf (aref result index) val)))
             result))
          (t (error "invalid axis")))))

(defmethod m-v-prod ((m lil-matrix) (v simple-vector) &key)
  (let ((result (make-array (first (shape m)) :initial-element 0.0)))
    (with-slots (rows) m
      (loop for row across rows
            for i from 0
            do (setf (aref result i)
                     (loop for (index . value) in row
                           summing (* value (aref v index))))))
    result))

(defmethod m-m-prod ((m1 lil-matrix) (m2 simple-array) &key)
  (let* ((m (first (shape m1)))
         (n (second (shape m1)))
         (k (first (shape m2)))
         (l (second (shape m2)))
         (result))
    (assert (= n k))
    (setf result (make-array (list m l) :initial-element 0.0))
    (loop for col-idx from 0 below l
          for col = (m-v-prod m1 (m-col m2 col-idx))
          do (loop for val across col
                   for idx from 0
                   do (setf (aref result idx col-idx) val)))
    result))