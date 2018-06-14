(in-package :cl-user)

(defpackage :mimir
  (:use :common-lisp)
  (:export :train :predict :log-prob :replicate))

(in-package :mimir)

(defgeneric train (model x y &key))

(defgeneric predict (model x &key))

(defgeneric log-prob (model x &key))

(defgeneric replicate (model &key))
