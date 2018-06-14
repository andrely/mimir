(in-package :cl-user)

(defpackage :mimir-test
  (:use :common-lisp :mimir :mimir/logistic :mimir/data :mimir/linalg :fiveam))

(in-package :mimir-test)

(def-suite :mimir-all)
