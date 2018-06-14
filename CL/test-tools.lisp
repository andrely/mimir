(in-package :mimir)

(defun array-almost-= (a1 a2 &key (epsilon .001))
  (loop for i across (make-array (array-total-size a1) :displaced-to a1)
     for j across (make-array (array-total-size a2) :displaced-to a2)
     when (> (abs (- i j)) epsilon)
     do (return-from array-almost-= nil))
  t)

(defun almost-= (s1 s2 &key (epsilon .001))
  (< (abs (- s1 s2)) epsilon))
