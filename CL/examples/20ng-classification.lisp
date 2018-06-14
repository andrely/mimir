(ql:quickload :mimir)

(defparameter *20ng-path* #p"/Users/stinky/Documents/Work/mimir/data/20newsgroups/mini_newsgroups/")

(defparameter *newsgroups* (mimir/data:newsgroups #p"/Users/stinky/Documents/Work/mimir/data/20newsgroups/mini_newsgroups/"))

(defparameter *vectorizer* (mimir/text:make-vectorizer :max-features 1000))

(train *vectorizer* (getf *newsgroups* :x) nil)

(defparameter *m* (predict *vectorizer* (getf *newsgroups* :x)))


