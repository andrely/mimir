(in-package :cl-user)

(defpackage :mimir/text
  (:use :common-lisp :mimir :mimir/sparse)
  (:export
   :tokenize
   :vocabulary :make-vocabulary :add-docs :id-to-token :token-to-id :compact
   :vectorizer :make-vectorizer))

(in-package :mimir/text)

(defun tokenize (str)
  (mapcar #'string-downcase (cl-ppcre:all-matches-as-strings "\\w+" str)))

(defclass vocabulary ()
  ((id-to-token :initform (make-array 1000 :adjustable t :fill-pointer 0))
   (counts      :initform (make-array 1000 :initial-element 0 :adjustable t :fill-pointer 0))
   (doc-counts  :initform (make-array 1000 :initial-element 0 :adjustable t :fill-pointer 0))
   (token-to-id :initform (make-hash-table :test #'equal))
   (num-docs    :initform 0)))


(defmethod make-vocabulary (&optional (docs nil) &key (capacity 1000))
  (let ((vocab (make-instance 'vocabulary)))
    (add-docs vocab docs)))

(defgeneric add-docs (vocab docs &key))

(defgeneric id-to-token (vocab id &key))

(defgeneric token-to-id (vocab token &key))

(defmethod add-docs ((vocab vocabulary) docs &key)
  (with-slots (id-to-token counts doc-counts token-to-id num-docs) vocab
    (loop for doc in docs
        for seen = nil
        do (loop for token in doc
                 do (let ((idx (gethash token token-to-id)))
                      (if idx
                        (incf (aref counts idx))
                        (let ((new-idx (fill-pointer id-to-token)))
                           (when (>= new-idx (array-dimension id-to-token 0))
                             (adjust-array counts (+ (array-dimension id-to-token 0) 1000))
                             (adjust-array doc-counts (+ (array-dimension id-to-token 0) 1000))
                             (adjust-array id-to-token (+ (array-dimension id-to-token 0) 1000)))
                           (vector-push token id-to-token)
                           (vector-push 1 counts)
                           (vector-push 0 doc-counts)
                           (setf (gethash token token-to-id) new-idx)
                           (setf idx new-idx)))
                      (when (not (member token seen :test #'equal))
                        (push token seen)
                        (incf (aref doc-counts idx)))))
        do (incf num-docs)))
  vocab)

(defmethod id-to-token ((vocab vocabulary) id &key)
  (with-slots (id-to-token) vocab
    (aref id-to-token id)))

(defmethod token-to-id ((vocab vocabulary) token &key)
  (with-slots (token-to-id) vocab
    (gethash token token-to-id)))

(defun filtered-token-id (vocab id &key (min-tf 5) (max-idf 0.8))
  (with-slots (counts doc-counts num-docs) vocab
    (or (< (aref counts id) min-tf)
        (> (/ (aref doc-counts id) num-docs) max-idf))))

(defun compact (vocab &key (min-tf 5) (max-idf 0.8) (max-features nil))
  (with-slots (id-to-token counts doc-counts token-to-id num-docs) vocab
      (let* ((new-length (loop for idx from 0 below (length id-to-token)
                              count (not (filtered-token-id vocab idx :min-tf min-tf :max-idf max-idf))))
             (new-id-to-token (make-array new-length :adjustable t :fill-pointer new-length))
             (new-counts (make-array new-length :initial-element 0 :adjustable t :fill-pointer new-length))
             (new-doc-counts (make-array new-length :initial-element 0 :adjustable t :fill-pointer new-length))
             (new-token-to-id (make-hash-table :test #'equal))
             (cutoff (and max-features (> (length counts) max-features) (aref (sort counts #'>) max-features)))
             (new-id 0))
        (loop for i from 0 below (length counts)
              if (not (filtered-token-id vocab i  :min-tf min-tf :max-idf max-idf))
              do (progn
                   (setf (aref new-id-to-token new-id) (aref id-to-token i))
                   (setf (aref new-counts new-id) (aref counts i))
                   (setf (aref new-doc-counts new-id) (aref doc-counts i))
                   (setf (gethash (aref id-to-token i) new-token-to-id) new-id)
                   (incf new-id)))

        (when cutoff
          (let ((new-id 0))
            (loop for i from 0 below (length new-counts)
                  for c = (aref new-counts i)
                  for token = (aref new-id-to-token i)
                  if (>= c cutoff)
                  do (progn
                       (setf (aref new-id-to-token new-id) (aref new-id-to-token i))
                       (setf (aref new-counts new-id) (aref new-counts i))
                       (setf (aref new-doc-counts new-id) (aref new-doc-counts i))
                       (setf (gethash token new-token-to-id) new-id)
                       (incf new-id))
                  else do (remhash token new-token-to-id)
                  while (< new-id max-features))
            (setf (fill-pointer new-id-to-token) max-features)
            (setf (fill-pointer new-counts) max-features)
            (setf (fill-pointer doc-counts) max-features)))

        (setf (slot-value vocab 'id-to-token) new-id-to-token)
        (setf (slot-value vocab 'counts) new-counts)
        (setf (slot-value vocab 'doc-counts) new-doc-counts)
        (setf (slot-value vocab 'token-to-id) new-token-to-id)))
    
  vocab)

(defun counter (items)
  (when (null items)
    (return-from counter nil))
  (let ((i 0)
        (items (sort items #'string-lessp)))
    (loop while i
          for item = (elt items i)
          for end = (position-if-not #'(lambda (x) (string-equal x item)) items :start i)
          for tokens = (subseq items i (or end (length items)))
          collect (list (first tokens) (length tokens))
          do (setf i end))))

(defclass vectorizer ()
  ((max-features :initform nil :initarg :max-features)
   (vocab :initform (make-vocabulary))))

(defun make-vectorizer (&key (max-features nil))
  (make-instance 'vectorizer :max-features max-features))

(defmethod train ((transformer vectorizer) x y &key)
  (declare (ignore y))
  (with-slots (vocab max-features) transformer
    (loop for doc in x
          do (cond ((listp doc)
                    (add-docs vocab (list (loop for sent in doc
                                                append (tokenize sent)))))
                   ((stringp doc)
                    (list (add-docs vocab (tokenize doc))))
                   (t (error "malformed document"))))
    (compact vocab :max-features max-features)

    transformer))

(defmethod predict ((transformer vectorizer) x &key)
  (with-slots (vocab) transformer
    (mimir/sparse:make-lil-matrix (list (length x) (length (slot-value vocab 'id-to-token)))
                                  (loop for doc in x
                                        collect (let ((tokens (cond ((listp doc)
                                                                     (loop for sent in doc
                                                                           append (tokenize sent)))
                                                                    ((stringp doc) (tokenize doc))
                                                                    (t (error "malformed document")))))
                                                  (loop for (word count) in (counter tokens)
                                                        for token-id = (mimir/text:token-to-id vocab word)
                                                        if token-id
                                                        collect (list token-id count)))))))
