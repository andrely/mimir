from numpy.ma import array, argmax


class OVRClassifier():
    def __init__(self, base_model):
        self.base_model = base_model
        self.models = None

    def train(self, X, y, verbose=True):
        _, C = y.shape
        self.models = [self.base_model.replicate().train(X, cls_y, verbose=verbose)
                       for cls_y in y.T]

        return self

    def predict(self, X):
        log_probs = array([m.log_prob(X) for m in self.models]).T

        return argmax(log_probs, axis=1)
