class Encoder(object):
    def __init__(self, *features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def add_features(self, features):
        self.features += features

        return self

    def __call__(self, x, y):
        f = []

        for i, feature in enumerate(self.features):
            val = feature(x, y)

            if val != 0:
                f.append((i, val))

        return f

