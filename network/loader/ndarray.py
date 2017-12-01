from loader import Loader


class LoaderNdarray(Loader):
    def __init__(self, x):
        self.data = x
        Loader.__init__(self, len(self.data))

    def _sample(self, indices):
        return self.data[indices, ...]
