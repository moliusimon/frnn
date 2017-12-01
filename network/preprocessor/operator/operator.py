class Operator:
    def __init__(self, num_modes):
        self.num_modes = num_modes

    def get_num_modes(self):
        return self.num_modes

    def apply(self, streams, full_augment):
        return (self.apply_full if full_augment else self.apply_random)(streams)

    def apply_random(self, streams):
        raise NotImplementedError('apply_random method not implemented for the sampling operator!')

    def apply_full(self, streams):
        raise NotImplementedError('apply_random method not implemented for the sampling operator!')