from operator import Operator
import numpy as np


class OperatorMirror(Operator):
    def __init__(self, axes):
        """
        Mirror data across specified axes, randomly during pre-processing, and using all permutations during full
        pre-processing.
        :param axes: Dimensions across which data is mirrored. (list)
        """

        # Check first dimension (batch size) is not flipped
        if 0 in axes:
            raise ValueError('Cannot flip across dimension 0 (batch size)!')

        # Prepare amount of crops and size at each dimension
        self.axes = axes

        # Call parent initializer
        Operator.__init__(self, num_modes=2**len(self.axes))

    def apply_random(self, streams):
        flips = np.random.randint(0, 2, size=(len(self.axes), streams[0].shape[0]), dtype=np.bool)
        to_flip = [np.where(d)[0] for d in flips]

        ret = tuple(streams)
        for stream in streams:
            for i, inds in zip(self.axes, to_flip):
                stream[inds] = np.flip(stream[inds], axis=i)

        return ret

    def apply_full(self, streams):
        pass  # TODO
