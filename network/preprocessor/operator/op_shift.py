from operator import Operator
from scipy.ndimage.interpolation import shift
import numpy as np


class OperatorShift(Operator):
    def __init__(self, axes, ranges):
        """
        Shift data across specified axes, randomly during pre-processing, doing nothing during full-augmentation.
        :param axes: Dimensions across which to shift. (list)
        :param ranges: Range of the shifts across each dimension. (list of lists)
        """

        # Check first dimension (batch size) is not shifted
        if 0 in axes:
            raise ValueError('Cannot shift across dimension 0 (batch size)!')

        # Prepare operator attributes
        self.axes = axes
        self.ranges = ranges

        # Call parent initializer
        Operator.__init__(self, num_modes=1)

    def apply_random(self, streams):
        # Prepare shift sampling ranges for all dimensions (even those not considered)
        ranges = [(0, 1)] * (len(streams[0].shape) - 1)
        for i, ax in enumerate(self.axes):
            ranges[ax-1] = self.ranges[i]

        # Sample shift values
        disps = np.concatenate([np.random.randint(
            rng_lb, rng_ub,
            size=(streams[0].shape[0],)
        ) for (rng_lb, rng_ub) in ranges], axis=0)

        # Perform shifts
        ret = tuple(streams)
        for stream in ret:
            for i, (inst, disp) in enumerate(zip(stream, disps)):
                stream[i] = shift(inst, disp)

        return ret

    def apply_full(self, streams):
        return streams
