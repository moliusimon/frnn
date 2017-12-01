from operator import Operator
import numpy as np


class OperatorCrop(Operator):
    def __init__(self, sizes, amounts=None):
        """
        Crop data across all dimensions, randomly during random pre-processing, and at N evenly spread locations per
        dimension during full pre-processing.
        :param sizes: Size of the crop at each dimension. Use -1 to specify full size. (list)
        :param amounts: Number of evenly-spread crops per dimension during full pre-processing. (list)
        """

        # Prepare amount of crops and size at each dimension
        amounts = [1] * len(sizes) if amounts is None else amounts
        self.amounts = amounts if isinstance(amounts, (list, tuple)) else [amounts] * len(self.sizes)
        self.sizes = sizes

        # Call parent initializer
        Operator.__init__(self, num_modes=np.prod(amounts))

    def apply_random(self, streams):
        n_inst, shape = streams[0].shape[0], streams[0].shape[1:]
        sizes = np.array([(sha if exp == -1 else exp) for sha, exp in zip(shape, self.sizes)])
        offsets = np.cast[np.int32](np.random.random(size=(n_inst, len(shape))) * np.expand_dims(shape - sizes, axis=0))
        slc = [[slice(off, off + sz) for i, (off, sz) in enumerate(zip(offs, sizes))] for offs in offsets]
        return tuple(np.stack([ss[sl] for ss, sl in zip(s, slc)], axis=0) for s in streams)

    def apply_full(self, streams):
        pass  # TODO
        return self.apply_random(streams)
