from loader import Loader


def build_loader(x):
    from multistream import LoaderMultistream
    from ndarray import LoaderNdarray
    from hdf5 import LoaderHdf5
    import numpy as np
    import cPickle
    import os

    # If already a loader, return as-is
    if isinstance(x, Loader):
        return x

    # If input specified as an ndarray
    if isinstance(x, np.ndarray):
        return LoaderNdarray(x)

    # If input specified as a tuple
    if isinstance(x, tuple):
        return LoaderMultistream(tuple(build_loader(s) for s in x))

    # If input specified as a string (file path)
    if isinstance(x, basestring):
        # Check file type and existence
        file_type = x.split('.')[-1].lower()
        if not os.path.exists(x):
            raise IOError('Specified file does not exist!')

        # If input specified as an HDF5 file
        if file_type in ['hdf5', 'h5']:
            return LoaderHdf5(x)

        # If input specified as a pkl file
        if file_type in ['pkl', 'pickle']:
            return build_loader(cPickle.load(open(x, 'rb')))

        # If input specified as a numpy file
        if file_type in ['npy', 'numpy']:
            return build_loader(np.load(x))
