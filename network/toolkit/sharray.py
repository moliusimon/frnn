import numpy as np
from multiprocessing import sharedctypes


class sharray(np.ndarray):
    """ Subclass of ndarray with overridden pickling functions which record dtype, shape
    etc... but defer pickling of the underlying data to the original data source """

    def __new__(cls, ctypesArray, shape, dtype=float, strides=None, offset=0, order=None):
        obj = np.ndarray.__new__(cls, shape, dtype, ctypesArray, offset, strides, order)
        obj.ctypesArray = ctypesArray
        return obj

    def __array_finalize__(self, obj):
        self.ctypesArray = None if obj is None else getattr(obj, 'ctypesArray', None)

    def __reduce_ex__(self, protocol):
        """ Delegate pickling of the data to the underlying storage, but keep copies of shape, dtype & strides """
        return sharray, (self.ctypesArray, self.shape, self.dtype, self.strides)

    def __reduce__(self):
        return self.__reduce_ex__(0)


def create(shape, dtype='d'):
    """ Create an uninitialised shared array """
    # Define shape, data type
    shape = np.atleast_1d(shape).astype('i')
    dtype = np.dtype(dtype)

    # Create and return shared memory array
    st = dtype.char in sharedctypes.typecode_to_type.keys()
    return sharray(sharedctypes.RawArray(
        dtype.char if st else 'b',
        np.prod(shape) * (1 if st else dtype.itemsize)
    ), shape, dtype)


def zeros(shape, dtype='d'):
    """ Create an shared array initialised to zeros """
    sa = create(shape, dtype='d')
    sa[:] = np.zeros(1, dtype)
    return sa


def ones(shape, dtype='d'):
    """ Create an shared array initialised to ones """
    sa = create(shape, dtype='d')
    sa[:] = np.ones(1, dtype)
    return sa


def create_copy(a):
    """ Create a a shared copy of an array """
    b = create(a.shape, a.dtype)
    b[:] = a[:]
    return b
