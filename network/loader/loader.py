import numpy as np
import os
from random import shuffle
import copy


class Loader:
    def __init__(self, num_samples, shuffle_data=False):
        # Initialize counter/number of samples
        self.counter = 0
        self.num_samples = num_samples
        self.sample_indices = range(self.num_samples)
        self._buffered_samples = None

        # Shuffle samples if requested
        if shuffle_data:
            shuffle(self.sample_indices)

    def has_batch(self):
        n_buffered = 0 if self._buffered_samples is None else len(self._buffered_samples[0])
        return (n_buffered > 0) or (self.counter < self.num_samples-1)

    def instantiate(self, sample_indices=None):
        ret = copy.copy(self)
        if sample_indices is not None:
            ret.counter = 0
            ret.num_samples = len(sample_indices)
            ret.sample_indices = [self.sample_indices[i] for i in sample_indices]
            ret._buffered_samples = None

        return ret

    def reset(self):
        self.counter = 0
        if hasattr(self, '_reset'):
            getattr(self, '_reset')()

    def sample(self, mode, batch_size):
        self._buffer_samples(mode, batch_size)
        return self._sample_buffer(batch_size)

    def retrieve(self, indices=None):
        # Prepare indices to retrieve
        indices = range(self.num_samples) if indices is None else indices
        indices = indices if isinstance(indices, (list, tuple)) else [indices]
        indices = [self.sample_indices[i] for i in indices]

        # Obtain samples from indices
        sampled = (getattr(self, '_sample_test') if hasattr(self, '_sample_test') else self._sample)(indices)
        return (sampled,) if not isinstance(sampled, tuple) else sampled

    def _sample(self, indices):
        raise NotImplementedError('Sample function not implemented for this loader!')

    def _buffer_samples(self, mode, batch_size):
        n_buffered = 0 if self._buffered_samples is None else len(self._buffered_samples[0])
        to_sample = max(batch_size - n_buffered, 0)
        if to_sample == 0:
            return

        # Prepare indices of samples to use
        i1 = range(self.counter, min(self.num_samples, self.counter+to_sample))
        i2 = range(0, to_sample-len(i1)) if mode is 'train' else []
        indices = [self.sample_indices[i] for i in (i1 + i2)]

        # If no indices (out of data to sample) return
        if len(indices) == 0:
            return

        # Increase counter
        self.counter += len(indices)
        self.counter = self.counter % self.num_samples if mode is 'train' else self.counter

        # Sample missing elements
        sampled = (
            getattr(self, '_sample_' + mode) if hasattr(self, '_sample_' + mode) else self._sample
        )(indices)
        sampled = (sampled,) if not isinstance(sampled, tuple) else sampled

        # If nothing buffered, set new samples as buffer
        if self._buffered_samples is None:
            self._buffered_samples = sampled

        # If buffered elements, extend with new samples
        else:
            self._buffered_samples = tuple(
                np.concatenate((b, s), axis=0) for b, s in zip(self._buffered_samples, sampled)
            )

    def _sample_buffer(self, batch_size):
        buffer_size = len(self._buffered_samples[0])
        num_samples = min(buffer_size, batch_size)

        # If number of samples equals buffer size, return buffer
        if num_samples == buffer_size:
            ret = self._buffered_samples
            self._buffered_samples = None
            return ret

        # Extract samples from buffer, return extracted samples
        ret, self._buffered_samples = zip(*[(b[:num_samples], b[num_samples:]) for b in self._buffered_samples])
        return ret
