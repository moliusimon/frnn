# from operator import Operator
# from .. import Preprocessor
# from numpy import np
#
#
# class OperatorReroute(Operator):
#     def __init__(self, groups, preprocessing_streams):
#         """
#         Reroute the streams into different preprocessing streams and merge the output streams.
#         :param groups: Indices of the streams fed to each preprocessing stream (tuple of lists)
#         :param preprocessing_streams: Operators for each preprocessing stream (tuple of lists)
#         """
#
#         # Check there are as many groups as operator streams
#         if len(groups) != len(preprocessing_streams):
#             raise ValueError('The reroute operator must be given as many groups as operator streams!')
#
#         # Build operator streams
#         self.groups = groups
#         self.preprocessors = [Preprocessor(stream) for stream in preprocessing_streams]
#         self.ordering = np.argsort(np.concatenate(tuple(self.groups)))
#
#         # Call parent constructor
#         Operator.__init__(self, num_modes=1)
#
#     def apply_random(self, streams):
#         # Apply preprocessing streams
#         unordered_streams = sum([preprocessor.preprocess(
#             tuple(streams[i] for i in indices),
#             full_augment=False
#         ) for indices, preprocessor in zip(self.groups, self.preprocessors)])
#
#         # Reorder output streams and return
#         return tuple(unordered_streams[i] for i in self.ordering)
#
#     def apply_full(self, streams):
#         # Apply preprocessing streams
#         unordered_streams = sum([preprocessor.preprocess(
#             tuple(streams[i] for i in indices),
#             full_augment=True
#         ) for indices, preprocessor in zip(self.groups, self.preprocessors)])
#
#         # Reorder output streams and return
#         return tuple(unordered_streams[i] for i in self.ordering)
