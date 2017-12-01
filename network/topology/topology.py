from ..layer import layer_types
import tensorflow as tf


class Topology:
    def __init__(self, specifications):
        self.specifications = specifications
        self.topology = []
        self.scope = None

    def build(self, scope):
        # Create the layers used on the topology
        self.topology, _ = self._create_layers(self.specifications)
        self.scope = scope

        # Call topology-specific builder if it exists
        if hasattr(self, '_build'):
            getattr(self, '_build')()

    def forward(self, x):
        for l in self._flatten_topology():
            x = l.forward(x)
        return x

    def backward(self, x):
        for l in self._flatten_topology()[::-1]:
            x = l.backward(x)
        return x

    def is_recurrent(self):
        return len(self.gather()) > 0

    def gather(self, placeholder=False):
        ret = []

        # Gather from base topology if function defined
        if hasattr(self, '_gather'):
            t_ret = getattr(self, '_gather')(placeholder)
            ret = list(t_ret) if isinstance(t_ret, (list, tuple)) else [t_ret]

        # Gather from model topology
        return ret + [v for l in self._flatten_topology() for v in l.gather(placeholder)]

    def scatter(self, values):
        # Scatter to base model if function defined
        if hasattr(self, '_scatter'):
            getattr(self, '_scatter')(values)

        # Scatter to topology layers
        for l in self._flatten_topology():
            l.scatter(values)

    def get_input_shape(self):
        return self.topology[0].get_input_shape()

    def get_output_shape(self):
        return self.topology[-1].get_output_shape()

    def get_model_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def _create_layers(self, layers, i_l=0):
        ret = []

        for l in layers:
            if isinstance(l, dict):
                with tf.variable_scope('layer_' + str(i_l)):
                    ret.append(layer_types[l['type']](
                        **{k: v for k, v in l.iteritems() if k is not 'type'}
                    ))

            elif isinstance(l, list):
                t_l, i_l = self._create_layers(l, i_l=i_l)
                ret.append(t_l)

            else:
                raise ValueError('Unknown element is present in the list of layers!')

            i_l += 1

        return ret, i_l

    def _flatten_topology(self, topology=None):
        topology = self.topology if topology is None else topology
        layers = []

        for l in topology:
            if isinstance(l, list):
                layers.extend(self._flatten_topology(l))

            else:
                layers.append(l)

        return layers
