from convolutional import LayerConvolutional
from fully import LayerFully
from pooling import LayerPooling
from reshape import LayerReshape
from gru import LayerGru
from bconvgru import LayerBConvgru
from lconvgru import LayerLConvgru
from marginalize import LayerMarginalize

layer_types = {
    'convolutional': LayerConvolutional,
    'fully': LayerFully,
    'pooling': LayerPooling,
    'reshape': LayerReshape,
    'gru': LayerGru,
    'bconvgru': LayerBConvgru,
    'lconvgru': LayerLConvgru,
    'marginalize': LayerMarginalize,
}
