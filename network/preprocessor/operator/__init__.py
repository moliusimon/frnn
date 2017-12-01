from operator import Operator


def build_operator(specs):
    from op_crop import OperatorCrop
    # from op_reroute import OperatorReroute
    from op_swapaxes import OperatorSwapaxes
    from op_rescale import OperatorRescale
    from op_mirror import OperatorMirror
    from op_shift import OperatorShift

    # Get operator type, raise error if not specified
    type = specs.get('type', None)
    if type is None:
        raise ValueError('You must specify the type of a sampling operator!')

    # Prepare operator name and parameters
    name = 'Operator' + type.capitalize()
    specs = dict(specs)
    del specs['type']

    # Build operator
    return locals()[name](**specs)
