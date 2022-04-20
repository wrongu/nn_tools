import torch
import torch.nn as nn
from typing import Dict, Tuple, Callable, Union, Any, Iterable, List
import warnings


# This file is inspired by https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py

# An 'Operation' is essentially a callable object that behaves like a nn.Module. Use None for input values.
OpType = Union[None, Callable, nn.Module, 'ModelType']
# Specify inputs by name (string), relative index (negative integer), or a list of either
InputType = Union[str, int, Iterable[Union[str, int]]]
# A model is a dict of (i) {name: op}, or (ii) {name: (op, input)}, where 'op' can itself be a sub-model
ModelType = Dict[str, Union[OpType, Tuple[OpType, InputType]]]
# A Graph is a map like {node: (edge, [parents])}, where edge can be Any. Here, edges will be Callables
GraphType = Dict[str, Tuple[Any, List[str]]]


class Network(nn.Module):
    def __init__(self, architecture: ModelType):
        super(Network, self).__init__()
        # Convert ModelType specification of an architecture into 'flatter' GraphType
        self.graph = model2graph(architecture)
        if 'input' not in self.graph:
            warnings.warn("No 'input' node in the graph -- calls to forward() will need to specify inputs by name!")
        # Add all {name: operation} modules as a ModuleDict so that they appear in self.parameters
        self.module_dict = nn.ModuleDict({path: op for path, (op, inpts) in self.graph.items()
                                          if isinstance(op, nn.Module)})

    def forward(self, initial_inputs: Union[torch.Tensor, dict], warn_if_missing: bool = True):
        if not isinstance(initial_inputs, dict):
            initial_inputs = {'input': initial_inputs}
        # Run inputs forward and compute everything we can, or just compute 'outputs' if supplied.
        # Returned dict contains all inputs, hidden activations, and outputs, keyed by their 'path' name in the graph
        out = dict(initial_inputs)
        for layer_name, (op, layer_inputs) in self.graph.items():
            if layer_name in initial_inputs:
                continue
            elif all(inpt in out for inpt in layer_inputs):
                out[layer_name] = op(*(out[inpt] for inpt in layer_inputs))
            elif warn_if_missing:
                warnings.warn(f"Skipping {layer_name} because inputs are not available!")
        return out


#####################
## graph building ##
#####################

def _iter_dict(d: dict, join_op: Callable, prefix=tuple()):
    """Iterate a nested dict in order, yielding (k1k2k3, v) from a dict like {k1: {k2: {k3: v}}}. Uses the given join_op
    to join keys together. In this example, join_op(k1, k2, k3) should return k1k2k3
    """
    for k, v in d.items():
        new_prefix = prefix + (k,)
        if type(v) is dict:
            yield from _iter_dict(v, join_op, new_prefix)
        else:
            yield join_op(new_prefix), v


def _canonicalize_input(op):
    if isinstance(op, tuple):
        op, inpts = op
        if isinstance(inpts, str):
            return (op, [inpts])
        elif isinstance(inpts, int):
            return (op, [inpts])
        elif isinstance(inpts, list):
            return (op, inpts)
        else:
            raise ValueError(f"Cannot parse (op, inputs): {(op, inpts)}")
    else:
        # If no input is explicitly specified, assume a single input pointing to the previous layer's output
        return (op, [-1])


def _normpath(path, sep='/'):
    #simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == '..':
            parts.pop()
        elif p.startswith(sep):
            parts = [p]
        else:
            parts.append(p)
    return sep.join(parts)


def model2graph(model: ModelType, sep='/') -> GraphType:
    # Convert nested dict like {'layer1': {'step1': op1, 'step2': (op2, 'step1')}} into a flattened list like
    # [('layer1/step1', op1), ('layer1/step2', (op2, 'step1'))], and ensure some canonical form of 'op'. Note that we
    # don't convert this into a dict here in case of name conflicts.
    flattened_layers = [(joined_path, _canonicalize_input(op)) for joined_path, op in _iter_dict(model, join_op=sep.join)]

    # Resolve input references. In the example above, op1 has no named input so it will use whatever the previous layer
    # was, and 'step2' refers locally to 'step1'.
    graph = {}
    for idx, (path, (op, inpts)) in enumerate(flattened_layers):
        # Iterate over inputs and resolve any relative paths or indices
        if op is not None:
            for i, inpt in enumerate(inpts):
                if isinstance(inpt, int):
                    # 'inpt' is an int like -1, referring to some number of layers back. Get that layer's name as a string
                    inpts[i] = flattened_layers[idx + inpt][0]
                elif isinstance(inpt, str) and inpt not in graph:
                    # 'inpt' is a string specifying a particular layer, but it's not a key in 'graph' (not an absolute path)
                    inpts[i] = _normpath(sep.join([path, '..', inpt]), sep=sep)
                # Sanity-check
                assert inpts[i] in graph, \
                    f"While building graph, input to {path} includes {inpts[i]}, but keys so far are {list(graph.keys())}"

        # Add this op to the graph using its absolute paths
        graph[path] = (op, inpts)

    return graph


__all__ = ['model2graph', 'Network']