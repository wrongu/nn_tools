import torch.nn as nn
from typing import Dict, Tuple, Callable, Union, Any, Iterable, List

# An 'Operation' is essentially a callable object that behaves like a nn.Module. Use None for input values.
OpType = Union[None, Callable, nn.Module, 'ModelType']
# Specify inputs by name (string), relative index (negative integer), or a list of either
InputType = Union[str, int, Iterable[Union[str, int]]]
# A model is a dict of (i) {name: op}, or (ii) {name: (op, input)}, where 'op' can itself be a sub-model
ModelType = Dict[str, Union[OpType, Tuple[OpType, InputType]]]
# A Graph is a map like {node: (edge, [parents])}, where edge can be Any. Here, edges will be Callables
GraphType = Dict[str, Tuple[Any, List[str]]]