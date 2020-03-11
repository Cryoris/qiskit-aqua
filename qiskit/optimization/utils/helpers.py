# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from typing import Union, List, Tuple, Dict, Sequence

from cplex import SparsePair, SparseTriple

from qiskit.optimization.utils import QiskitOptimizationError

_defaultgetindex = {}


def _defaultgetindexfunc(item):
    if item not in _defaultgetindex:
        _defaultgetindex[item] = len(_defaultgetindex)
    return _defaultgetindex[item]


def _cachelookup(item, getindexfunc, cache):
    if item not in cache:
        cache[item] = getindexfunc(item)
    return cache[item]


def _convert_sequence(seq, getindexfunc, cache):
    results = []
    for item in seq:
        if isinstance(item, str):
            idx = _cachelookup(item, getindexfunc, cache)
            results.append(idx)
        else:
            results.append(item)
    return results


def listify(x):
    """Returns [x] if x isn't already a list.

    This is used to wrap arguments for functions that require lists.
    """
    # Assumes name to index conversions already done.
    assert not isinstance(x, str)
    if isinstance(x, Sequence):
        return x
    else:
        return [x]


def convert(name, getindexfunc=_defaultgetindexfunc, cache=None):
    """Converts from names to indices as necessary.

    If name is a string, an index is returned.

    If name is a sequence, a sequence of indices is returned.

    If name is neither (i.e., it's an integer), then that is returned
    as is.

    getindexfunc is a function that takes a name and returns an index.

    The optional cache argument allows for further localized
    caching (e.g., within a loop).
    """
    # In some cases, it can be beneficial to cache lookups.
    if cache is None:
        cache = {}
    if isinstance(name, str):
        return _cachelookup(name, getindexfunc, cache)
    elif isinstance(name, Sequence):
        # It's tempting to use a recursive solution here, but that kills
        # performance for the case where all indices are passed in (i.e.,
        # no names). This is due to the fact that we end up doing the
        # extra check for sequence types over and over (above).
        return _convert_sequence(name, getindexfunc, cache)
    else:
        return name


class NameIndex:
    def __init__(self):
        self._dict = {}

    def to_dict(self) -> Dict[str, int]:
        return self._dict

    def __contains__(self, item: str) -> bool:
        return item in self._dict

    def build(self, names: List[str]):
        self._dict = {e: i for i, e in enumerate(names)}

    def _convert_one(self, arg: Union[str, int]) -> int:
        if isinstance(arg, int):
            return arg
        if not isinstance(arg, str):
            raise QiskitOptimizationError('Invalid argument" {}'.format(arg))
        if arg not in self._dict:
            self._dict[arg] = len(self._dict)
        return self._dict[arg]

    def convert(self, *args) -> Union[int, List[int]]:
        if len(args) == 0:
            return list(self._dict.values())
        elif len(args) == 1:
            a0 = args[0]
            if isinstance(a0, (int, str)):
                return self._convert_one(a0)
            elif isinstance(a0, Sequence):
                return [self._convert_one(e) for e in a0]
            else:
                raise QiskitOptimizationError('Invalid argument: {}'.format(args))
        elif len(args) == 2:
            begin = self._convert_one(args[0])
            end = self._convert_one(args[1]) + 1
            return list(range(begin, end))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))


def init_list_args(*args):
    """Initialize default arguments with empty lists if necessary."""
    return tuple([] if a is None else a for a in args)


if __debug__:

    # With non-optimzied bytecode, validate_arg_lengths actually does
    # something.
    def validate_arg_lengths(arg_list, allow_empty=True):
        """Checks for equivalent argument lengths.

        If allow_empty is True (the default), then empty arguments are not
        checked against the max length of non-empty arguments. Some functions
        allow NULL arguments in the Callable Library, for example.
        """
        arg_lengths = [len(x) for x in arg_list]
        if allow_empty:
            arg_lengths = [x for x in arg_lengths if x > 0]
        if len(arg_lengths) == 0:
            return
        max_length = max(arg_lengths)
        for arg_length in arg_lengths:
            if arg_length != max_length:
                raise QiskitOptimizationError("inconsistent arguments")

else:

    # A no-op if using python -O or the PYTHONOPTIMIZE environment
    # variable is defined as a non-empty string.
    def validate_arg_lengths(arg_list, allow_empty=True):
        pass


def unpack_pair(item: Union[SparsePair, List, Tuple]) -> Tuple[List[int], List[float]]:
    """Extracts the indices and values from an object.

    The argument item can either be an instance of SparsePair or a
    sequence of length two.

    Example usage:

    >>> sp = SparsePair()
    >>> ind, val = unpack_pair(sp)
    >>> lin_expr = [[], []]
    >>> ind, val = unpack_pair(lin_expr)
    """
    if isinstance(item, SparsePair):
        assert item.isvalid()
        ind, val = item.unpack()
    elif isinstance(item, (tuple, list)):
        ind, val = item[0:2]
    else:
        raise QiskitOptimizationError('Invalid object for unpack_pair {}'.format(item))
    validate_arg_lengths([ind, val])
    return ind, val


def unpack_triple(item: Union[SparseTriple, List, Tuple]) \
        -> Tuple[List[int], List[int], List[float]]:
    """Extracts the indices and values from an object.

    The argument item can either be an instance of SparseTriple or a
    sequence of length three.

    Example usage:

    >>> st = SparseTriple()
    >>> ind1, ind2, val = unpack_triple(st)
    >>> quad_expr = [[], [], []]
    >>> ind1, ind2, val = unpack_triple(quad_expr)
    """
    if isinstance(item, SparseTriple):
        assert item.isvalid()
        ind1, ind2, val = item.unpack()
    elif isinstance(item, (list, tuple)):
        ind1, ind2, val = item[0:3]
    validate_arg_lengths([ind1, ind2, val])
    return ind1, ind2, val


def max_arg_length(arg_list):
    """Returns the max length of the arguments in arg_list."""
    return max(len(x) for x in arg_list)