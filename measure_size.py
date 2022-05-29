"This code was adapted from https://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/"

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import numpy as np
import types

try:
    from idlib import id
except ImportError:
    pass

def total_size(o, handlers={}, ignore={}, verbose=False, verbose_threshold=-1, return_dict=False, dict_threshold=-1):
    """ Returns the approximate memory footprint an object and all of its contents.
    """
    dict_handler = lambda d: iter(d.values())  # don't count dict keys
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set(id(i) for i in ignore)     # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    if return_dict:
        result_dict = dict()

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))

        if isinstance(o,types.ModuleType):  # these are just imports, ignore them
            return 0

        if callable(o):  # don't try to measure functions
            return 0

        base_modules_to_ignore =[
                                    'tensorflow',  # network objects counted separately
                                    'gym',  # environment to be ignored
                                ]
        modules_to_ignore = [
                                'vecmonitor',  # monitor to be ignored
                                'stable_baselines.bench.monitor',  # monitor
                                'stable_baselines.common.vec_env.base_vec_env',  # environment
                                'stable_baselines.common.vec_env.dummy_vec_env',  # environment
                            ]

        o_module = getattr(o,'__module__','')
        if o_module in modules_to_ignore:
            return 0

        for mod in base_modules_to_ignore:
            if o_module.split('.')[0]==mod:
                return 0

        if isinstance(o,np.ndarray):
            return o.nbytes

        s = getsizeof(o, default_size)

        if verbose and s>verbose_threshold:
            print(s, type(o), id(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                for child in handler(o):
                    child_size = sizeof(child)
                    s += child_size
                break

        if hasattr(o.__class__, '__slots__'):
            slots_size = sum(sizeof(getattr(o, x)) for x in o.__class__.__slots__ if hasattr(o, x))
            s += slots_size
        if hasattr(o, '__dict__'):
            for child in o.__dict__.values():
                child_size = sizeof(child)
                s += child_size

        if return_dict and s>dict_threshold:
                result_dict[id(o)]=(s,o)

        return s

    result = sizeof(o)
    if return_dict:
        return result, result_dict
    else:
        return result


##### Example call #####

if __name__ == '__main__':
    d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
    print(total_size(d, verbose=True))
