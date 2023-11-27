"""
Methods for managing context and attributes
"""

import functools
import os
import sys
import pandas as pd
from contextlib import contextmanager, redirect_stderr, redirect_stdout

__all__ = [
    'suppress_stdout_stderr',
    'suppress_stdout',
    'remove_prefix',
    'remove_suffix',
    'rsetattr',
    'rgetattr',
    'try_except',
    'storeH5',
]


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@contextmanager
def suppress_stdout(show_output):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if not show_output:
            sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text  # or whatever


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def try_except(success, failure, *exceptions):
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure

def storeH5(df, path=None, key=None, mode=None, **kwargs):
    if path is not None:
        if mode is None:
            if os.path.isfile(path):
                mode = 'a'
            else:
                mode = 'w'

        if key is not None:

            try:
                store = pd.HDFStore(path, mode=mode)
                store[key] = df
                store.close()
            except:
                if mode == 'a':
                    storeH5(df, path=path, key=key, mode='w', **kwargs)
        elif key is None and isinstance(df, dict):
            store = pd.HDFStore(path, mode=mode)
            for k, v in df.items():
                store[k] = v
            store.close()
        else:
            raise ValueError('H5key not provided.')