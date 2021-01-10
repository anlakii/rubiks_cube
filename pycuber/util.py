import collections.abc

"""
Utilities
"""
class FrozenDict(collections.abc.Mapping):
    def __init__(self, *args, **kwargs):
        self._map = dict(*args, **kwargs)

    def __getitem__(self, k):
        return self._map[k]

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def __str__(self):
        return str(self._map)
