import numpy as np


def byte_offset(array, source):
    return array.__array_interface__['data'][0] - np.byte_bounds(source)[0]


class SharedPickleList(object):
    def __init__(self, arrays):
        self.arrays = list(arrays)

    def __getstate__(self):
        unique_ids = {id(array) for array in self.arrays}
        source_arrays = {}
        view_tuples = {}
        for array in self.arrays:
            if array.base is None or id(array.base) not in unique_ids:
                # only use views if the base is also being pickled
                source_arrays[id(array)] = array
            else:
                view_tuples[id(array)] = (array.shape,
                                          array.dtype,
                                          id(array.base),
                                          byte_offset(array, array.base),
                                          array.strides)
        order = [id(array) for array in self.arrays]
        return (source_arrays, view_tuples, order)

    def __setstate__(self, state):
        source_arrays, view_tuples, order = state
        view_arrays = {}
        for k, view_state in view_tuples.items():
            (shape, dtype, source_id, offset, strides) = view_state
            buffer = source_arrays[source_id].data
            array = np.ndarray(shape, dtype, buffer, offset, strides)
            view_arrays[k] = array
        self.arrays = [source_arrays[i]
                       if i in source_arrays
                       else view_arrays[i]
                       for i in order]
