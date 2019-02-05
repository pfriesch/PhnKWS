import pickle

import numpy as np
import torch

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


# unit tests
def check_roundtrip(arrays):
    dumped = torch.save(
        SharedPickleList(arrays), "test.pyt")
    unpickled_arrays = torch.load("test.pyt").arrays
    assert all(a.shape == b.shape and (a == b).all()
               for a, b in zip(arrays, unpickled_arrays))


# indexers = [0, None, slice(None), slice(2), slice(None, -1),
#             slice(None, None, -1), slice(None, 6, 2)]

source0 = np.random.rand(100, 8, 5)
view0 = source0[0:4]
view1 = source0[1:4]
view2 = source0[3:4]
view3 = source0[5:-1]
arrays0 = [view0, view1, view2, view3]
check_roundtrip([source0] + arrays0)

# source1 = np.random.randint(100, size=(8, 10))
# arrays1 = [np.asarray(source1[k1, k2]) for k1 in indexers for k2 in indexers]
# check_roundtrip([source1] + arrays1)

source = np.random.rand(1000)
arrays = [source] + [source[n:] for n in range(99)]
print(len(pickle.dumps(arrays, protocol=-1)))
# 766372
print(len(pickle.dumps(SharedPickleList(arrays), protocol=-1)))
# 11833
