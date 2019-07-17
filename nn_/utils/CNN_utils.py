import itertools
from math import ceil

# import asciichartpy
# from tabulate import tabulate


class LayerStats:

    def __init__(self, kernel_size, stride, dilation, padding, name=""):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.prev_layer = None
        self.input_size = None
        self.name = name

    def dilated_kernel_size(self):
        return (self.kernel_size - 1) * self.dilation + 1

    def padding(self):
        return self.output_size() - self.raw_output_size()

    def output_size(self):
        return ((self.input_size + 2 * self.padding - self.dilation * (
                self.kernel_size - 1) - 1) // self.stride) + 1

    #
    def raw_output_size(self):
        return ceil((self.input_size - (self.dilated_kernel_size() - 1)) / self.stride)

    # def output_size(self):
    #     # TODO figure out pytorch padding type
    #
    #     if self.padding == 0:
    #         return self.raw_output_size()
    #     else:
    #         # TODO
    #         raise NotImplementedError
    #         # return ceil(self.input_size / self.stride)

    def growth_rate(self):
        growth_rate = self.prev_layer.growth_rate() if self.prev_layer is not None else 1
        return self.stride * growth_rate

    def receptive_field(self):
        if self.prev_layer is not None:
            return self.prev_layer.receptive_field() + (
                    (self.kernel_size - 1) * self.dilation) * self.prev_layer.growth_rate()
        else:
            return (self.kernel_size - 1) * self.dilation + 1

    # print("{} \t {}")


def get_layer_stats(cnn):
    return list(itertools.chain(*[layer.get_layer_stats() for layer in cnn.layers]))


def receptive_field(cnn):
    layers = get_layer_stats(cnn)

    _receptive_field = []
    for i in range(len(layers)):
        layer = layers[i]

        if i > 0:
            prev_layer = layers[i - 1]
            layer.prev_layer = prev_layer
        else:
            layer.prev_layer = None

        _receptive_field.append(layer.receptive_field())
    return _receptive_field


def output_size_layer(layer_elems, input_size):
    layers = layer_elems

    _output_size = []
    for i in range(len(layers)):
        layer = layers[i]

        if i > 0:
            prev_layer = layers[i - 1]
            layer.prev_layer = prev_layer
            layer.input_size = prev_layer.output_size()

        else:
            layer.prev_layer = None
            layer.input_size = input_size

        _output_size.append(layer.output_size())
    return _output_size[-1]


def output_size(cnn):
    layers = get_layer_stats(cnn)

    _output_size = []
    for i in range(len(layers)):
        layer = layers[i]

        if i > 0:
            prev_layer = layers[i - 1]
            layer.prev_layer = prev_layer
            layer.input_size = prev_layer.output_size()

        else:
            layer.prev_layer = None
            layer.input_size = input_size

        _output_size.append(layer.output_size())
    return _output_size


def cnn_stats(cnn, input_size):
    layers = get_layer_stats(cnn)

    _print_tabular = []
    _receptive_field = []
    for i in range(len(layers)):
        layer = layers[i]

        if i > 0:
            prev_layer = layers[i - 1]
            layer.prev_layer = prev_layer
            layer.input_size = prev_layer.output_size()
        else:
            layer.prev_layer = None
            layer.input_size = input_size

        _print_tabular.append([i, type(layer).__name__, layer.kernel_size, layer.stride, layer.dilation,
                               layer.padding, layer.input_size,
                               layer.output_size(), layer.receptive_field()])
        _receptive_field.append(layer.receptive_field())
    print(tabulate(_print_tabular, headers=(
        "Layer #", "Name", "Kernel Size", "Stride", "Dilation", "Padding", "Input Size", "Output Size",
        "Receptive Field")))

    print(asciichartpy.plot(
        list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in _receptive_field))
        , cfg={"height": 10}))

    return layer.output_size()


def cnn_receptive_field(cnn):
    _receptive_field = receptive_field(cnn)
    return _receptive_field[-1]


if __name__ == '__main__':
    input_size = 1000

    layers = [
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
        LayerStats(kernel_size=4, stride=2, dilation=2, padding_type="VALID"),
    ]

    print_tabular = []
    receptive_field = []
    for i in range(len(layers)):
        layer = layers[i]

        if i > 0:
            prev_layer = layers[i - 1]
            layer.prev_layer = prev_layer
            layer.input_size = prev_layer.output_size()
        else:
            layer.prev_layer = None
            layer.input_size = input_size

        print_tabular.append([i, layer.name, layer.kernel_size, layer.stride, layer.dilation,
                              layer.padding(), layer.input_size,
                              layer.output_size(), layer.receptive_field()])
        receptive_field.append(layer.receptive_field())

    print(tabulate(print_tabular, headers=(
        "Layer #", "Name", "Kernel Size", "Stride", "Dilation", "Padding", "Input Size", "Output Size",
        "Receptive Field")))

    print(asciichartpy.plot(
        list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in receptive_field))
        , cfg={"height": 20}))
