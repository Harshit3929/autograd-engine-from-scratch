class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x, activation="relu"):

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if activation == "relu":
            return act.relu()
        elif activation == "sigmoid":
            return act.sigmoid()
        elif activation == "tanh":
            return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neuron = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x, activation="relu"):
        outs = [n(x) for n in self.neuron]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neuron for p in neuron.parameters()]


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x, activation="relu"):
        for layers in self.layers:
            x = layers(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]