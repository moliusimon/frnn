from topology import Topology


class TopologyFolded(Topology):
    def __init__(self, specifications):
        Topology.__init__(self, specifications)

    def forward(self, x):
        rets_down = []
        for l in self._flatten_topology():
            if hasattr(l, 'input'):
                rets_down.append(x)
            x = l.forward(x)

        return x, rets_down

    def backward(self, x):
        rets_up = []
        for l in self._flatten_topology()[::-1]:
            x = l.backward(x)
            if hasattr(l, 'input'):
                rets_up.append(x)

        return x, rets_up[::-1]

    def update_state(self):
        updater = self._flatten_topology()[-1]
        return updater.forward(updater.input)
