from typing import List

import tensorflow as tf

from active_learning_ts.data_instance import DataInstance


class Blackboard:
    """
    Blackboard pattern: https://en.wikipedia.org/wiki/Blackboard_(design_pattern)

    Stores all data instances. Function analogous to a Stack
    """

    def __init__(self) -> None:
        self.instances: List[DataInstance] = []

    @property
    def last_instance(self) -> DataInstance:
        return self.instances[-1]

    def instance_pool(self):
        return self.instances

    def add_instance(self, instance: DataInstance):
        self.instances.append(instance)

    def get_blackboard(self):
        # haskell > python
        f = lambda x: x.numpy().tolist() if isinstance(x, tf.Tensor) else x
        out = []
        for d in self.instances:
            temp = []
            for (k, v) in d.__dict__.items():
                temp.append('"' + k + '" : ' + str(f(v)))
            out.append('{\n' + ',\n'.join(temp) + '\n}')
        return '[' + ',\n'.join(out) + ']'
