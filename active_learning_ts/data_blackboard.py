from typing import List

from active_learning_ts.data_instance import DataInstance


class Blackboard:
    def __init__(self) -> None:
        self.instances: List[DataInstance] = []

    @property
    def last_instance(self) -> DataInstance:
        return self.instances[-1]

    def instance_pool(self):
        return self.instances

    def add_instance(self, instance: DataInstance):
        self.instances.append(instance)
