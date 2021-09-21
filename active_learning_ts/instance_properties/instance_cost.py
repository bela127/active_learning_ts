from typing_extensions import Protocol
from active_learning_ts.pipeline_element import PipelineElement


class InstanceCost(PipelineElement,  Protocol):
    def apply(self, instance):
        objective_score = 1
        return objective_score
