from active_learning_ts.pipeline_element import PipelineElement
from typing import Protocol


class InstanceObjective(PipelineElement, Protocol):
    def apply(self, instance):
        objective_score = 1
        return objective_score
