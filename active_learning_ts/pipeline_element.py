from typing import Protocol


class PipelineElement(Protocol):
    def calc(self, input):
        return input
