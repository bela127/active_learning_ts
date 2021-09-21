from typing import Protocol


class DataPipeline(Protocol):
    def __init__(self) -> None:
        pass

    def apply(self, data_input):
        data_output = data_input
        return data_output
