from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Protocol
from dataclasses import dataclass

if TYPE_CHECKING:
    from active_learning_ts.pipeline_element import PipelineElement
    from typing import Type


class PipelineConfig(Protocol):
    pipline_element: Type[PipelineElement]

    def instantiate(self) -> PipelineElement:
        ...
        
@dataclass(frozen = True)    
class BlueprintConfig(PipelineConfig):
    pipline_element: Type[PipelineElement]

    def instantiate(self):
        return self.pipline_element(self)