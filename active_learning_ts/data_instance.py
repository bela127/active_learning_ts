from dataclasses import dataclass
from typing import Any
from typing_extensions import Protocol


@dataclass()
class DataInstance:
    query_candidates: Any = None
    actual_queries: Any = None
    query_results: Any = None
    quality: Any = None
    cost: Any = None


class DataInstanceFactory(Protocol):
    def __call__(self) -> DataInstance:
        return DataInstance()
