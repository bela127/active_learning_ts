from typing import Protocol


class DataSource(Protocol):
    def __init__(self) -> None:
        pass

    def query(self, actual_queries):
        query_results = actual_queries
        return query_results

    def possible_queries(self):
        return None
