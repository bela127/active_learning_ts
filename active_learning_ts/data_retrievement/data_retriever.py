from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.retrievement_strategy import (
    RetrievementStrategy,
)


class DataRetriever:
    def __init__(
        self,
        data_source: DataSource,
        retrievement_strategy: RetrievementStrategy,
        augmentation_pipeline: DataPipeline,
    ) -> None:
        self.data_source = data_source
        self.retrievement_strategy = retrievement_strategy
        self.augmentation_pipeline = augmentation_pipeline

    def query(self, query_candidates):
        actual_queries, query_results = self.retrievement_strategy.retriev(
            self.data_source, query_candidates
        )
        augmented_query_results = self.augmentation_pipeline.apply(query_results)
        return actual_queries, augmented_query_results
