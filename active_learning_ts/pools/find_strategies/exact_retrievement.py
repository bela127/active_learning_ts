from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy


class ExactRetrievement(RetrievementStrategy):
    def post_init(self, data_source: DataSource):
        self.data_source = data_source
