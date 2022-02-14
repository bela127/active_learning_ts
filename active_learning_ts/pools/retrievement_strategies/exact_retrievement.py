from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy
from active_learning_ts.pool import Pool


class ExactRetrievement(RetrievementStrategy):
    
    def __init__(self):
        self.pool: Pool
