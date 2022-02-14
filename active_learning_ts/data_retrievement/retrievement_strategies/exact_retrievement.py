from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy
from active_learning_ts.data_retrievement.pool import Pool


class ExactRetrievement(RetrievementStrategy):
    
    def __init__(self):
        self.pool: Pool
