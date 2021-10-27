from typing import Union
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy
from active_learning_ts.pool import Pool


class ExactRetrievement(RetrievementStrategy):

    def __init__(self, query_pool: Union[Pool, None] = None) -> None:
        super().__init__(query_pool=query_pool)
    pass