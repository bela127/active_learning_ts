from typing import Union
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy
from active_learning_ts.pool import Pool


# TODO: discuss the removal (replacement) of Retrievement strategies. It is much better for the Pool to be given a
#  retrievement strategy object, that it can use to find query candidates
class ExactRetrievement(RetrievementStrategy):

    def __init__(self, query_pool: Union[Pool, None] = None) -> None:
        super().__init__(query_pool=query_pool)

    pass
