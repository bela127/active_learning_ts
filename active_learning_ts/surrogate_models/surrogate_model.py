from typing import Protocol, List
import tensorflow as tf


class SurrogateModel(Protocol):
    """
    The goal of a SurrogateModel is to as best as possible, emulate the Data Retrievement process. What constitutes a
    good emulation of the Data Retrievement process may be model/use-case specific
    """
    def uncertainty(self, point: List[tf.Tensor]) -> tf.Tensor:
        pass

    # TODO: if i understant this correctly, this method should not be here, but in the trainer. That however means that
    #   that you need specific training_strategies for specific models (obviously). So e.g. gausian_greedy_trainer,
    #   gaussian_blabla_trainer for the gaussion SM

    # I think the trainer has a totally different task then you think, it only updates the model with the new training data and possibly a loss function
    # the collection of the training data is the task of the query selector
        # still the training process can be model specific, so i guess the trainer needs a strong connection to the model
        # we talk about this
    def learn(self, points: List[tf.Tensor], values: List[tf.Tensor]):
        pass

    def query(self, points: List[tf.Tensor]) -> List[tf.Tensor]:
        pass
