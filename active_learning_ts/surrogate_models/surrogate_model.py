from typing import Protocol, List
import tensorflow as tf


class SurrogateModel(Protocol):
    def uncertainty(self, point: List[tf.Tensor]) -> tf.Tensor:
        pass

    # TODO: if i understant this correctly, this method should not be here, but in the trainer. That however means that
    #   that you need specific trainers for specific models (obviously). So e.g. gausian_greedy_trainer,
    #   gaussian_blabla_trainer for the gaussion SM
    def learn(self, points: List[tf.Tensor], values: List[tf.Tensor]):
        pass

    def query(self, points: List[tf.Tensor]) -> List[tf.Tensor]:
        pass
