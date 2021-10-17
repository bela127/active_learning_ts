from typing import Protocol, List
import tensorflow as tf


class SurrogateModel(Protocol):
    def uncertainty(self, point: List[tf.Tensor]) -> tf.Tensor:
        pass

    def learn(self, points: List[tf.Tensor], values: List[tf.Tensor]):
        pass

    def get_values(self, points: List[tf.Tensor]) -> List[tf.Tensor]:
        pass
