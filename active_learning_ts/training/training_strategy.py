from typing import Protocol, List
import tensorflow as tf


class TrainingStrategy(Protocol):
    def __init__(self):
        pass

    def train(self, query: List[tf.Tensor], feedback: List[tf.Tensor]):
        pass
