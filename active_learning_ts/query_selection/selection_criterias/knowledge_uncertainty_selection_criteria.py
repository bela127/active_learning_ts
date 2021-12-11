import tensorflow as tf

from active_learning_ts.query_selection.selection_criteria import SelectionCriteria


class KnowledgeUncertaintySelectionCriteria(SelectionCriteria):
    """
    Used to encourage the model to explore the search space as much as possible, by encouraging training in areas where
    uncertainty of the knowledge discovery task is high
    """

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        out = self.knowledge_discovery.uncertainty(queries)
        return tf.reshape(out, (out.shape[0], 1))
