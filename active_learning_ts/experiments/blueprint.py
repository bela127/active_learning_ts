from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.retrievement_strategy import (
    RetrievementStrategy,
)
from typing import Protocol, List

from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from active_learning_ts.training.training_strategy import TrainingStrategy


class Blueprint(Protocol):
    """
    A blueprint is created in order to set up an experiment.

    Following field MUST be in the blueprint file, with the same names
    """
    repeat: int
    learning_steps: int

    data_source: DataSource
    retrievement_strategy: RetrievementStrategy
    augmentation_pipeline: DataPipeline

    instance_level_objective: InstanceObjective
    instance_cost: InstanceCost

    surrogate_model: SurrogateModel
    training_strategy: TrainingStrategy

    query_optimizer: QueryOptimizer
    selection_criteria: SelectionCriteria

    evaluation_metrics: List[EvaluationMetric]
