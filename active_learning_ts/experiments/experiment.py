from active_learning_ts.evaluation.evaluator import Evaluator
from active_learning_ts.training.trainer import Trainer
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.data_instance import DataInstance
from active_learning_ts.query_selection.query_selector import QuerySelector
from active_learning_ts.oracle import Oracle
from active_learning_ts.active_learner import ActiveLearner
from active_learning_ts.data_blackboard import Blackboard


class Experiment:
    """
    Builds an experiment based on the instructions obtained from the blueprint.

    An experiment consists of:
    A blackboard where data is stored
    An Oracle in order to query data from the data source
    Another Oracle in order to query data from the surrogate
    A trainer in order to train the surrogate
    An Evaluator in order to evaluate the Experiment, according to the given metrics
    """

    def __init__(self, experiment_blueprint: Blueprint) -> None:
        self.blackboard: Blackboard = Blackboard()
        self.experiment_blueprint = experiment_blueprint
        self.setup(self.experiment_blueprint)

    def setup(self, experiment_blueprint: Blueprint):
        self.repeat: int = experiment_blueprint.repeat
        self.learning_steps: int = experiment_blueprint.learning_steps

        data_source_pool = experiment_blueprint.data_source.possible_queries()

        experiment_blueprint.retrievement_strategy.post_init(data_source_pool)
        experiment_blueprint.training_strategy.post_init(experiment_blueprint.surrogate_model)
        experiment_blueprint.selection_criteria.post_init(experiment_blueprint.surrogate_model)
        experiment_blueprint.query_optimizer.post_init(experiment_blueprint.surrogate_model,
                                                       experiment_blueprint.selection_criteria,
                                                       experiment_blueprint.retrievement_strategy.get_query_pool())

        data_retriever = DataRetriever(
            experiment_blueprint.data_source,
            experiment_blueprint.retrievement_strategy,
            experiment_blueprint.augmentation_pipeline,
        )
        oracle = Oracle(
            DataInstance,
            self.blackboard,
            data_retriever,
            experiment_blueprint.instance_cost,
            experiment_blueprint.instance_level_objective,
        )

        sg_oracle = Oracle(
            DataInstance,
            self.blackboard,
            data_retriever,
            experiment_blueprint.instance_cost,
            experiment_blueprint.instance_level_objective,
        )
        query_selector = QuerySelector(
            self.blackboard,
            experiment_blueprint.query_optimizer,
            experiment_blueprint.selection_criteria,
            sg_oracle
        )

        trainer = Trainer(
            self.blackboard,
            experiment_blueprint.training_strategy
        )

        evaluator = Evaluator(
            evaluation_metrics=experiment_blueprint.evaluation_metrics,
            blackboard=self.blackboard,
            blueprint=experiment_blueprint
        )

        active_learner = ActiveLearner(oracle, query_selector, self.blackboard, trainer, evaluator)
        self.active_learner: ActiveLearner = active_learner

    def run(self):
        for i in range(self.learning_steps):
            self.active_learner.learning_step()
