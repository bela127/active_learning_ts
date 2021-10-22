from active_learning_ts.training.trainer import Trainer
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.data_instance import DataInstance
from active_learning_ts.query_selection.query_selector import QuerySelector
from active_learning_ts.oracle import Oracle
from active_learning_ts.active_learner import ActiveLearner
from active_learning_ts.data_blackboard import Blackboard


class Experiment:
    def __init__(self, experiment_blueprint: Blueprint) -> None:
        self.blackboard: Blackboard = Blackboard()
        self.experiment_blueprint = experiment_blueprint
        self.setup(self.experiment_blueprint)

    def setup(self, experiment_blueprint: Blueprint):
        self.repeat: int = experiment_blueprint.repeat
        self.learning_steps: int = experiment_blueprint.learning_steps

        experiment_blueprint.training_strategy.post_init(experiment_blueprint.surrogate_model)
        experiment_blueprint.selection_criteria.post_init(experiment_blueprint.surrogate_model)
        experiment_blueprint.query_optimizer.post_init(experiment_blueprint.surrogate_model,
                                                       experiment_blueprint.selection_criteria)

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
        trainer = Trainer(self.blackboard,
                          experiment_blueprint.training_strategy)
        active_learner = ActiveLearner(oracle, query_selector, self.blackboard, trainer)
        self.active_learner: ActiveLearner = active_learner

    def run(self):
        for i in range(self.learning_steps):
            self.active_learner.learning_step()
