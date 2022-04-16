from active_learning_ts.experiments.blueprint import Blueprint


class BlueprintInstance:
    """
    Do not directly implement.

    This is used to represent a blueprint, but where all the elements are instantiated
    """
    
    def __init__(self, blueprint: Blueprint):
        self.repeat = blueprint.repeat
        self.learning_steps = blueprint.learning_steps
        self.num_knowledge_discovery_queries = blueprint.num_knowledge_discovery_queries
        self.data_source = blueprint.data_source.instantiate()
        self.retrievement_strategy = blueprint.retrievement_strategy.instantiate()
        self.augmentation_pipeline = blueprint.augmentation_pipeline.instantiate()
        self.interpolation_strategy = blueprint.interpolation_strategy.instantiate()

        self.instance_level_objective = blueprint.instance_level_objective.instantiate()
        self.instance_cost = blueprint.instance_cost.instantiate()

        self.surrogate_model = blueprint.surrogate_model.instantiate()
        self.training_strategy = blueprint.training_strategy.instantiate()

        self.surrogate_sampler = blueprint.surrogate_sampler.instantiate()
        self.query_optimizer = blueprint.query_optimizer.instantiate()
        self.selection_criteria = blueprint.selection_criteria.instantiate()

        self.evaluation_metrics = [x.instantiate() for x in blueprint.evaluation_metrics]

        self.knowledge_discovery_sampler = blueprint.knowledge_discovery_sampler.instantiate()
        self.knowledge_discovery_task = blueprint.knowledge_discovery_task.instantiate()

