from active_learning_ts.data_retrievement.data_source import DataSource


class TestDataSource(DataSource):
    def __init__(self):
        self.value_shape = (1,)
        self.point_shape = (1,)
