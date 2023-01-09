import ray
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score


class AutoML_Tabular:
    def __init__(self, X_train, y_train, X_test=None, y_test=None, num_cpus=4):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_cpus = num_cpus
        self.automl = AutoML()

    def fit(self):
        settings = {
            "time_budget": 600,
            "metric": "accuracy",
            "task": "classification",
            "log_file_name": "experiment.log",
            "seed": 7654321,
        }
        self.automl.fit(self.X_train, self.y_train, **settings)
        return self.automl.model.estimator

    def predict(self):
        if self.X_test:
            y_pred = self.automl.predict(self.X_test)
            print("accuracy =", sklearn_metric_loss_score("accuracy", y_pred, self.y_test))
        else:
            raise Exception("No valid testset!")
        return y_pred

    def parallel_tuning(self):
        ray.init(num_cpus=self.num_cpus)
        self.automl.fit(self.X_train, self.y_train, time_budget=30, n_jobs=2, n_concurrent_trials=2)

