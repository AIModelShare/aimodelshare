import ray
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score

import aimodelshare as ai

# !pip install optuna


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
            "starting_points": "data",
            "log_file_name": "experiment.log",
            "seed": 7654321,
        }
        self.automl.fit(self.X_train, self.y_train, **settings)

    def predict(self):
        if self.X_test is not None:
            y_pred = self.automl.predict(self.X_test)
            # y_test = self.automl._label_transformer.transform(self.y_test)
            print("accuracy =", 1 - sklearn_metric_loss_score("accuracy", y_pred, self.y_test))
        else:
            raise Exception("No valid testset!")
        return y_pred

    def parallel_tuning(self):
        try:
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)
        except RuntimeError:
            pass
        self.automl.fit(self.X_train, self.y_train, time_budget=30, n_jobs=2, n_concurrent_trials=2)
        return self.automl.model.estimator

    def save_model(self):
        preprocessor = self.automl._preprocess
        ai.export_preprocessor(preprocessor, "")
        return self.automl.model.estimator, preprocessor


if __name__ == "__main__":
    datasets_name = "titanic"
    datasets_train, datasets_test, labels_train, labels_test, example_data, _ = ai.import_quickstart_data(datasets_name)

    automl = AutoML_Tabular(datasets_train, labels_train, datasets_test, labels_test)
    automl.fit()
    estimator = automl.parallel_tuning()
    predicted_labels = automl.predict()
    automl.save_model()

