import os
import shutil
import glob
import numpy as np
import pandas as pd
from typing import Any, Sequence

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes)
from skl2onnx.common.data_types import FloatTensorType
import xgboost
import lightgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

import dabl
from supervised.automl import AutoML
import aimodelshare as ai
from aimodelshare import ModelPlayground
from aimodelshare.aws import configure_credentials, set_credentials
from aimodelshare.aimsonnx import model_to_onnx


class autoMLTabular:
    def __init__(self, dataset_names: Sequence,
                 keyword="label",
                 automl_dir="automl",
                 mode=None,
                 playground: ModelPlayground = None,
                 **kwargs):
        """
        :param dataset_names: There are a few possibilities, a series of csv file, a directory or a keyword
        :param keyword: represent a label keyword
        :param automl_dir:
        :param mode:
        :param playground:
        :param kwargs:
        """
        self.datasets_train = None
        self.labels_train = None
        self.datasets_test = None
        self.labels_test = None
        self.keyword = keyword  # This is the name for the ground truth column

        self.automl_dir = automl_dir
        self.automl = AutoML(results_path=automl_dir, mode=mode if mode else "Explain")

        if isinstance(dataset_names, list) or isinstance(dataset_names, tuple):
            self.datasets_train, self.datasets_test = pd.read_csv(dataset_names[0]), pd.read_csv(
                dataset_names[1])
            self.labels_train = self.datasets_train.loc[keyword]
            self.datasets_train = self.datasets_train.drop([keyword], axis=1)
            self.labels_test = self.datasets_test.loc[keyword]
            self.datasets_test = self.datasets_test.drop([keyword], axis=1)
        elif isinstance(dataset_names, str) and os.path.isdir(dataset_names):
            dataset_names = sorted(glob.glob("/".join((dataset_names, "*.csv"))), reverse=True)  # train.csv, test.csv
            self.datasets_train, self.datasets_test = pd.read_csv(dataset_names[0]), pd.read_csv(
                dataset_names[1])
            self.labels_train = self.datasets_train.loc[keyword]
            self.datasets_train = self.datasets_train.drop([keyword], axis=1)
            self.labels_test = self.datasets_test.loc[keyword]
            self.datasets_test = self.datasets_test.drop([keyword], axis=1)
        else:
            # Now it is not working
            try:
                self.datasets_train, self.datasets_test, self.labels_train, self.labels_test, example_data, y_test_labels = ai.import_quickstart_data(
                    dataset_names)
            except:
                self.datasets_train, self.datasets_test, self.labels_train, self.labels_test, example_data = kwargs["datasets_train"], \
                                                                                               kwargs["datasets_test"], \
                                                                                               kwargs["labels_train"], \
                                                                                               kwargs["labels_test"], \
                                                                                               kwargs.get("example_data", None)
        self.preprocessor = dabl.EasyPreprocessor()  # This is for fast preprocessing

        self._fast_preprocessing_fit()
        self._fast_preprocessing_transform()

        self.configure_credential()

        if playground is None or playground.playground_url is None:
            # Create a new competition
            playground = ModelPlayground(model_type="tabular", classification=True, private=False)
            try:
                playground.deploy("model.onnx", "preprocessor.zip", self.labels_train, example_data if example_data else pd.DataFrame(self.datasets_train[0:4]))
            except:
                playground.deploy("model.onnx", "preprocessor.zip", self.labels_train, pd.DataFrame(self.datasets_train[0:4]))
            playground.create_competition(data_directory='competition_data',
                                                      y_test=self.labels_test)
        self.competition_playground = ai.Competition(playground.playground_url)
        self.set_credential(apiurl=playground.playground_url)

    def _fast_preprocessing_fit(self):
        if self.datasets_train is not None:
            self.datasets_train = dabl.clean(self.datasets_train)
            self.preprocessor.fit(self.datasets_train)
            self.datasets_train = self.preprocessor.transform(self.datasets_train)
            self._label_preprocessing_train()
            return
        raise Exception("No training set!")

    def _fast_preprocessing_transform(self):
        if self.datasets_test is not None:
            self.datasets_test = dabl.clean(self.datasets_test)
            self.datasets_test = self.preprocessor.transform(self.datasets_test)
            self._label_preprocessing_test()
            return
        raise Exception("No testing set!")

    def _label_preprocessing_train(self):
        if self.labels_train.dtype == int:
            self.labels_train_processed = self.labels_train.copy()
            return
        self.labels_train_processed = np.empty((len(self.labels_train),))
        if self.labels_train is not None:
            if isinstance(self.labels_train, pd.DataFrame):
                unique_labels = pd.unique(self.labels_train)
            else:
                unique_labels = np.unique(self.labels_train)
            self.mapping = {label: i for i, label in enumerate(unique_labels)}
            i = 0
            for label in self.labels_train:
                self.labels_train_processed[i] = self.mapping[label]
                i += 1

    def _label_preprocessing_test(self):
        if self.labels_test.dtype == int:
            self.labels_test_processed = self.labels_test.copy()
            return
        self.labels_test_processed = np.empty((len(self.labels_test),))
        if self.labels_test is not None:
            i = 0
            for label in self.labels_test:
                self.labels_test_processed[i] = self.mapping[label]
                i += 1

    def train(self):
        """
        Model fit
        """
        assert self.datasets_train is not None
        self.automl.fit(self.datasets_train, self.labels_train_processed)

    def predict(self):
        """
        AutoML predict
        """
        assert self.datasets_test is not None
        predictions = self.automl.predict(self.datasets_test)
        return predictions

    def prediction_score(self):
        """
        AutoML score
        """
        assert self.datasets_test is not None
        return self.automl.score(self.datasets_test, self.labels_test_processed)

    def get_leaderboard(self, competition=False):
        """
        Get leaderboard of different models
        """
        if competition:
            data = self.competition_playground.get_leaderboard()
            self.competition_playground.stylize_leaderboard(data)
        else:
            print(self.automl.get_leaderboard())

    def get_ensemble_model(self):
        """
        Record ensemble model and return sklearn pipeline and preprocessor function
        """
        selected_models = self.automl.ensemble.selected_models
        parsed_models = []

        for model_ in selected_models:
            model = model_["model"]
            preprocessor = model.preprocessings[0]
            learner = model.learners[0].model
            parsed_models.append((preprocessor, learner))

        preprocessor = parsed_models[0][0]

        model_pipelines = []
        for preprocessor, model in parsed_models:
            if not hasattr(model, "fit"):

                if isinstance(model, xgboost.core.Booster):
                    new_model = xgboost.XGBClassifier()
                    new_model._Booster = model

                    update_registered_converter(
                        XGBClassifier, 'XGBoostXGBClassifier',
                        calculate_linear_classifier_output_shapes, convert_xgboost,
                        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
                elif isinstance(model, lightgbm.Booster):
                    new_model = model
                    update_registered_converter(
                        LGBMClassifier, 'LightGbmLGBMClassifier',
                        calculate_linear_classifier_output_shapes, convert_lightgbm,
                        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
                else:
                    continue
                assert hasattr(model,
                               "predict")  # and hasattr(preprocessor, "fit") and hasattr(preprocessor, "transform")
                pipeline = new_model
            else:
                pipeline = model
            model_pipelines.append(pipeline)

        estimators = [("model_" + str(i), pipeline) for i, pipeline in enumerate(model_pipelines)]

        stacked_classifier = StackingClassifier(estimators=estimators,
                                                final_estimator=LogisticRegression())

        pipeline = make_pipeline(stacked_classifier)
        packed_fitting = self._fit_pipeline(pipeline)
        predicted_labels = packed_fitting[1]
        return pipeline, preprocessor, predicted_labels

    def _fit_pipeline(self, pipeline):
        """
        Fit pipeline for onnx save
        """
        pipeline.fit(self.datasets_train, self.labels_train_processed)
        predicted_labels = pipeline.predict(self.datasets_test)
        if self.labels_test_processed is not None:
            score_ = pipeline.score(self.datasets_test, self.labels_test_processed)
            return pipeline, predicted_labels, score_
        return pipeline, predicted_labels

    def reset(self):
        if os.path.exists(self.automl_dir):
            shutil.rmtree(self.automl_dir)

    def to_onnx(self, pipeline_fitted):
        initial_types = [('float_input', FloatTensorType([None, self.datasets_train.shape[1]]))]
        onnx_model = model_to_onnx(pipeline_fitted,
                                   initial_types=initial_types,
                                   framework='sklearn',
                                   transfer_learning=False,
                                   deep_learning=False)
        with open("model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        return onnx_model

    @staticmethod
    def configure_credential():
        if not os.path.exists('credentials.txt'):
            configure_credentials()
        else:
            print("Credentials already exist!")

    @staticmethod
    def set_credential(apiurl=None):
        set_credentials(credential_file="credentials.txt", type="deploy_model")
        if apiurl:
            set_credentials(apiurl=apiurl)

    @staticmethod
    def zip_preprocessor(preprocessor):
        ai.export_preprocessor(preprocessor, "")

    def submit_leaderboard(self, prediction_labels):
        reversed_mapping = {self.mapping[key]: key for key in self.mapping}
        if self.competition_playground is not None:
            self.competition_playground.submit_model(model_filepath="model.onnx",
                                                     preprocessor_filepath="preprocessor.zip",
                                                     prediction_submission=list(
                                                         map(lambda x: reversed_mapping[x], prediction_labels)))
        else:
            print("No competition created.")

    def activate(self):
        self.reset()
        self.train()
        print("Prediction score", self.prediction_score())
        self.get_leaderboard()
        pipeline, preprocessor, predicted_labels = self.get_ensemble_model()
        onnx_model = self.to_onnx(pipeline)
        self.zip_preprocessor(preprocessor)
        self.submit_leaderboard(predicted_labels)
        self.get_leaderboard(True)
        return pipeline, preprocessor


if __name__ == "__main__":
    datasets_name = "titanic"
    datasets_train, datasets_test, labels_train, labels_test, example_data, _ = ai.import_quickstart_data(datasets_name)
    myplayground = ModelPlayground(model_type="tabular",
                                   classification=True,
                                   private=False)
    myplayground.playground_url = 'https://die2py70qa.execute-api.us-east-2.amazonaws.com/prod/m'
    automl = autoMLTabular(datasets_name,
                           competition_playground=myplayground,
                           datasets_train=datasets_train,
                           datasets_test=datasets_test,
                           labels_train=labels_train,
                           labels_test=labels_test,
                           example_data=example_data)
    pipeline, preprocessor = automl.activate()
