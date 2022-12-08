import aimodelshare as ai
import dabl
from supervised.automl import AutoML

import os
import shutil
import numpy as np
import pandas as pd

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
from onnxmltools import convert_sklearn
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

import argparse

from typing import Any, Sequence


class autoMLTabular:
    def __init__(self, dataset_names: Sequence, keyword="label", automl_dir="automl", mode=None, **kwargs):
        self.datasets_train = None
        self.labels_train = None
        self.datasets_test = None
        self.labels_test = None
        self.keyword = keyword  # This is the name for the ground truth column

        self.automl_dir = automl_dir
        self.automl = AutoML(results_path=automl_dir, mode=mode if mode else "Perform")

        self.dataset_names = dataset_names
        if isinstance(self.dataset_names, list):
            self.datasets_train, self.datasets_test = pd.read_csv(self.dataset_names[0]), pd.read_csv(self.dataset_names[1])
            self.labels_train = self.datasets_train.loc[keyword]
            self.datasets_train = self.datasets_train.drop([keyword], axis=1)
            self.labels_test = self.datasets_test.loc[keyword]
            self.datasets_test = self.datasets_test.drop([keyword], axis=1)
        else:
            # Now it is not working
            try:
                self.datasets_train, self.datasets_test, self.labels_train, self.labels_test, example_data, y_test_labels = ai.import_quickstart_data(dataset_names)
            except:
                self.datasets_train, self.datasets_test, self.labels_train, self.labels_test = kwargs["datasets_train"], kwargs["datasets_test"], kwargs["labels_train"], kwargs["labels_test"]
        self.preprocessor = dabl.EasyPreprocessor()  # This is for fast preprocessing

        self._fast_preprocessing_fit()
        self._fast_preprocessing_transform()

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
        self.labels_train_processed = np.empty((len(self.labels_train), ))
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
        self.labels_test_processed = np.empty((len(self.labels_test), ))
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

    def get_leaderboard(self):
        """
        Get leaderboard of different models
        """
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
                    raise NotImplementedError
                assert hasattr(model, "predict")  # and hasattr(preprocessor, "fit") and hasattr(preprocessor, "transform")
                pipeline = new_model
            else:
                pipeline = model
            model_pipelines.append(pipeline)

        estimators = [("model_" + str(i), pipeline) for i, pipeline in enumerate(model_pipelines)]

        stacked_classifier = StackingClassifier(estimators=estimators,
                                                final_estimator=LogisticRegression())

        pipeline = make_pipeline(stacked_classifier)
        self._fit_pipeline(pipeline)
        return pipeline, preprocessor

    def _fit_pipeline(self, pipeline):
        """
        Fit pipeline for onnx save
        """
        pipeline.fit(self.datasets_train, self.labels_train_processed)
        score_ = pipeline.score(self.datasets_test, self.labels_test_processed)
        return pipeline, score_

    def reset(self):
        if os.path.exists(self.automl_dir):
            shutil.rmtree(self.automl_dir)

    def to_onnx(self, pipeline_fitted):
        initial_types = [('float_input', FloatTensorType([None, self.datasets_train.shape[1]]))]
        onnx_model = convert_sklearn(
            pipeline_fitted, 'pipeline_ensemble',
            initial_types,
            target_opset={'': 12, 'ai.onnx.ml': 2})
        return onnx_model


def automl_tabular(datasets_name, automl_dir="automl", keyword="label", mode=None, **kwargs):
    modulize = autoMLTabular(datasets_name, automl_dir, keyword, mode=mode, **kwargs)
    modulize.reset()
    modulize.train()
    print("Prediction score", modulize.prediction_score())
    modulize.get_leaderboard()
    pipeline, preprocessor = modulize.get_ensemble_model()
    # onnx_model = modulize.to_onnx(pipeline)
    return pipeline, preprocessor


if __name__ == "__main__":
    datasets_name = "titanic"
    datasets_train, datasets_test, labels_train, labels_test, _, _ = ai.import_quickstart_data(datasets_name)
    pipeline, preprocessor = automl_tabular(datasets_name,
                                            datasets_train=datasets_train,
                                            datasets_test=datasets_test,
                                            labels_train=labels_train,
                                            labels_test=labels_test)
