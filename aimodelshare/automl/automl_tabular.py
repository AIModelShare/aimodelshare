import aimodelshare as ai
import dabl
from supervised.automl import AutoML

import pandas as pd
import os
import shutil

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes)
import xgboost
import lightgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

import argparse


def data_preprocessing(*datasets):
    datasets_processed = []
    for dataset in datasets:
        preprocessor = dabl.EasyPreprocessor()
        dataset = dabl.clean(dataset)
        preprocessor.fit(dataset)
        datasets_processed.append(preprocessor.transform(dataset))
    return datasets_processed


def label_preprocessing(*labels_pd):
    labels_processed = []
    for labels in labels_pd:
        preprocessor = dabl.EasyPreprocessor()
        labels.columns = ["label"]
        labels = preprocessor.fit_transform(labels.to_numpy().reshape(len(labels), 1))
        labels_processed.append(labels[:, 1].astype(int))
    return labels_processed


def data_preparation(*args, keyword=None):
    """
    args takes either one or two parameters, if one, a quick start dataset; if two, train + test dataset
    """
    cnt = 0
    dataset_train = dataset_test = None
    for dataset_name in args:
        assert isinstance(dataset_name, str)
        if dataset_name.split(".")[-1] == "csv":
            assert keyword is not None
            if cnt & 1 == 0:
                dataset_train = pd.read_csv(dataset_name)
            else:
                dataset_test = pd.read_csv(dataset_name)
        # else:
        #     quickstart_repository = None
        #     X_train, X_test, y_train, y_test, example_data, y_test_labels = ai.import_quickstart_data(dataset_name)
        cnt += 1
    dataset_train, dataset_test = data_preprocessing(dataset_train, dataset_test)
    y_train = dataset_train.loc[keyword]
    X_train = dataset_train.drop([keyword], axis=1)
    y_test = dataset_test.loc[keyword]
    X_test = dataset_test.drop([keyword], axis=1)
    return X_train, X_test, y_train, y_test


def model_preparation(automl_dir, mode="Perform"):
    """
    Create an AutoML model
    """
    # automl = AutoML(results_path=automl_dir, mode="Compete")
    if mode:
        automl = AutoML(results_path=automl_dir, mode=mode)
    else:
        automl = AutoML(results_path=automl_dir)
    return automl


def train(automl, X_train, y_train):
    """
    Model fit
    """
    automl.fit(X_train, y_train)
    return automl


def predict(automl, X_test):
    """
    AutoML predict
    """
    predictions = automl.predict(X_test)
    return predictions


def prediction_score(automl, X_test, y_test):
    """
    AutoML score
    """
    return automl.score(X_test, y_test)


def get_leaderboard(automl):
    """
    Get leaderboard of different models
    """
    print(automl.get_leaderboard())


def get_ensemble_model(automl):
    """
    Record ensemble model and return sklearn pipeline and preprocessor function
    """
    selected_models = automl.ensemble.selected_models
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
    return pipeline, preprocessor


def fit_pipeline(pipeline, preprocessor, X_train, y_train, X_test, y_test):
    """
    Fit pipeline for onnx save
    """
    X_train_transformed, y_train_transformed, _ = preprocessor.fit_and_transform(X_train, y_train)
    X_test_transformed, y_test_transformed, _ = preprocessor.fit_and_transform(X_test, y_test)
    pipeline.fit(X_train_transformed.to_numpy(), y_train_transformed)
    score_ = pipeline.score(X_test_transformed, y_test_transformed)
    return pipeline, score_


def reset(automl_dir):
    if os.path.exists(automl_dir):
        shutil.rmtree(automl_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="titanic", help="dataset names separated by comma")
    parser.add_argument("--keyword", type=str, default="", help="representation of gt labels in csv input")
    parser.add_argument("--automl_dir", type=str, default="automl", help="automl directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset.split(",")
    if len(dataset_name) == 1:
        X_train, X_test, y_train, y_test, example_data, y_test_labels = ai.import_quickstart_data(args.dataset)
    else:
        X_train, X_test, y_train, y_test = data_preparation(dataset_name, keyword=args.keyword)

    # X_train, X_test = data_preprocessing(X_train, X_test)
    if y_train.dtype != int or y_test.dtype != int:
        y_train, y_test = label_preprocessing(y_train, y_test)
    automl = model_preparation(automl_dir=args.automl_dir)
    automl = train(automl, X_train, y_train)
    score_ = prediction_score(automl, X_test, y_test)
    get_leaderboard(automl)
    pipeline, preprocessor = get_ensemble_model(automl)
    pipeline_fitted, score_pipeline = fit_pipeline(pipeline, preprocessor, X_train, y_train, X_test, y_test)