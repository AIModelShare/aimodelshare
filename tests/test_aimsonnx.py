from aimodelshare.aimsonnx import _get_layer_names
from aimodelshare.aimsonnx import _get_layer_names_pytorch
from aimodelshare.aimsonnx import _get_sklearn_modules
from aimodelshare.aimsonnx import model_from_string
from aimodelshare.aimsonnx import _get_pyspark_modules
from aimodelshare.aimsonnx import pyspark_model_from_string
from aimodelshare.aimsonnx import layer_mapping
from aimodelshare.aimsonnx import _sklearn_to_onnx
from aimodelshare.aimsonnx import _pyspark_to_onnx
from aimodelshare.aimsonnx import _keras_to_onnx
from aimodelshare.aimsonnx import _pytorch_to_onnx
from aimodelshare.aimsonnx import _misc_to_onnx
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import onnx
from xgboost import XGBClassifier
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
from keras.models import Sequential
from torch import nn

def test_sklearn_to_onnx():


    model = LogisticRegression(C=10, penalty='l1', solver='liblinear')
    onnx_model = _sklearn_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)

    model = MLPClassifier()
    onnx_model = _sklearn_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)


def test_misc_to_onnx():

    model = XGBClassifier()
    onnx_model = _misc_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)


def test_pyspark_to_onnx():

    model = RandomForestClassifier()
    onnx_model = _pyspark_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)

    model = MultilayerPerceptronClassifier()
    onnx_model = _pyspark_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)

def test_keras_to_onnx():

    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    onnx_model = _keras_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)


def test_pytorch_to_onnx():

    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_out),
                          nn.Sigmoid())

    onnx_model = _pytorch_to_onnx(model)
    assert isinstance(onnx_model, onnx.ModelProto)


def test_get_layer_names():

    layers = _get_layer_names()

    assert isinstance(layers, tuple)


def test_get_layer_names_pytorch():

        layers = _get_layer_names_pytorch()

        assert isinstance(layers, tuple)


def test_get_sklearn_modules():

        modules = _get_sklearn_modules()

        assert isinstance(modules, dict)

def test_model_from_string():

    model_class = model_from_string("RandomForestClassifier")

    assert model_class.__name__ == "RandomForestClassifier"


def test_get_pyspark_modules():

    modules = _get_pyspark_modules()

    assert isinstance(modules, dict)


def test_pyspark_model_from_string():

    model_class = pyspark_model_from_string("RandomForestClassifier")

    assert model_class.__name__ == "RandomForestClassifier"


def test_layer_mapping():

    layer_map = layer_mapping(direction="torch_to_keras")
    assert isinstance(layer_map, dict)

    layer_map = layer_mapping(direction="keras_to_torch")
    assert isinstance(layer_map, dict)

    layer_map = layer_mapping(direction="torch_to_keras", activation=True)
    assert isinstance(layer_map, dict)

    layer_map = layer_mapping(direction="keras_to_torch", activation=True)
    assert isinstance(layer_map, dict)

