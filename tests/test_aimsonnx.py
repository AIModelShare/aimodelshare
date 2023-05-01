from aimodelshare.aimsonnx import _get_layer_names
from aimodelshare.aimsonnx import _get_layer_names_pytorch
from aimodelshare.aimsonnx import _get_sklearn_modules
from aimodelshare.aimsonnx import model_from_string
from aimodelshare.aimsonnx import _get_pyspark_modules
from aimodelshare.aimsonnx import pyspark_model_from_string
from aimodelshare.aimsonnx import layer_mapping


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