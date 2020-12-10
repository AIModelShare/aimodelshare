# data wrangling
import pandas as pd
import numpy as np

# ml frameworks
import sklearn
import keras 
import torch

# onnx modules
import onnx
import skl2onnx
from skl2onnx import convert_sklearn
import keras2onnx
from keras2onnx import convert_keras
from torch.onnx import export
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer


# os etc
import os
import ast
import tempfile
import json


def _extract_onnx_metadata(onnx_model, framework):
    '''Extracts model metadata from ONNX file.'''

    try:
        # get model graph
        graph = onnx_model.graph

        # initialize metadata dict
        metadata_onnx = {}

        # get input shape
        metadata_onnx["input_shape"] = graph.input[0].type.tensor_type.shape.dim[1].dim_value

        # get output shape
        metadata_onnx["output_shape"] = graph.output[0].type.tensor_type.shape.dim[1].dim_value 

        # get layers & activations 
        layer_nodes = ['MatMul']
        activation_nodes = ['Relu', 'Softmax']
    
        layers = []
        activations = []
        
        for op_id, op in enumerate(graph.node):

            if op.op_type in layer_nodes:
                layers.append(op.op_type)

            if op.op_type in activation_nodes:
                activations.append(op.op_type)
                
                
        # get shapes and parameters
        layers_shapes = []
        layers_n_params = []

        for layer_id, layer in enumerate(reversed(graph.initializer)):

            if(len(layer.dims)== 2):
                layers_shapes.append(layer.dims[1])

                n_params = layer.dims[0] * layer.dims[1] + layer.dims[1]
                layers_n_params.append(n_params)
                

        # get model architecture stats
        model_architecture = {'layers_number': len(layers),
                          'layers_sequence': layers,
                          'layers_summary': {i:layers.count(i) for i in set(layers)},
                          'layers_n_params': layers_n_params,
                          'layers_shapes': layers_shapes,
                          'activations_sequence': activations,
                          'activations_summary': {i:activations.count(i) for i in set(activations)}
                         }
        
        metadata_onnx["model_architecture"] = model_architecture
        
    except Exception as e:
        print(e)

    return metadata_onnx


def _sklearn_to_onnx(model, initial_types, transfer_learning=None,
                    deep_learning=None, task_type=None):
    '''Extracts metadata from sklearn model object.'''
    
    # check whether this is a fitted sklearn model
    sklearn.utils.validation.check_is_fitted(model)
    
    try:
        # convert to onnx
        onx = convert_sklearn(model, initial_types=initial_types)
                
        # generate metadata dict 
        metadata = {}
        
        # placeholders, need to be generated elsewhere
        metadata['model_id'] = None
        metadata['data_id'] = None
        metadata['preprocessor_id'] = None
        
        # infer ml framework from function call
        metadata['ml_framework'] = 'sklearn'
        
        # get model type from model object
        metadata['model_type'] = str(model)
        
        # get transfer learning bool from user input
        metadata['transfer_learning'] = transfer_learning 

        # get deep learning bool from user input
        metadata['deep_learning'] = deep_learning
        
        # get task type from user input
        metadata['task_type'] = task_type
        
        # placeholders, need to be inferred from data 
        metadata['target_distribution'] = None
        metadata['input_type'] = None
        metadata['input_shape'] = None
        metadata['input_dtypes'] = None       
        metadata['input_distribution'] = None
        
        # get model config dict from sklearn model object
        metadata['model_config'] = str(model.get_params())
        
        # get model state from sklearn model object
        metadata['model_state'] = None
        
        # TODO
        metadata['model_architecture'] = None
        
        # placeholder, needs evaluation engine
        metadata['eval_metrics'] = None  
        
        # add metadata from onnx object
        metadata['metadata_onnx'] = str(_extract_onnx_metadata(onx, framework='sklearn'))

        meta = onx.metadata_props.add()
        meta.key = 'model_metadata'
        meta.value = str(metadata)
    
    except Exception as e:
        print(e)
        
    return onx


def _keras_to_onnx(model, transfer_learning=None,
                  deep_learning=None, task_type=None):
    '''Extracts metadata from keras model object.'''

    # check whether this is a fitted keras model
    # isinstance...
    
    try:
        # convert to onnx
        onx = convert_keras(model)
        
        # generate metadata dict 
        metadata = {}
        
        # placeholders, need to be generated elsewhere
        metadata['model_id'] = None
        metadata['data_id'] = None
        metadata['preprocessor_id'] = None
        
        # infer ml framework from function call
        metadata['ml_framework'] = 'keras'
        
        # get model type from model object
        metadata['model_type'] =  str(model.__class__.__name__)
        
        # get transfer learning bool from user input
        metadata['transfer_learning'] = transfer_learning
        
        # get deep learning bool from user input
        metadata['deep_learning'] = deep_learning
        
        # get task type from user input
        metadata['task_type'] = task_type
        
        # placeholders, need to be inferred from data 
        metadata['target_distribution'] = None
        metadata['input_type'] = None
        metadata['input_shape'] = None
        metadata['input_dtypes'] = None       
        metadata['input_distribution'] = None
        
        # get model config dict from keras model object
        metadata['model_config'] = str(model.get_config())
        
        # get model state from pytorch model object
        metadata['model_state'] = None
        
        # extract model architecture metadata 
        layers = []
        layers_n_params = []
        layers_shapes = []
        activations = []
        
        for i in model.layers: 
            
            if i.__class__.__name__ != 'Activation':
                
                layers.append(i.__class__.__name__)
                layers_n_params.append(i.count_params())
                layers_shapes.append(i.output_shape)

            try:
                if i.activation.__name__ != 'linear':
                    activations.append(i.activation.__name__)
            except:
                activations.append(None)

        model_architecture = {'layers_number': len(layers),
                              'layers_sequence': layers,
                              'layers_summary': {i:layers.count(i) for i in set(layers)},
                              'layers_n_params': layers_n_params,
                              'layers_shapes': layers_shapes,
                              'activations_sequence': activations,
                              'activations_summary': {i:activations.count(i) for i in set(activations)}
                             }
                              
        metadata['model_architecture'] = str(model_architecture)
            
        # placeholder, needs evaluation engine
        metadata['eval_metrics'] = None
        
        # add metadata from onnx object
        metadata['metadata_onnx'] = str(_extract_onnx_metadata(onx, framework='keras'))
        
        # add metadata dict to onnx object
        meta = onx.metadata_props.add()
        meta.key = 'model_metadata'
        meta.value = str(metadata)
    
    except Exception as e:
        print(e)
        
    return onx


def _pytorch_to_onnx(model, model_input, transfer_learning=None, 
                    deep_learning=None, task_type=None):
    
    '''Extracts metadata from pytorch model object.'''

    # TODO check whether this is a fitted pytorch model
    # isinstance...
    
    try:
        # generate tempfile for onnx object 
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'temp_file_name')
        
        # generate onnx file and save it to temporary path
        export(model, model_input, temp_path)
        
        # load onnx file from temporary path
        onx = onnx.load(temp_path)
        
        # generate metadata dict 
        metadata = {}
        
        # placeholders, need to be generated elsewhere
        metadata['model_id'] = None
        metadata['data_id'] = None
        metadata['preprocessor_id'] = None
        
        # infer ml framework from function call
        metadata['ml_framework'] = 'pytorch'
        
        # get model type from model object
        metadata['model_type'] = str(model.__class__).split('.')[-1].split("'")[0] + '()'
        
        # get transfer learning bool from user input
        metadata['transfer_learning'] = transfer_learning 
        
        # get deep learning bool from user input
        metadata['deep_learning'] = deep_learning
        
        # get task type from user input 
        metadata['task_type'] = task_type
       
        # placeholders, need to be inferred from data 
        metadata['target_distribution'] = None
        metadata['input_type'] = None
        metadata['input_shape'] = None
        metadata['input_dtypes'] = None       
        metadata['input_distribution'] = None
        
        # get model config dict from pytorch model object
        metadata['model_config'] = str(model.__dict__)
        
        # get model state from pytorch model object
        metadata['model_state'] = str(model.state_dict())
        
        # extract model architecture metadata
        layers = [j._get_name() for i, j  in model.__dict__['_modules'].items()]
        
        layers_shapes = []
        for i, j  in model.__dict__['_modules'].items():
            try:
                layers_shapes.append(j.out_features)
            except:
                layers_shapes.append(0)
                
        n_params = []
        for i, j  in model.__dict__['_modules'].items():
            try:
                weights = np.prod(j._parameters['weight'].shape)
            except:
                weights = 0

            try:
                bias = np.prod(j._parameters['bias'].shape)
            except:
                bias = 0
                
            n_params.append(weights+bias)
        
        
        model_architecture = {'layers_number': len(layers),
                      'layers_sequence': layers,
                      'layers_summary': {i:layers.count(i) for i in set(layers)},
                      'layers_n_params': n_params,
                      'layers_shapes': layers_shapes,
                      #'activations_sequence': activations,
                      #'activations_summary': {i:activations.count(i) for i in set(activations)}
                     }
        
        metadata['model_architecture'] = str(model_architecture)
        
        # placeholder, needs evaluation engine
        metadata['eval_metrics'] = None
        
        # add metadata from onnx object
        metadata['metadata_onnx'] = str(_extract_onnx_metadata(onx, framework='pytorch'))
        
        # add metadata dict to onnx object
        meta = onx.metadata_props.add()
        meta.key = 'model_metadata'
        meta.value = str(metadata)
    
    except Exception as e:
        print(e)
        
    return onx


def model_to_onnx(model, framework, model_input=None, initial_types=None,
                  transfer_learning=None, deep_learning=None, task_type=None):
    
    '''Transforms sklearn, keras, or pytorch model object into ONNX format 
    and extracts model metadata dictionary. The model metadata dictionary 
    is saved in the ONNX file's metadata_props. 
    
    Parameters: 
    model: fitted sklearn, keras, or pytorch model object
    Specifies the model object that will be converted to ONNX. 
    
    framework: {"sklearn", "keras", "pytorch"}
    Specifies the machine learning framework of the model object.
    
    model_input: array_like, default=None
    Required when framework="pytorch".

    initial_types: initial types tuple, default=None
    Required when framework="sklearn".
     
    transfer_learning: bool, default=None
    Indicates whether transfer learning was used. 
    
    deep_learning: bool, default=None
    Indicates whether deep learning was used. 
    
    task_type: {"classification", "regression"}
    Indicates whether the model is a classification model or
    a regression model.
    
    Returns:
    ONNX object with model metadata saved in metadata props

    '''

    # assert that framework exists
    frameworks = ['sklearn', 'keras', 'pytorch']
    assert framework in frameworks, \
    'Please choose "sklearn", "keras", or "pytorch".'
    
    # assert model input type THIS IS A PLACEHOLDER
    if model_input != None:
        assert isinstance(model_input, (list, pd.core.series.Series, np.ndarray, torch.Tensor)), \
        'Please format model input as XYZ.'
    
    # assert initialtypes 
    if initial_types != None:
        assert isinstance(initial_types[0][1], (skl2onnx.common.data_types.FloatTensorType)), \
        'Please use FloatTensorType as initial types.'
        
    # assert transfer_learning
    if transfer_learning != None:
        assert isinstance(transfer_learning, (bool)), \
        'Please pass boolean to indicate whether transfer learning was used.'
        
    # assert deep_learning
    if deep_learning != None:
        assert isinstance(deep_learning, (bool)), \
        'Please pass boolean to indicate whether deep learning was used.'
        
    # assert task_type
    if task_type != None:
        assert task_type in ['classification', 'regression'], \
        'Please specify task type as "classification" or "regression".'
    

    if framework == 'sklearn':
        onnx = _sklearn_to_onnx(model, initial_types=initial_types, 
                                transfer_learning=transfer_learning, 
                                deep_learning=deep_learning, 
                                task_type=task_type)
        
    elif framework == 'keras':
        onnx = _keras_to_onnx(model, transfer_learning=transfer_learning, 
                              deep_learning=deep_learning, 
                              task_type=task_type)
        
    elif framework == 'pytorch':
        onnx = _pytorch_to_onnx(model, model_input=model_input,
                                transfer_learning=transfer_learning, 
                                deep_learning=deep_learning, 
                                task_type=task_type)
        
    return onnx



def _get_metadata(onnx_model):
    '''Fetches previously extracted model metadata from ONNX object
    and returns model metadata dict.'''
    
    # double check this 
    assert(isinstance(onnx_model, onnx.onnx_ml_pb2.ModelProto)), \
     "Please pass a onnx model object."
    
    try: 
        onnx_meta = onnx_model.metadata_props

        onnx_meta_dict = {}

        for i in onnx_meta:
            onnx_meta_dict[i.key] = i.value

        onnx_meta_dict = ast.literal_eval(onnx_meta_dict['model_metadata'])
        
        if onnx_meta_dict['model_config'] != None and \
        onnx_meta_dict['ml_framework'] != 'pytorch':
            onnx_meta_dict['model_config'] = ast.literal_eval(onnx_meta_dict['model_config'])
        
        if onnx_meta_dict['model_architecture'] != None:
            onnx_meta_dict['model_architecture'] = ast.literal_eval(onnx_meta_dict['model_architecture'])
            
        if onnx_meta_dict['metadata_onnx'] != None:
            onnx_meta_dict['metadata_onnx'] = ast.literal_eval(onnx_meta_dict['metadata_onnx'])
        
        onnx_meta_dict['model_image'] = onnx_to_image(onnx_model)

    except Exception as e:
    
        print(e)
        
        onnx_meta_dict = ast.literal_eval(onnx_meta_dict)
        
    return onnx_meta_dict



def _model_summary(meta_dict, from_onnx=False):
    '''Creates model summary table from model metadata dict.'''
    
    assert(isinstance(meta_dict, dict)), \
    "Please pass valid metadata dict."
    
    assert('model_architecture' in meta_dict.keys()), \
    "Please make sure model architecture data is included."
    
    
    if from_onnx == False:
        architecture = meta_dict["model_architecture"] 

        model_summary = pd.DataFrame({'Layer':architecture['layers_sequence'],
                                      'Activation':architecture['activations_sequence'],
                                      'Shape':architecture['layers_shapes'],
                                      'Params':architecture['layers_n_params']})
        
    if from_onnx == True:
        architecture = meta_dict['metadata_onnx']["model_architecture"]
        
        model_summary = pd.DataFrame({'Layer':architecture['layers_sequence'],
                              'Activation':architecture['activations_sequence'],
                              'Shape':architecture['layers_shapes'],
                              'Params':architecture['layers_n_params']})
        
    
    return model_summary



def onnx_to_image(model):
    '''Creates model graph image in pydot format.'''
    
    OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
    }
    
    pydot_graph = GetPydotGraph(
        model.graph,
        name=model.graph.name,
        rankdir='TB',
        node_producer=GetOpNodeProducer(
            embed_docstring=False,
            **OP_STYLE
        )
    )
    
    return pydot_graph



