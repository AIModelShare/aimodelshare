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
import re


def _extract_onnx_metadata(onnx_model, framework):
    '''Extracts model metadata from ONNX file.'''

    # get model graph
    graph = onnx_model.graph

    # initialize metadata dict
    metadata_onnx = {}

    # get input shape
    metadata_onnx["input_shape"] = graph.input[0].type.tensor_type.shape.dim[1].dim_value

    # get output shape
    metadata_onnx["output_shape"] = graph.output[0].type.tensor_type.shape.dim[1].dim_value 
    
    # get layers and activations NEW
    # match layers and nodes and initalizers in sinle object
    # insert here....

    # get layers & activations 
    layer_nodes = ['MatMul', 'Gemm', 'Conv'] #add MaxPool, Transpose, Flatten
    activation_nodes = ['Relu', 'Softmax']

    layers = []
    activations = []

    for op_id, op in enumerate(graph.node):

        if op.op_type in layer_nodes:
            layers.append(op.op_type)
            #if op.op_type == 'MaxPool':
                #activations.append(None)
            
        if op.op_type in activation_nodes:
            activations.append(op.op_type)


    # get shapes and parameters
    layers_shapes = []
    layers_n_params = []

    if framework == 'keras':
        initializer = list(reversed(graph.initializer))
        for layer_id, layer in enumerate(initializer):
            if(len(layer.dims)>= 2):
                layers_shapes.append(layer.dims[1])
                n_params = int(np.prod(layer.dims) + initializer[layer_id-1].dims)
                layers_n_params.append(n_params)
                

    elif framework == 'pytorch':
        initializer = graph.initializer
        for layer_id, layer in enumerate(initializer):
            if(len(layer.dims)>= 2):
                layers_shapes.append(layer.dims[0])
                n_params = int(np.prod(layer.dims) + initializer[layer_id-1].dims)
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

    return metadata_onnx


def _sklearn_to_onnx(model, initial_types, transfer_learning=None,
                    deep_learning=None, task_type=None):
    '''Extracts metadata from sklearn model object.'''
    
    # check whether this is a fitted sklearn model
    sklearn.utils.validation.check_is_fitted(model)
    

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
    model_type = str(model).split('(')[0]
    metadata['model_type'] = model_type
    
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

    # get model architecture    
    if model_type == 'MLPClassifier' or model_type == 'MLPRegressor':

        if model_type == 'MLPClassifier':
            loss = 'log-loss'
        if model_type == 'MLPRegressor':
            loss = 'squared-loss'

        n_params = []
        layer_dims = [model.n_features_in_] + model.hidden_layer_sizes + [model.n_outputs_]
        for i in range(len(layer_dims)-1):
            n_params.append(layer_dims[i]*layer_dims[i+1] + layer_dims[i+1])

        # insert data into model architecture dict 
        model_architecture = {'layers_number': len(model.hidden_layer_sizes),
                              'layers_sequence': ['Dense']*len(model.hidden_layer_sizes),
                              'layers_summary': {'Dense': len(model.hidden_layer_sizes)},
                              'layers_n_params': n_params, #double check 
                              'layers_shapes': model.hidden_layer_sizes,
                              'activations_sequence': [model.activation]*len(model.hidden_layer_sizes),
                              'activations_summary': {model.activation: len(model.hidden_layer_sizes)},
                              'loss': loss,
                              'optimizer': model.solver
                             }

        metadata['model_architecture'] = str(model_architecture)

    else:
        model_architecture = {}

        if hasattr(model, 'coef_'):
            model_architecture['layers_n_params'] = [len(model.coef_.flatten())]
        if hasattr(model, 'solver'):
            model_architecture['optimizer'] = model.solver

        metadata['model_architecture'] = str(model_architecture)

    # placeholder, needs evaluation engine
    metadata['eval_metrics'] = None  
    
    # add metadata from onnx object
    # metadata['metadata_onnx'] = str(_extract_onnx_metadata(onx, framework='sklearn'))
    metadata['metadata_onnx'] = None

    meta = onx.metadata_props.add()
    meta.key = 'model_metadata'
    meta.value = str(metadata)

    return onx



def _keras_to_onnx(model, transfer_learning=None,
                  deep_learning=None, task_type=None):
    '''Extracts metadata from keras model object.'''

    # check whether this is a fitted keras model
    # isinstance...
    
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
    
    # get list of current layer types 
    layer_list, activation_list = _get_layer_names()

    # extract model architecture metadata 
    layers = []
    layers_n_params = []
    layers_shapes = []
    activations = []
    
    for i in model.layers: 
        
        # get layer names 
        if i.__class__.__name__ in layer_list:
            layers.append(i.__class__.__name__)
            layers_n_params.append(i.count_params())
            layers_shapes.append(i.output_shape)
        
        # get activation names
        if i.__class__.__name__ in activation_list: 
            activations.append(i.__class__.__name__.lower())
        if hasattr(i, 'activation') and i.activation.__name__ in activation_list:
            activations.append(i.activation.__name__)
            
    if hasattr(model, 'loss'):
        loss = model.loss.__class__.__name__
    else:
        loss = None
        
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer.__class__.__name__
    else:
        optimizer = None 
            
    
    # insert data into model architecture dict 
    model_architecture = {'layers_number': len(layers),
                          'layers_sequence': layers,
                          'layers_summary': {i:layers.count(i) for i in set(layers)},
                          'layers_n_params': layers_n_params,
                          'layers_shapes': layers_shapes,
                          'activations_sequence': activations,
                          'activations_summary': {i:activations.count(i) for i in set(activations)},
                          'loss':loss,
                          'optimizer': optimizer
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

    return onx


def _pytorch_to_onnx(model, model_input, transfer_learning=None, 
                    deep_learning=None, task_type=None):
    
    '''Extracts metadata from pytorch model object.'''

    # TODO check whether this is a fitted pytorch model
    # isinstance...

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
    activation_names = ['ReLU', 'Softmax']   #placeholder
    layer_names = []

    layers = []
    layers_shapes = []
    n_params = []


    for i, j  in model.__dict__['_modules'].items():

        if j._get_name() not in activation_names:

            layers.append(j._get_name())
            layers_shapes.append(j.out_features)
            weights = np.prod(j._parameters['weight'].shape)
            bias = np.prod(j._parameters['bias'].shape)
            n_params.append(weights+bias)


    activations = []
    for i, j in model.__dict__['_modules'].items():
        if str(j).split('(')[0] in activation_names:
            activations.append(str(j).split('(')[0])


    model_architecture = {'layers_number': len(layers),
                  'layers_sequence': layers,
                  'layers_summary': {i:layers.count(i) for i in set(layers)},
                  'layers_n_params': n_params,
                  'layers_shapes': layers_shapes,
                  'activations_sequence': activations,
                  'activations_summary': {i:activations.count(i) for i in set(activations)}
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

        onnx_meta_dict = {'model_metadata': ''}

        for i in onnx_meta:
            onnx_meta_dict[i.key] = i.value

        onnx_meta_dict = ast.literal_eval(onnx_meta_dict['model_metadata'])
        
        #if onnx_meta_dict['model_config'] != None and \
        #onnx_meta_dict['ml_framework'] != 'pytorch':
        #    onnx_meta_dict['model_config'] = ast.literal_eval(onnx_meta_dict['model_config'])
        
        if onnx_meta_dict['model_architecture'] != None:
            onnx_meta_dict['model_architecture'] = ast.literal_eval(onnx_meta_dict['model_architecture'])
            
        if onnx_meta_dict['metadata_onnx'] != None:
            onnx_meta_dict['metadata_onnx'] = ast.literal_eval(onnx_meta_dict['metadata_onnx'])
        
        onnx_meta_dict['model_image'] = onnx_to_image(onnx_model)

    except Exception as e:
    
        print(e)
        
        onnx_meta_dict = ast.literal_eval(onnx_meta_dict)
        
    return onnx_meta_dict



def _get_leaderboard_data(onnx_model, eval_metrics=None):
    
    if eval_metrics is not None:
        metadata = eval_metrics
    else:
        metadata = dict()
        
    metadata_raw = _get_metadata(onnx_model)

    # get list of current layer types 
    layer_list, activation_list = _get_layer_names()

    # get general model info
    metadata['ml_framework'] = metadata_raw['ml_framework']
    metadata['transfer_learning'] = metadata_raw['transfer_learning']
    metadata['deep_learning'] = metadata_raw['deep_learning']
    metadata['model_type'] = metadata_raw['model_type']


    # get neural network metrics
    if metadata_raw['ml_framework'] == 'keras' or metadata_raw['model_type'] in ['MLPClassifier', 'MLPRegressor']:
        metadata['depth'] = metadata_raw['model_architecture']['layers_number']
        metadata['num_params'] = sum(metadata_raw['model_architecture']['layers_n_params'])

        for i in layer_list:
            if i in metadata_raw['model_architecture']['layers_summary']:
                metadata[i.lower()+'_layers'] = metadata_raw['model_architecture']['layers_summary'][i]
            else:
                metadata[i.lower()+'_layers'] = 0

        for i in activation_list:
            if i in metadata_raw['model_architecture']['activations_summary']:
                if i.lower()+'_act' in metadata:
                    metadata[i.lower()+'_act'] += metadata_raw['model_architecture']['activations_summary'][i]
                else:    
                    metadata[i.lower()+'_act'] = metadata_raw['model_architecture']['activations_summary'][i]
            else:
                if i.lower()+'_act' not in metadata:
                    metadata[i.lower()+'_act'] = 0
                
        metadata['loss'] = metadata_raw['model_architecture']['loss']
        metadata['optimizer'] = metadata_raw['model_architecture']["optimizer"]
        metadata['model_config'] = metadata_raw['model_config']

    # get sklearn model metrics
    elif metadata_raw['ml_framework'] == 'sklearn':
        metadata['depth'] = 0

        try:
            metadata['num_params'] = sum(metadata_raw['model_architecture']['layers_n_params'])
        except:
            metadata['num_params'] = 0

        for i in layer_list:
            metadata[i.lower()+'_layers'] = 0

        for i in activation_list:
            metadata[i.lower()+'_act'] = 0

        metadata['loss'] = None

        try:
            metadata['optimizer'] = metadata_raw['model_architecture']['optimizer']
        except:
            metadata['optimizer'] = None

        try:
            metadata['model_config'] = metadata_raw['model_config']
        except:
            metadata['model_config'] = None
    
    return metadata
    


def _model_summary(meta_dict, from_onnx=False):
    '''Creates model summary table from model metadata dict.'''
    
    assert(isinstance(meta_dict, dict)), \
    "Please pass valid metadata dict."
    
    assert('model_architecture' in meta_dict.keys()), \
    "Please make sure model architecture data is included."

    if from_onnx == True:
        architecture = meta_dict['metadata_onnx']["model_architecture"]
    else:
        architecture = meta_dict["model_architecture"] 
       
        
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


def _get_layer_names():

    activation_list = [i for i in dir(keras.activations)]
    activation_list = [i for i in activation_list if callable(getattr(keras.activations, i))]
    activation_list = [i for i in activation_list if  not i.startswith("_")]
    activation_list.remove('deserialize')
    activation_list.remove('get')
    activation_list.remove('linear')
    activation_list = activation_list+['ReLU', 'Softmax', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']


    layer_list = [i for i in dir(keras.layers)]
    layer_list = [i for i in dir(keras.layers) if callable(getattr(keras.layers, i))]
    layer_list = [i for i in layer_list if not i.startswith("_")]
    layer_list = [i for i in layer_list if re.match('^[A-Z]', i)]
    layer_list = [i for i in layer_list if i.lower() not in [i.lower() for i in activation_list]]

    return layer_list, activation_list



