# data wrangling
import pandas as pd
import numpy as np

# ml frameworks
import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
import xgboost
import tensorflow as tf

# onnx modules
import onnx
import skl2onnx
from skl2onnx import convert_sklearn
import tf2onnx
from torch.onnx import export
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import importlib
import onnxmltools


# aims modules
from aimodelshare.aws import run_function_on_lambda

# os etc
import os
import ast
import tempfile
import json
import re
import pickle

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



def _misc_to_onnx(model, initial_types, transfer_learning=None,
                    deep_learning=None, task_type=None):

    # generate metadata dict 
    metadata = {}
    
    # placeholders, need to be generated elsewhere
    metadata['model_id'] = None
    metadata['data_id'] = None
    metadata['preprocessor_id'] = None
    
    # infer ml framework from function call
    if isinstance(model, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
        metadata['ml_framework'] = 'xgboost'
        onx = onnxmltools.convert.convert_xgboost(model, initial_types=initial_types)

    # also integrate lightGBM 
    
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



def _sklearn_to_onnx(model, initial_types, transfer_learning=None,
                    deep_learning=None, task_type=None):
    '''Extracts metadata from sklearn model object.'''
    
    # check whether this is a fitted sklearn model
    # sklearn.utils.validation.check_is_fitted(model)

    # deal with pipelines and parameter search 
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        model = model.best_estimator_

    if isinstance(model, sklearn.pipeline.Pipeline):
        model = model.steps[-1][1]

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

    # get weights for pretrained models 
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_file_name')

    with open(temp_path, 'wb') as f:
        pickle.dump(model, f)

    with open(temp_path, "rb") as f:
        pkl_str = f.read()

    metadata['model_weights'] = pkl_str
    
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


    # handle keras models in sklearn wrapper
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        model = model.best_estimator_

    if isinstance(model, sklearn.pipeline.Pipeline):
        model = model.steps[-1][1]

    sklearn_wrappers = (tf.keras.wrappers.scikit_learn.KerasClassifier,
                    tf.keras.wrappers.scikit_learn.KerasRegressor)

    if isinstance(model, sklearn_wrappers):
        model = model.model
    
    # convert to onnx
    #onx = convert_keras(model)
    # generate tempfile for onnx object 
    temp_dir = os.path.join(tempfile.gettempdir(), 'test')
    temp_dir = tempfile.gettempdir()

    model.save(temp_dir)
    output_path = os.path.join(temp_dir, 'temp.onnx')
    modelstringtest="python -m tf2onnx.convert --saved-model "+temp_dir+" --output "+output_path
    os.system(modelstringtest)
    onx = onnx.load(output_path)


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

    # get model weights from keras object 
    def to_list(x):
        return x.tolist()

    metadata['model_weights'] = str(list(map(to_list, model.get_weights())))

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
    frameworks = ['sklearn', 'keras', 'pytorch', 'xgboost']
    assert framework in frameworks, \
    'Please choose "sklearn", "keras", "pytorch", or "xgboost".'
    
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
    elif framework == 'xgboost':
        onnx = _misc_to_onnx(model, initial_types=initial_types, 
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
    #assert(isinstance(onnx_model, onnx.onnx_ml_pb2.ModelProto)), \
     #"Please pass a onnx model object."
    
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
    elif metadata_raw['ml_framework'] == 'sklearn' or metadata_raw['ml_framework'] == 'xgboost':
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
                          #'Activation':architecture['activations_sequence'],
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

def inspect_model(apiurl, aws_token, aws_client, version=None):

    onnx_model = _get_onnx_from_bucket(apiurl, aws_token, aws_client, version=version)

    meta_dict = _get_metadata(onnx_model)

    if meta_dict['ml_framework'] == 'keras':
        inspect_pd = _model_summary(meta_dict)
        
    elif meta_dict['ml_framework'] in ['sklearn', 'xgboost']:
        model_config = meta_dict["model_config"]
        model_config = ast.literal_eval(model_config)
        inspect_pd = pd.DataFrame({'param_name': model_config.keys(),
                                   'param_value': model_config.values()})
    
    return inspect_pd
    

def compare_models(apiurl, aws_token, aws_client, version_list=None, 
    by_model_type=None, best_model=None, verbose=3):
    
    if not isinstance(version_list, list):
        raise Exception("Argument 'version' must be a list.")
    
    models_dict = {}
    
    for i in version_list: 
        
        onnx_model = _get_onnx_from_bucket(apiurl, aws_token, aws_client, version=i)
        meta_dict = _get_metadata(onnx_model)
        
        models_dict[i] = meta_dict
        
    ml_framework_list = [models_dict[i]['ml_framework'] for i in models_dict]
    model_type_list = [models_dict[i]['model_type'] for i in models_dict]
    
    if not all(x==ml_framework_list[0] for x in ml_framework_list):
        raise Exception("Incongruent frameworks. Please compare models from the same ML frameworks.")
        
    if not all(x==model_type_list[0] for x in model_type_list):
        raise Exception("Incongruent model types. Please compare models of the same model type.")
    
    if ml_framework_list[0] == 'sklearn':
        
        model_type = model_type_list[0]
        model_class = model_from_string(model_type)
        default = model_class()
        default_config = default.get_params()
        
        comp_pd = pd.DataFrame({'param_name': default_config.keys(),
                           'param_value': default_config.values()})
        
        for i in version_list: 
            
            temp_pd = inspect_model(apiurl, aws_token, aws_client, version=i)
            comp_pd = pd.concat([comp_pd, temp_pd.drop(columns='param_name')], axis=1)
        
        comp_pd.columns = ['param_name', 'model_default'] + ["Model_"+str(i) for i in version_list]
        
        df_styled = comp_pd.style.apply(lambda x: ["background: tomato" if v != x.iloc[0] else "" for v in x], 
                                        axis = 1, subset=comp_pd.columns[1:])

        
    if ml_framework_list[0] == 'keras':

        comp_pd = pd.DataFrame()

        for i in version_list: 

            temp_pd = inspect_model(apiurl, aws_token, aws_client, version=i)

            temp_pd = temp_pd.iloc[:,0:verbose]

            temp_pd = temp_pd.add_prefix('Model_'+str(i)+'_')    
            comp_pd = pd.concat([comp_pd, temp_pd], axis=1)
        
        df_styled = comp_pd.style.apply(lambda x: ["background: tomato" if v == 'Dense' else "" for v in x], 
                                axis = 1)
        
        df_styled = df_styled.apply(lambda x: ["background: yellow" if v == 'Dropout' else "" for v in x], 
                                axis = 1)
        
        df_styled = df_styled.apply(lambda x: ["background: yellow" if v == 'Conv2D' else "" for v in x], 
                                axis = 1)
    
    return df_styled



def _get_onnx_from_bucket(apiurl, aws_token, aws_client, version=None):

    # generate name of onnx model in bucket
    onnx_model_name = "/onnx_model_v{version}.onnx".format(version = version)

    # Get bucket and model_id for user
    response, error = run_function_on_lambda(
        apiurl, aws_token, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, bucket, model_id = json.loads(response.content.decode("utf-8"))

    try:
        onnx_string = aws_client["client"].get_object(
            Bucket=bucket, Key=model_id + onnx_model_name
        )

        onnx_string = onnx_string['Body'].read()

    except Exception as err:
        raise err

    # generate tempfile for onnx object 
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_file_name')

    # save onnx to temporary path
    with open(temp_path, "wb") as f:
        f.write(onnx_string)

    # load onnx file from temporary path
    onx = onnx.load(temp_path)

    return onx


def instantiate_model(apiurl, aws_token, aws_client, version=None, trained=False):

    onnx_model = _get_onnx_from_bucket(apiurl, aws_token, aws_client, version=version)
    meta_dict = _get_metadata(onnx_model)

    # get model config 
    model_config = ast.literal_eval(meta_dict['model_config'])
    ml_framework = meta_dict['ml_framework']
    
    if ml_framework == 'sklearn':

        if trained == False:
            model_type = meta_dict['model_type']
            model_class = model_from_string(model_type)
            model = model_class(**model_config)

        if trained == True:
            
            model_pkl = meta_dict['model_weights']

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, 'temp_file_name')

            with open(temp_path, "wb") as f:
                f.write(model_pkl)

            with open(temp_path, 'rb') as f:
                model = pickle.load(f)


    if ml_framework == 'keras':

        if trained == False:
            model = tf.keras.Sequential().from_config(model_config)

        if trained == True: 
            model = tf.keras.Sequential().from_config(model_config)
            model_weights = ast.literal_eval(meta_dict['model_weights'])

            def to_array(x):
                return np.array(x, dtype="float32")

            model_weights = list(map(to_array, model_weights))

            model.set_weights(model_weights)

    return model


def _get_layer_names():

    activation_list = [i for i in dir(tf.keras.activations)]
    activation_list = [i for i in activation_list if callable(getattr(tf.keras.activations, i))]
    activation_list = [i for i in activation_list if  not i.startswith("_")]
    activation_list.remove('deserialize')
    activation_list.remove('get')
    activation_list.remove('linear')
    activation_list = activation_list+['ReLU', 'Softmax', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']


    layer_list = [i for i in dir(tf.keras.layers)]
    layer_list = [i for i in dir(tf.keras.layers) if callable(getattr(tf.keras.layers, i))]
    layer_list = [i for i in layer_list if not i.startswith("_")]
    layer_list = [i for i in layer_list if re.match('^[A-Z]', i)]
    layer_list = [i for i in layer_list if i.lower() not in [i.lower() for i in activation_list]]

    return layer_list, activation_list

def _get_sklearn_modules():
    
    import sklearn

    sklearn_modules = ['ensemble', 'gaussian_process', 'isotonic',
                       'linear_model', 'mixture', 'multiclass', 'naive_bayes', 
                       'neighbors', 'neural_network', 'svm', 'tree']

    models_modules_dict = {}

    for i in sklearn_modules:
        models_list = [j for j in dir(eval('sklearn.'+i)) if callable(getattr(eval('sklearn.'+i), j))]
        models_list = [j for j in models_list if re.match('^[A-Z]', j)]

        for k in models_list: 
            models_modules_dict[k] = 'sklearn.'+i
    
    return models_modules_dict


def model_from_string(model_type):
    models_modules_dict = _get_sklearn_modules()
    module = models_modules_dict[model_type]
    model_class = getattr(importlib.import_module(module), model_type)
    return model_class
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



def _misc_to_onnx(model, initial_types, transfer_learning=None,
                    deep_learning=None, task_type=None):

    # generate metadata dict 
    metadata = {}
    
    # placeholders, need to be generated elsewhere
    metadata['model_id'] = None
    metadata['data_id'] = None
    metadata['preprocessor_id'] = None
    
    # infer ml framework from function call
    if isinstance(model, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
        metadata['ml_framework'] = 'xgboost'
        onx = onnxmltools.convert.convert_xgboost(model, initial_types=initial_types)

    # also integrate lightGBM 
    
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



def _sklearn_to_onnx(model, initial_types, transfer_learning=None,
                    deep_learning=None, task_type=None):
    '''Extracts metadata from sklearn model object.'''
    
    # check whether this is a fitted sklearn model
    # sklearn.utils.validation.check_is_fitted(model)

    # deal with pipelines and parameter search 
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        model = model.best_estimator_

    if isinstance(model, sklearn.pipeline.Pipeline):
        model = model.steps[-1][1]

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
    frameworks = ['sklearn', 'keras', 'pytorch', 'xgboost']
    assert framework in frameworks, \
    'Please choose "sklearn", "keras", "pytorch", or "xgboost".'
    
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
    elif framework == 'xgboost':
        onnx = _misc_to_onnx(model, initial_types=initial_types, 
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
    #assert(isinstance(onnx_model, onnx.onnx_ml_pb2.ModelProto)), \
     #"Please pass a onnx model object."
    
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
    elif metadata_raw['ml_framework'] == 'sklearn' or metadata_raw['ml_framework'] == 'xgboost':
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

def instantiate_model(apiurl, aws_token, aws_client, version=None):

    # Get bucket and model_id for user
    response, error = run_function_on_lambda(
        apiurl, aws_token, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, bucket, model_id = json.loads(response.content.decode("utf-8"))

    # Get mastertable
    try:
        master_table = aws_client["client"].get_object(
            Bucket=bucket, Key=model_id + "/model_eval_data_mastertable.csv"
        )

        assert (
            master_table["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), "There was a problem in accessing the leaderboard file"

        master_table = pd.read_csv(master_table["Body"], sep="\t")

    except Exception as err:
        raise err

    # get model config 
    model_config = ast.literal_eval(master_table.model_config[master_table.version == version].iloc[0])

    if master_table.ml_framework[master_table.version == version].iloc[0] == 'sklearn':

        model_type = master_table.model_type[master_table.version == version].iloc[0]

        models_modules_dict = _get_sklearn_modules()
        module = models_modules_dict[model_type]
        model_class = getattr(importlib.import_module(module), model_type)
        model = model_class(**model_config)


    if master_table.ml_framework[master_table.version == version].iloc[0] == 'keras':

        model = keras.Sequential().from_config(model_config)

    return model 


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

def _get_sklearn_modules():
    
    import sklearn

    sklearn_modules = ['ensemble', 'gaussian_process', 'isotonic',
                       'linear_model', 'mixture', 'multiclass', 'naive_bayes', 
                       'neighbors', 'neural_network', 'svm', 'tree']

    models_modules_dict = {}

    for i in sklearn_modules:
        models_list = [j for j in dir(eval('sklearn.'+i)) if callable(getattr(eval('sklearn.'+i), j))]
        models_list = [j for j in models_list if re.match('^[A-Z]', j)]

        for k in models_list: 
            models_modules_dict[k] = 'sklearn.'+i
    
    return models_modules_dict
