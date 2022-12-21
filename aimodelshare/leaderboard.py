import json
import numpy as np
import pandas as pd
import os
import requests

import matplotlib.pyplot as plt

from collections import Counter
from aimodelshare.aws import run_function_on_lambda, get_aws_client
from aimodelshare.aimsonnx import _get_layer_names, layer_mapping


def get_leaderboard(apiurl, verbose=3, columns=None, submission_type="competition"):
    if all(["username" in os.environ, 
           "password" in os.environ]):
        pass
    else:
        return print("'get_leaderboard()' unsuccessful. Please provide credentials with set_credentials().")

    if columns == None: 
        columns = str(columns)

    post_dict = {"y_pred": [],
               "return_eval": "False",
               "return_y": "False",
               "inspect_model": "False",
               "version": "None", 
               "compare_models": "False",
               "version_list": "None",
               "get_leaderboard": "True",
               "submission_type": submission_type,
               "verbose": verbose,
               "columns": columns}
    
    headers = { 'Content-Type':'application/json', 'authorizationToken': os.environ.get("AWS_TOKEN"),} 

    apiurl_eval=apiurl[:-1]+"eval"

    leaderboard_json = requests.post(apiurl_eval,headers=headers,data=json.dumps(post_dict)) 

    leaderboard_pd = pd.DataFrame(json.loads(leaderboard_json.text))

    return leaderboard_pd



def stylize_leaderboard(leaderboard, naming_convention="keras"):

    leaderboard = consolidate_leaderboard(leaderboard, naming_convention=naming_convention)

    # Dropping some columns {{{
    drop_cols = ["timestamp"]
    leaderboard = leaderboard.drop(drop_cols, axis=1)

    #truncate model config info

    if "model_config" in leaderboard.columns:
        leaderboard.model_config = leaderboard.model_config.fillna('None')
        leaderboard.model_config = leaderboard.model_config.map(lambda x: x[0:30]+'...')

    # }}}

    # Setting default properties {{{
    default_props = {"text-align": "center"}

    board = leaderboard.style
    board = board.set_properties(**default_props)
    # }}}

    # infer task type 
    if 'accuracy' in leaderboard.columns.tolist(): 
        task_type = 'classification'
    else: 
        task_type = 'regression'

    # Setting percentage columns' properties {{{
    if task_type == "regression":
        percent_cols = ["mse", 'rmse', 'mae', "r2"]
        percent_colors = ["#f5f8d6", "#c778c8", "#ff4971", "#aadbaa"]

        percent_props = {"color": "#251e1b", "font-size": "12px"}

        for col, color in zip(percent_cols, percent_colors):
            board = board.bar(align="left", color=color, subset=col, vmin=0, vmax=leaderboard[col].max())

        board = board.set_properties(**percent_props, subset=percent_cols)
        board = board.format(lambda x: "{:.2f}".format(x), subset=percent_cols)


    else:
        percent_cols = ["accuracy", "f1_score", "precision", "recall"]
        percent_colors = ["#f5f8d6", "#c778c8", "#ff4971", "#aadbaa"]

        percent_props = {"color": "#251e1b", "font-size": "12px"}

        for col, color in zip(percent_cols, percent_colors):
            board = board.bar(align="left", color=color, subset=col, vmin=0, vmax=1)

        board = board.set_properties(**percent_props, subset=percent_cols)
        board = board.format(lambda x: "{:.2f}%".format(x * 100), subset=percent_cols)
    # }}}

    return board


def consolidate_leaderboard(data, naming_convention="keras"):

  for i in data: 

    i = i.replace('_layers', '')
    i = i.replace('_act', '')

    if 'maxpooling2d_layers' in data.columns and 'maxpool2d_layers' in data.columns:
      matched_cols = ['maxpooling2d_layers', 'maxpool2d_layers']
      sum_cols = data[matched_cols].sum(axis=1)
      data[matched_cols[1]] = sum_cols
      data = data.drop(matched_cols[0], axis=1)

    if naming_convention == 'keras':
      mapping = layer_mapping('torch_to_keras')
      mapping_inverse = layer_mapping('keras_to_torch')
    elif naming_convention == 'pytorch':
      mapping = layer_mapping('keras_to_torch')
      mapping_inverse = layer_mapping('torch_to_keras')

    mapping_lower = {i.lower(): mapping[i].lower() for i in mapping if i is not None and mapping[i] is not None}
    mapping_inverse_lower = {i.lower(): mapping_inverse[i].lower() for i in mapping_inverse if i is not None and mapping_inverse[i] is not None}

    try:

      if i in mapping_lower.keys():

        if not i == mapping_lower.get(i):
          
          matched_cols = [i+"_layers", mapping_lower.get(i)+"_layers"]

          sum_cols = data[matched_cols].sum(axis=1)

          data[matched_cols[1]] = sum_cols

          data = data.drop(matched_cols[0], axis=1)

    except:
      pass

    try:
      if i in mapping_inverse_lower.keys() and naming_convention == 'keras':

        if not i == mapping_inverse_lower.get(i):
          
          matched_cols = [i+"_layers", mapping_inverse_lower.get(i)+"_layers"]

          sum_cols = data[matched_cols].sum(axis=1)

          data[matched_cols[0]] = sum_cols

          data = data.drop(matched_cols[1], axis=1)

    except:
      pass
    
  return data



__all__ = [get_leaderboard,
    stylize_leaderboard]
