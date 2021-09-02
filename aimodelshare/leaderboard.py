import json
import numpy as np
import pandas as pd
import os
import requests

from aimodelshare.aws import run_function_on_lambda, get_aws_client
from aimodelshare.aimsonnx import _get_layer_names


def get_leaderboard_aws(apiurl, category="classification", verbose=3, columns=None):
    # Confirm that creds are loaded, print warning if not
    if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ,
           "username" in os.environ, 
           "password" in os.environ]):
        pass
    else:
        return print("Get Leaderboard unsuccessful. Please provide credentials with set_credentials().")

    aws_client=get_aws_client(aws_key=os.environ.get('AWS_ACCESS_KEY_ID'), 
                              aws_secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                              aws_region=os.environ.get('AWS_REGION'))
    
    # Get bucket and model_id for user {{{
    response, error = run_function_on_lambda(
        apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, bucket, model_id = json.loads(response.content.decode("utf-8"))
    # }}}

    # Get leaderboard {{{
    try:
        leaderboard = aws_client["client"].get_object(
            Bucket=bucket, Key=model_id + "/model_eval_data_mastertable.csv"
        )
        assert (
            leaderboard["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), "There was a problem in accessing the leaderboard file"

        leaderboard = pd.read_csv(leaderboard["Body"], sep="\t")

        clf =["accuracy", "f1_score", "precision", "recall"]
        reg = ['mse', 'rmse', 'mae', 'r2']
        other = ['timestamp']

        if columns:
        	leaderboard = leaderboard.filter(clf+reg+columns+other)


        if category == "classification":
            leaderboard_eval_metrics = leaderboard[clf]
        else:
            leaderboard_eval_metrics = leaderboard[reg]

        leaderboard_model_meta = leaderboard.drop(clf+reg, axis=1).replace(0,np.nan).dropna(axis=1,how="all")

        leaderboard = pd.concat([leaderboard_eval_metrics, leaderboard_model_meta], axis=1, ignore_index=False)

        if verbose == 1:
        	leaderboard = leaderboard.filter(regex=("^(?!.*(_layers|_act))"))
        elif verbose == 2:
        	leaderboard = leaderboard.filter(regex=("^(?!.*_act)"))

    except Exception as err:
        raise err
    # }}}

    # Specifying problem wise columns {{{
    if category == "classification":
        sort_cols = ["accuracy", "f1_score", "precision", "recall"]
        #leaderboard = leaderboard.drop(columns = ['mse', 'rmse', 'mae', 'r2'])

    else:
        sort_cols = ["-mae", "r2"]
        #leaderboard = leaderboard.drop(columns = ["accuracy", "f1_score", "precision", "recall"])

    # }}}

    # Sorting leaderboard {{{
    ranks = []
    for col in sort_cols:
        ascending = False
        if col[0] == "-":
            col = col[1:]
            ascending = True

        ranks.append(leaderboard[col].rank(method="dense", ascending=ascending))

    ranks = np.mean(ranks, axis=0)
    order = np.argsort(ranks)

    leaderboard = leaderboard.loc[order].reset_index().drop("index", axis=1)
    # }}}

    return leaderboard


def get_leaderboard_lambda(apiurl, category="classification", verbose=3, columns=None):
    if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ, 
           "username" in os.environ, 
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
               "category": category,
               "verbose": verbose,
               "columns": columns}
    
    headers = { 'Content-Type':'application/json', 'authorizationToken': os.environ.get("AWS_TOKEN"),} 

    apiurl_eval=apiurl[:-1]+"eval"

    leaderboard_json = requests.post(apiurl_eval,headers=headers,data=json.dumps(post_dict)) 

    leaderboard_pd = pd.DataFrame(json.loads(leaderboard_json.text))

    return leaderboard_pd


def get_leaderboard(apiurl, category="classification", verbose=3, columns=None):

    if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ, 
           "username" in os.environ, 
           "password" in os.environ]):
        pass
    else:
        return print("'get_leaderboard()' unsuccessful. Please provide credentials with set_credentials().")

    
    try: 
        leaderboard_pd = get_leaderboard_lambda(apiurl, category, verbose, columns)
    except: 
        leaderboard_pd = get_leaderboard_aws(apiurl, category, verbose, columns)
    
    return leaderboard_pd




def stylize_leaderboard(leaderboard, category="classficiation"):
    # Dropping some columns {{{
    drop_cols = ["timestamp"]
    leaderboard = leaderboard.drop(drop_cols, axis=1)

    #truncate model config info
    leaderboard.model_config = leaderboard.model_config.map(lambda x: x[0:30]+'...')

    # }}}

    # Setting default properties {{{
    default_props = {"text-align": "center"}

    board = leaderboard.style
    board = board.set_properties(**default_props)
    # }}}

    # Setting percentage columns' properties {{{
    if category == "regression":
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


__all__ = [
    get_leaderboard,
    stylize_leaderboard,
]
