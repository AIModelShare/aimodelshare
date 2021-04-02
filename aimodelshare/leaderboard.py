import json
import numpy as np
import pandas as pd

from aimodelshare.aws import run_function_on_lambda
from aimodelshare.aimsonnx import _get_layer_names


def get_leaderboard(apiurl, category="classification", verbose=3, columns=None):
    # Confirm that creds are loaded, print warning if not
    if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ,
           "username" in os.environ, 
           "password" in os.environ]):
        pass
    else:
        return print("Get Leaderboard unsuccessful. Please provide credentials with set_credentials().")

    aws_client=ai.aws.get_aws_client(aws_key=os.environ.get('AWS_ACCESS_KEY_ID'), 
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

        if columns:
        	clf =["accuracy", "f1_score", "precision", "recall"]
        	reg = ['mse', 'rmse', 'mae', 'r2']
        	other = ['timestamp']
        	leaderboard = leaderboard.filter(clf+reg+columns+other)

        leaderboard = leaderboard.replace(0,np.nan).dropna(axis=1,how="all")

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
        percent_cols = ["r2"]
        percent_colors = ["#f5f8d6"]
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
