import json
import numpy as np
import pandas as pd

from aimodelshare.aws import run_function_on_lambda


def get_leaderboard(apiurl, aws_token, aws_client, category="classification"):
    # Get bucket and model_id for user {{{
    response, error = run_function_on_lambda(
        apiurl, aws_token, **{"delete": "FALSE", "versionupdateget": "TRUE"}
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

    except Exception as err:
        raise err
    # }}}

    # Specifying problem wise columns {{{
    if category == "classification":
        sort_cols = ["accuracy", "f1_score", "precision", "recall"]
    else:
        sort_cols = ["-mae", "r2"]
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
    drop_cols = ["layers", "timestamp"]
    leaderboard = leaderboard.drop(drop_cols, axis=1)
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
