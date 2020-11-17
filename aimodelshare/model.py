import os
import json
import onnx
import numpy as np
import pandas as pd
import requests 
import json 

from datetime import datetime

from aimodelshare.aws import run_function_on_lambda
from aimodelshare.aws import get_token


def _get_file_list(client, bucket, model_id):
    #  Reading file list {{{
    try:
        objects = client["client"].list_objects(Bucket=bucket, Prefix=model_id + "/")
    except Exception as err:
        return None, err

    file_list = []
    if "Contents" in objects:
        for key in objects["Contents"]:
            file_list.append(key["Key"].split("/")[1])
    #  }}}

    return file_list, None


def _delete_s3_object(client, bucket, model_id, filename):
    deletionobject = client["resource"].Object(bucket, model_id + "/" + filename)
    deletionobject.delete()

def _get_predictionmodel_key(unique_model_id,file_extension):
    if file_extension==".pkl":
        file_key = unique_model_id + "/runtime_model" + file_extension
        versionfile_key = unique_model_id + "/predictionmodel_1" + file_extension
    else:
        file_key = unique_model_id + "/runtime_model" + file_extension
        versionfile_key = unique_model_id + "/predictionmodel_1" + file_extension
    return file_key,versionfile_key




def _upload_onnx_model(modelpath, client, bucket, model_id, model_version):
    # Check the model {{{
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"The model file at {modelpath} does not exist")

    file_name = os.path.basename(modelpath)
    file_name, file_ext = os.path.splitext(file_name)

    assert (
        file_ext == ".onnx"
    ), "modelshareai api only supports .onnx models at the moment"
    # }}}

    # Upload the model {{{
    try:
        client["client"].upload_file(
            modelpath, bucket, model_id + "/onnx_model_mostrecent.onnx"
        )
        client["client"].upload_file(
            modelpath,
            bucket,
            model_id + "/onnx_model_v" + str(model_version) + file_ext,
        )
    except Exception as err:
        return err
    # }}}

def _upload_preprocessor(preprocessor, client, bucket, model_id, model_version):

  try:

    
    # Check the preprocessor {{{
    if not os.path.exists(preprocessor):
        raise FileNotFoundError(
            f"The preprocessor file at {preprocessor} does not exist"
        )

    
    file_name = os.path.basename(preprocessor)
    file_name, file_ext = os.path.splitext(file_name)
    
    from zipfile import ZipFile
    dir_zip = preprocessor

    #zipObj = ZipFile(os.path.join("./preprocessor.zip"), 'a')
    #/Users/aishwarya/Downloads/aimodelshare-master
    client["client"].upload_file(dir_zip, bucket, model_id + "/preprocessor_v" + str(model_version)+ ".zip")
  except Exception as e:
    print(e)

def _extract_model_metadata(model, eval_metrics=None):
    # Getting the model metadata {{{
    graph = model.graph

    if eval_metrics is not None:
        metadata = eval_metrics
    else:
        metadata = dict()

    metadata["num_nodes"] = len(graph.node)
    metadata["depth"] = len(graph.initializer)
    metadata["num_params"] = sum(np.product(node.dims) for node in graph.initializer)

    # layers = ""
    # for node in graph.node:
    #     # consider type and get node attributes (??)
    #     layers += (
    #         node.op_type
    #         + "x".join(str(d.ints) for d in node.attribute if hasattr(d, 'ints'))
    #     )
    metadata["layers"] = "; ".join(node.op_type for node in graph.node)

    inputs = ""
    for inp in graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param != "":
                dims.append(d.dim_param)
            else:
                dims.append(str(d.dim_value))

        metadata["input_shape"] = dims
        inputs += f"{inp.name} ({'x'.join(dims)})"
    metadata["inputs"] = inputs

    outputs = ""
    for out in graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_param != "":
                dims.append(d.dim_param)
            else:
                dims.append(str(d.dim_value))

        outputs += f"{out.name} ({'x'.join(dims)})"
    metadata["outputs"] = outputs
    # }}}

    return metadata

def _update_leaderboard(
    modelpath, eval_metrics, client, token, bucket, model_id, model_version
):
    # Loading the model and its metadata {{{
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"The model file at {modelpath} does not exist")

    model = onnx.load(modelpath)
    metadata = _extract_model_metadata(model, eval_metrics)
    # }}}

    # Adding extra details to metadata {{{
    metadata["username"] = token["username"]
    metadata["timestamp"] = str(datetime.now())
    metadata["version"] = model_version
    # }}}

    # Read existing table {{{
    try:
        leaderboard = client["client"].get_object(
            Bucket=bucket, Key=model_id + "/model_eval_data_mastertable.csv"
        )
        leaderboard = pd.read_csv(leaderboard["Body"], sep="\t")
        columns = leaderboard.columns

    except client["client"].exceptions.NoSuchKey:
        # Create leaderboard if not exists
        # FIXME: Find a better way to get columns
        columns = list(metadata.keys())
        leaderboard = pd.DataFrame(columns=columns)

    except Exception as err:
        raise err
    # }}}

    # Update the leaderboard {{{
    metadata = {col: metadata.get(col, None) for col in columns}
    leaderboard = leaderboard.append(metadata, ignore_index=True, sort=False)

    leaderboard_csv = leaderboard.to_csv(index=False, sep="\t")

    try:
        s3_object = client["resource"].Object(
            bucket, model_id + "/model_eval_data_mastertable.csv"
        )
        s3_object.put(Body=leaderboard_csv)

    except Exception as err:
        return err
    # }}}


def submit_model(
    modelpath,
    apiurl,
    aws_token,
    aws_client,
    prediction_submission=None,
    preprocessor=None,
    sample_data=None,
):
    # Get bucket and model_id for user {{{
    response, error = run_function_on_lambda(
        apiurl, aws_token, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, bucket, model_id = json.loads(response.content.decode("utf-8"))
    # }}}

    # Get file list for current bucket {{{
    model_files, err = _get_file_list(aws_client, bucket, model_id)
    if err is not None:
        raise err
    # }}}

    # Delet recent model and/or preprocessor {{{
    recent_models = filter(lambda f: "mostrecent" in f, model_files)
    for model in recent_models:
        _delete_s3_object(aws_client, bucket, model_id, model)

    model_files = list(filter(lambda f: "mostrecent" not in f, model_files))
    # }}}

    # Get new model version {{{
    model_versions = [os.path.splitext(f)[0].split("_")[-1][1:] for f in model_files]

    model_versions = filter(lambda v: v.isnumeric(), model_versions)
    model_versions = list(map(int, model_versions))

    if model_versions:
        model_version = max(model_versions) + 1
    else:
        model_version = 1
    # }}}

    # Upload the preprocessor {{{
    if preprocessor is not None:
        err = _upload_preprocessor(
            preprocessor, aws_client, bucket, model_id, model_version
        )
        if err is not None:
            raise err
    # }}}

    # Upload the model {{{
    err = _upload_onnx_model(modelpath, aws_client, bucket, model_id, model_version)
    if err is not None:
        raise err
    # }}}
    


    token=get_token(aws_token['username'],aws_token['password'])
    headers = { 'Content-Type':'application/json', 'authorizationToken': token, } 
    apiurl_eval=apiurl[:-1]+"eval"
    prediction = requests.post(apiurl_eval,headers=headers,data=json.dumps(prediction_submission)) 

    eval_metrics=prediction.text




    # Upload model metrics and metadata {{{
    err = _update_leaderboard(
        modelpath, eval_metrics, aws_client, aws_token, bucket, model_id, model_version
    )
    if err is not None:
        raise err
    #  }}}

    # Update model version and sample data {{{
    data_types = None
    data_columns = None
    if sample_data is not None and isinstance(sample_data, pd.DataFrame):
        data_types = list(sample_data.dtypes.values.astype(str))
        data_columns = list(sample_data.columns)

    kwargs = {
        "delete": "FALSE",
        "versionupdateget": "FALSE",
        "versionupdateput": "TRUE",
        "version": model_version,
        "input_feature_dtypes": data_types,
        "input_feature_names": data_columns,
    }
    response, error = run_function_on_lambda(apiurl, aws_token, **kwargs)
    if error is not None:
        raise error
    # }}}

    return True


__all__ = [
    submit_model,
    _extract_model_metadata
]
