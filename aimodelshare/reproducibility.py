import os
import sys
import json
import random
import tempfile
import pkg_resources

import numpy as np
import tensorflow as tf

from aimodelshare.aws import get_s3_iam_client, run_function_on_lambda, get_aws_client

def export_reproducibility_env(seed, directory, mode="gpu"):
  # Change the output into json.dumps
  # Argument single seed for all inputs & mode
  data = {
    "global_seed_code": [
      "os.environ['PYTHONHASHSEED'] = '{}'".format(seed),
      "random.seed({})".format(seed),
      "tf.random.set_seed({})".format(seed),
      "np.random.seed({})".format(seed),
    ]
  }

  # Ignore this part for now
  # Local seed codes are tensorflow code that are
  # not affected by the global seed and sometimes require us
  # to define the seed in the function call
  data["local_seed_code"] = [
    "train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed={}, image_size=(img_height, img_width), batch_size=batch_size)".format(seed),
    "val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed={}, image_size=(img_height, img_width), batch_size=batch_size)".format(seed),
  ]

  if mode == "gpu":
    data["gpu_cpu_parallelism_ops"] = [
      "os.environ['TF_DETERMINISTIC_OPS'] = '1'",
      "os.environ['TF_CUDNN_DETERMINISTIC'] = '1'"
    ]
  elif mode == "cpu":
    data["gpu_cpu_parallelism_ops"] = [
      "tf.config.threading.set_inter_op_parallelism_threads(1)"
    ]
  else:
    raise Exception("Error: unknown 'mode' value, expected 'gpu' or 'cpu'")

  installed_packages = pkg_resources.working_set
  installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
    for i in installed_packages])

  data["session_runtime_info"] = {
    "installed_packages": installed_packages_list,
    "python_version": sys.version,
  }

  with open(os.path.join(directory, "reproducibility.json"), "w") as fp:
    json.dump(data, fp)

  return print("Your reproducibility environment is now saved to 'reproducibility.json'")

def set_reproducibility_env(reproducibility_env):
  # Change the input into dict / json
  for global_code in reproducibility_env["global_seed_code"]:
    exec("%s" % (global_code))

  for parallelism_ops in reproducibility_env["gpu_cpu_parallelism_ops"]:
    exec("%s" % (parallelism_ops))

def import_reproducibility_env(reproducibility_env_file):
  with open(reproducibility_env_file) as json_file:
    reproducibility_env = json.load(json_file)
    set_reproducibility_env(reproducibility_env)

  print("Your reproducibility environment is successfully setup")

def import_reproducibility_env_from_model(apiurl):
    if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ, 
            "username" in os.environ, 
            "password" in os.environ]):
        pass
    else:
        return print("'Instantiate Model' unsuccessful. Please provide credentials with set_credentials().")

    aws_client = get_aws_client()
    reproducibility_env_filename = "/runtime_reproducibility.json"

    # Get bucket and model_id for user
    response, error = run_function_on_lambda(
        apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, bucket, model_id = json.loads(response.content.decode("utf-8"))

    try:
        resp_string = aws_client["client"].get_object(
            Bucket=bucket, Key=model_id + reproducibility_env_filename
        )

        reproducibility_env_string = resp_string['Body'].read()

    except Exception as err:
        print("This model was not deployed with reproducibility support")
        raise err

    # generate tempfile for onnx object 
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_file_name')

    # save onnx to temporary path
    with open(temp_path, "wb") as f:
        f.write(reproducibility_env_string)

    import_reproducibility_env(temp_path)
