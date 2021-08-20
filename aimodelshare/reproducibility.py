import os
import sys
import json
import random
import tempfile
import pkg_resources

import numpy as np
import tensorflow as tf

from aimodelshare.aws import get_s3_iam_client, run_function_on_lambda

def capture_reproducibility_env(seed, mode="gpu"):
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

  return json.dumps(data)

def set_reproducibility_env(reproducibility_env):
  # Change the input into dict / json
  # Remove any unnecessary output
  reproducibility_env_dict = json.loads(reproducibility_env)
  for global_code in reproducibility_env_dict["global_seed_code"]:
    exec("%s" % (global_code))

  for parallelism_ops in reproducibility_env_dict["gpu_cpu_parallelism_ops"]:
    exec("%s" % (parallelism_ops))

# User needs to call aws.set_credentials & has submitted a model
# before submitting the reproducibility env
# Would be convenient if we integrate it with model submission right away
def submit_reproducibility_env(reproducibility_env, api_url):
    s3, iam, region = get_s3_iam_client(
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
        os.environ.get("AWS_REGION"),
    )

    # Get bucket and model_id subfolder for user based on apiurl {{{
    response, error = run_function_on_lambda(
        api_url, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, bucket, model_id = json.loads(response.content.decode("utf-8"))
    # }}}

    temp_dir = tempfile.mkdtemp()
    # Check the reproducibility_env {{{
    if "global_seed_code" not in reproducibility_env \
            or "local_seed_code" not in reproducibility_env \
            or "gpu_cpu_parallelism_ops" not in reproducibility_env \
            or "session_runtime_info" not in reproducibility_env:
        raise Exception("reproducibility_env is not complete")

    temp_json = os.path.join(temp_dir, "reproducibility_env.json")
    with open(temp_json, "w") as fp:
        json.dump(reproducibility_env, fp)
    # }}}

    # Upload the json {{{
    try:
        s3["client"].upload_file(
            temp_json, bucket, model_id + "/reproducibility_env/reproducibility_env.json"
        )
    except Exception as err:
        return err
    # }}}
