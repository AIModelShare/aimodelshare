import boto3
import botocore
import os
import requests
import uuid
import json
import math
import time
import datetime
from aimodelshare.tools import extract_varnames_fromtrainingdata, _get_extension_from_filepath
from aimodelshare.aws import get_s3_iam_client
from aimodelshare.bucketpolicy import _custom_upload_policy
from aimodelshare.exceptions import AuthorizationError, AWSAccessError, AWSUploadError
from aimodelshare.api import create_prediction_api
from aimodelshare.preprocessormodules import upload_preprocessor
from aimodelshare.model import _get_predictionmodel_key
def take_user_info_and_generate_api(model_filepath,my_credentials,model_type, categorical, private, labels,preprocessor_filepath="default", preprocessor="default"):
  #unpack user credentials
  username = my_credentials["username"]
  AI_MODELSHARE_AccessKeyId = my_credentials["AI_MODELSHARE_AccessKeyId"]
  AI_MODELSHARE_SecretAccessKey = my_credentials["AI_MODELSHARE_SecretAccessKey"]
  iamusername = my_credentials["iamusername"]
  returned_jwt_token = my_credentials["returned_jwt_token"]
  aws_key = my_credentials["aws_key"]
  aws_password = my_credentials["aws_password"]
  region = my_credentials["region"]
  bucket_name = my_credentials["bucket_name"]
  now = datetime.datetime.now()
  s3, iam, region = get_s3_iam_client(aws_key, aws_password, region)
  s3["client"].create_bucket(
            ACL='private',
            Bucket=bucket_name)
  #model upload
  Filepath = model_filepath
  #tab_imports ='./tabular_imports.pkl'
  #img_imports ='./image_imports.pkl'
  file_extension = _get_extension_from_filepath(Filepath)
  unique_model_id = str(uuid.uuid1().hex)
  file_key, versionfile_key = _get_predictionmodel_key(unique_model_id,file_extension)
  try:
    s3["client"].upload_file(Filepath, bucket_name,  file_key)
    s3["client"].upload_file(Filepath, bucket_name,  versionfile_key)

    #preprocessor upload
    #s3["client"].upload_file(tab_imports, bucket_name,  'tabular_imports.pkl')
    #s3["client"].upload_file(img_imports, bucket_name,  'image_imports.pkl')
    #preprocessor upload
    
    #ADD model/Preprocessor VERSION
    response = upload_preprocessor(preprocessor_filepath, s3, bucket_name, unique_model_id,1 )
    preprocessor_file_extension =_get_extension_from_filepath(preprocessor_filepath)
    #write runtime JSON
    json_path = "/tmp/runtime_data.json"
    if(preprocessor_file_extension == '.py'):
      runtime_preprocessor_type = "module"
    elif(preprocessor_file_extension == '.pkl'):
      runtime_preprocessor_type = "pickle object"
    else:
      runtime_preprocessor_type = "others"
    runtime_data = {}
    runtime_data["runtime_model"] = {"name": "runtime_model.onnx"}
    runtime_data["runtime_preprocessor"]= runtime_preprocessor_type

    #runtime_data = {"runtime_model": {"name": "runtime_model.onnx"},"runtime_preprocessor": runtime_preprocessor_type }
    json_string = json.dumps(runtime_data, sort_keys=False)
    with open(json_path, 'w') as outfile:
      outfile.write(json_string)
    s3["client"].upload_file(
           json_path , bucket_name , unique_model_id + "/runtime_data.json"
        )
    os.remove(json_path)  
  except Exception as err:
    raise AWSUploadError("There was a problem with model/preprocessor upload. "+str(err))
  model_filepath = file_key +file_extension
  #headers = {'content-type': 'application/json'}
  apiurl = create_prediction_api(my_credentials, model_filepath, unique_model_id, model_type, file_extension, categorical, labels,preprocessor, preprocessor_file_extension)
  finalresult= [apiurl["body"],apiurl["statusCode"], now, unique_model_id, bucket_name]
  return finalresult


def send_model_data_to_dyndb_and_return_api(api_info, my_credentials, private, categorical, variablename_and_type_data="default", preprocessor_filepath="default", preprocessor="default"):
    print("We need some information about your model before we can generate your API.  Please enter a name for your model, describe what your model does, and describe how well your model predicts new data.")
    print("   ")
    aishare_modelname = input("Enter model name:")
    aishare_modeldescription = input("Enter model description:")
    aishare_modeltype = input("Enter model category (i.e.- Text, Image, Audio, Video, or Tabular Data:")
    aishare_modelevaluation = input("Enter evaluation of how well model predicts new data:")
    aishare_tags = input("Enter search categories that describe your model (separate with commas):")
    aishare_apicalls = 0
    print("   ")
    #unpack user credentials
    username = my_credentials["username"]
    returned_jwt_token = my_credentials["returned_jwt_token"]
    aws_key = my_credentials["aws_key"]
    aws_password = my_credentials["aws_password"]
    AI_MODELSHARE_AccessKeyId = my_credentials["AI_MODELSHARE_AccessKeyId"]
    AI_MODELSHARE_SecretAccessKey = my_credentials["AI_MODELSHARE_SecretAccessKey"]
    region = my_credentials["region"]
    iamusername = my_credentials['iamusername']
    policy_name = my_credentials['policy_name']
    policy_arn = my_credentials['policy_arn']
    unique_model_id = api_info[3]
    bucket_name = api_info[4]
    if variablename_and_type_data == "default":
      variablename_and_type_data = ["",""]
    
   # needs to use double backslashes and have full filepath
    preprocessor_file_extension =_get_extension_from_filepath(preprocessor_filepath)
    bodydata={"id": int(math.log(1/((time.time()*1000000)))*100000000000000),
              "unique_model_id":unique_model_id,
      "apideveloper":username, # change this to first and last name
      "apimodeldescription":aishare_modeldescription,
      "apimodelevaluation": aishare_modelevaluation,
      "apimodeltype":aishare_modeltype ,
      "apiurl": api_info[0].strip('\"'), # getting rid of extra quotes that screw up dynamodb string search on apiurls
      "bucket_name":bucket_name,
      "version":1,
      "modelname": aishare_modelname,
      "tags": aishare_tags,
      "Private":private,
      "Categorical":categorical,
      "delete":"FALSE",
      "input_feature_dtypes":variablename_and_type_data[0],
      "input_feature_names":variablename_and_type_data[1],
      "preprocessor":preprocessor,
      "preprocessor_fileextension":preprocessor_file_extension
      }
    # Get the response
    headers_with_authentication= {'Content-Type': 'application/json','Authorization':returned_jwt_token,'Access-Control-Allow-Headers':'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization','Access-Control-Allow-Origin':'*'}
    #modeltoapi lambda function invoked through below url to return new prediction api in response
    requests.post("https://bbfgxopv21.execute-api.us-east-1.amazonaws.com/dev/todos", json=bodydata, headers=headers_with_authentication)
    start = api_info[2]
    end = datetime.datetime.now()
    difference = (end - start).total_seconds()
    finalresult2 = "Your AI Model Share API was created in " + str(int(difference)) + " seconds." + " API Url: " + api_info[0]
    s3, iam, region = get_s3_iam_client(aws_key, aws_password, region)
    policy_response = iam["client"].get_policy(
        PolicyArn=policy_arn
    )
    user_policy = iam["resource"].UserPolicy(iamusername,policy_response['Policy']['PolicyName'])
    response = iam["client"].detach_user_policy(
        UserName=iamusername,
        PolicyArn=policy_arn
    )
    # add new policy that only allows file upload to bucket
    policy = iam["resource"].Policy(policy_arn)
    response = policy.delete()
    s3upload_policy = _custom_upload_policy(bucket_name, unique_model_id)
    s3uploadpolicy_name = 'temporaryaccessAImodelsharePolicy'+str(uuid.uuid1().hex)
    s3uploadpolicy_response = iam["client"].create_policy(
      PolicyName=s3uploadpolicy_name,
      PolicyDocument=json.dumps(s3upload_policy)
    )
    user = iam["resource"].User(iamusername)
    response = user.attach_policy(
          PolicyArn=s3uploadpolicy_response['Policy']['Arn']
      )
    finalresultteams3info = "Your team members can submit improved models to your prediction api using the update_model_version() function. \nTo upload new models and/or preprocessors to this model team members should use the following awskey/password/region:\n\n aws_key = "+ AI_MODELSHARE_AccessKeyId+", aws_password = " + AI_MODELSHARE_SecretAccessKey + " region = " + region+".  \n\nThis aws key/password combination limits team members to file upload access only."
    return print(finalresult2+"\n"+finalresultteams3info)


def model_to_api(model_filepath, my_credentials, model_type, private, categorical, trainingdata, y_train, preprocessor_filepath, preprocessor):
  print("   ")
  print("Creating your prediction API. (Process typically takes less than one minute)...")
  if model_type=="tabular" or "keras_tabular" or'Tabular':
    variablename_and_type_data = extract_varnames_fromtrainingdata(trainingdata)
    print(variablename_and_type_data)
  if categorical=="TRUE":
      try:
        labels=y_train.columns.tolist()
      except:
        labels = list(set(y_train.to_frame()['tags'].tolist()))
  else:
      labels="no data"
  api_info = take_user_info_and_generate_api(model_filepath, my_credentials, model_type, categorical,private, labels,preprocessor_filepath, preprocessor)
  final_results = send_model_data_to_dyndb_and_return_api(api_info,my_credentials, private, categorical,variablename_and_type_data, preprocessor_filepath, preprocessor)
  return final_results


__all__ = [
    take_user_info_and_generate_api,
    send_model_data_to_dyndb_and_return_api,
    model_to_api,
]
