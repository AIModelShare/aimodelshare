import boto3
import botocore
import os
from numpy.core.fromnumeric import var
import requests
import uuid
import json
import math
import time
import datetime
import onnx
from aimodelshare.tools import extract_varnames_fromtrainingdata, _get_extension_from_filepath
from aimodelshare.aws import get_s3_iam_client
from aimodelshare.bucketpolicy import _custom_upload_policy
from aimodelshare.exceptions import AuthorizationError, AWSAccessError, AWSUploadError
from aimodelshare.api import create_prediction_api
from aimodelshare.preprocessormodules import upload_preprocessor
from aimodelshare.model import _get_predictionmodel_key, _extract_model_metadata


def take_user_info_and_generate_api(model_filepath, my_credentials, model_type, categorical,labels, preprocessor_filepath,y_test=None):
    """
    Generates an api using model parameters and user credentials, from the user

    -----------
    Parameters

    model_filepath :  string ends with '.onnx'
                    value - Absolute path to model file [REQUIRED] to be set by the user
                    .onnx is the only accepted model file extension
                    "example_model.onnx" filename for file in directory.
                    "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
    preprocessor_filepath:  string
                            value - absolute path to preprocessor file 
                            [REQUIRED] to be set by the user
                            "./preprocessor.zip" 
                            searches for an exported zip preprocessor file in the current directory
                            file is generated using export_preprocessor function from the AI Modelshare library 
    my_credentials :  dict 
                    value - User credentials generated and authenticated,  using aws credentials and aimodelshare username,password
    model_type :  string 
                values - [ 'text' , 'image' , 'tabular' , 'timeseries' ]
                Type of model data 
    categorical:  string
                  "TRUE" if model is Classification, categorical type
                  "FALSE" if model is continuous, Regression type
    labels:   list
            value - labels for training data
            can be extracted from columns of y train or can be provided by the user

    y_test :
    -----------
    Returns
    finalresult : list 
                length 5
                [ api url, api status code,   
                start time for api generation, 
                model unique key, 
                s3 bucket name
                  ]                   

    """
    import tempfile
    from zipfile import ZipFile
    import os
    import random

    # create temporary folder
    temp_dir = tempfile.gettempdir()
    # unpack user credentials
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
    # model upload
    Filepath = model_filepath
    model = onnx.load(model_filepath)
    metadata = _extract_model_metadata(model)
    input_shape = metadata["input_shape"]
    #tab_imports ='./tabular_imports.pkl'
    #img_imports ='./image_imports.pkl'
    file_extension = _get_extension_from_filepath(Filepath)
    unique_model_id = str(uuid.uuid1().hex)
    file_key, versionfile_key = _get_predictionmodel_key(
        unique_model_id, file_extension)
    try:
        s3["client"].upload_file(Filepath, bucket_name,  file_key)
        s3["client"].upload_file(Filepath, bucket_name,  versionfile_key)

        # preprocessor upload
        #s3["client"].upload_file(tab_imports, bucket_name,  'tabular_imports.pkl')
        #s3["client"].upload_file(img_imports, bucket_name,  'image_imports.pkl')
        # preprocessor upload

        # ADD model/Preprocessor VERSION
        response = upload_preprocessor(
            preprocessor_filepath, s3, bucket_name, unique_model_id, 1)
        preprocessor_file_extension = _get_extension_from_filepath(
            preprocessor_filepath)
        # write runtime JSON
        json_path = os.path.join(temp_dir, "runtime_data.json")
        if(preprocessor_file_extension == '.py'):
            runtime_preprocessor_type = "module"
        elif(preprocessor_file_extension == '.pkl'):
            runtime_preprocessor_type = "pickle object"
        else:
            runtime_preprocessor_type = "others"
        runtime_data = {}
        runtime_data["runtime_model"] = {"name": "runtime_model.onnx"}
        runtime_data["runtime_preprocessor"] = runtime_preprocessor_type


        if(y_test==None):
            pass
        else:
            ytest_path = os.path.join(temp_dir, "ytest.pkl")
            import pickle
            #ytest data to load to s3
            pickle.dump( list(y_test),open(ytest_path,"wb"))
            s3["client"].upload_file(ytest_path, bucket_name,  unique_model_id + "/ytest.pkl")
            
        #runtime_data = {"runtime_model": {"name": "runtime_model.onnx"},"runtime_preprocessor": runtime_preprocessor_type }
        json_string = json.dumps(runtime_data, sort_keys=False)
        with open(json_path, 'w') as outfile:
            outfile.write(json_string)
        s3["client"].upload_file(
            json_path, bucket_name, unique_model_id + "/runtime_data.json"
        )
        os.remove(json_path)
    except Exception as err:
        raise AWSUploadError(
            "There was a problem with model/preprocessor upload. "+str(err))

    #headers = {'content-type': 'application/json'}
    apiurl = create_prediction_api(my_credentials, model_filepath, unique_model_id,
                                   model_type, categorical, labels)

    finalresult = [apiurl["body"], apiurl["statusCode"],
                   now, unique_model_id, bucket_name, input_shape]
    return finalresult


def send_model_data_to_dyndb_and_return_api(api_info, my_credentials, private, categorical, preprocessor_filepath, variablename_and_type_data="default"):
    """
    Updates dynamodb with model data taken as input from user along with already generated api info
    -----------
    Parameters
    api_info :  list  
              length 5
              api and s3 bucket information 
              returned from take_user_info_and_generate_api function
    my_credentials :  dict 
                    value - User credentials generated and authenticated,  using aws credentials and aimodelshare username,password
    private :   string, default="FALSE"
              TRUE if model and its corresponding data is not public
              FALSE if model and its corresponding data is public            
    categorical: string, default="TRUE"
                  TRUE if model is of Classification type with categorical variables
                  FALSE if model is Regression type with continuous variables
    preprocessor_filepath:  string
                          value - Absolute path to preprocessor file [REQUIRED] to be set by the user
                          "./preprocessor.zip" 
                          searches for an exported zip preprocessor file in the current directory 
    variablename_and_type_data :  list, default='default'
                                value- extracted from trainingdata
                                [variable types,variable columns]
                                'default' when training data info is not available to extract columns
    
    -----------
    Results
    print (api_info) : statements with the generated live prediction API information for the user

    """
    print("We need some information about your model before we can generate your API.  Please enter a name for your model, describe what your model does, and describe how well your model predicts new data.")
    print("   ")
    aishare_modelname = input("Enter model name:")
    aishare_modeldescription = input("Enter model description:")
    aishare_modeltype = input(
        "Enter model category (i.e.- Text, Image, Audio, Video, or TimeSeries Data:")
    aishare_modelevaluation = input(
        "Enter evaluation of how well model predicts new data:")
    aishare_tags = input(
        "Enter search categories that describe your model (separate with commas):")
    aishare_apicalls = 0
    print("   ")
    # unpack user credentials
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
    input_shape = api_info[5]
    if variablename_and_type_data == "default":
        variablename_and_type_data = ["", ""]

     # needs to use double backslashes and have full filepath
    preprocessor_file_extension = _get_extension_from_filepath(
        preprocessor_filepath)
    bodydata = {"id": int(math.log(1/((time.time()*1000000)))*100000000000000),
                "unique_model_id": unique_model_id,
                "apideveloper": username,  # change this to first and last name
                "apimodeldescription": aishare_modeldescription,
                "apimodelevaluation": aishare_modelevaluation,
                "apimodeltype": aishare_modeltype,
                # getting rid of extra quotes that screw up dynamodb string search on apiurls
                "apiurl": api_info[0].strip('\"'),
                "bucket_name": bucket_name,
                "version": 1,
                "modelname": aishare_modelname,
                "tags": aishare_tags,
                "Private": private,
                "Categorical": categorical,
                "delete": "FALSE",
                "input_feature_dtypes": variablename_and_type_data[0],
                "input_feature_names": variablename_and_type_data[1],
                "preprocessor": preprocessor_filepath,
                "preprocessor_fileextension": preprocessor_file_extension,
                "input_shape": input_shape
                }
    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'Authorization': returned_jwt_token, 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    requests.post("https://bbfgxopv21.execute-api.us-east-1.amazonaws.com/dev/todos",
                  json=bodydata, headers=headers_with_authentication)
    start = api_info[2]
    end = datetime.datetime.now()
    difference = (end - start).total_seconds()
    finalresult2 = "Your AI Model Share API was created in " + \
        str(int(difference)) + " seconds." + " API Url: " + api_info[0]
    s3, iam, region = get_s3_iam_client(aws_key, aws_password, region)
    policy_response = iam["client"].get_policy(
        PolicyArn=policy_arn
    )
    user_policy = iam["resource"].UserPolicy(
        iamusername, policy_response['Policy']['PolicyName'])
    response = iam["client"].detach_user_policy(
        UserName=iamusername,
        PolicyArn=policy_arn
    )
    # add new policy that only allows file upload to bucket
    policy = iam["resource"].Policy(policy_arn)
    response = policy.delete()
    s3upload_policy = _custom_upload_policy(bucket_name, unique_model_id)
    s3uploadpolicy_name = 'temporaryaccessAImodelsharePolicy' + \
        str(uuid.uuid1().hex)
    s3uploadpolicy_response = iam["client"].create_policy(
        PolicyName=s3uploadpolicy_name,
        PolicyDocument=json.dumps(s3upload_policy)
    )
    user = iam["resource"].User(iamusername)
    response = user.attach_policy(
        PolicyArn=s3uploadpolicy_response['Policy']['Arn']
    )
    finalresultteams3info = "Your team members can submit improved models to your prediction api using the update_model_version() function. \nTo upload new models and/or preprocessors to this model team members should use the following awskey/password/region:\n\n aws_key = " + \
        AI_MODELSHARE_AccessKeyId+", aws_password = " + AI_MODELSHARE_SecretAccessKey + " region = " + \
        region+".  \n\nThis aws key/password combination limits team members to file upload access only."
    api_info = finalresult2+"\n"+finalresultteams3info
    return print(api_info)


def model_to_api(model_filepath, my_credentials, model_type, private, categorical, trainingdata, y_train,preprocessor_filepath, y_test=None):
    """
      Launches a live prediction REST API for deploying ML models using model parameters and user credentials, provided by the user
      Inputs : 8
      Output : model launched to an API
               detaled API info printed out
     
      -----------
      Parameters 
      
      my_credentials :  dict 
                        value - user credentials generated and authenticated,  
                        using user's aws credentials and aimodelshare username,password
      model_filepath :  string ends with '.onnx'
                        value - Absolute path to model file 
                        [REQUIRED] to be set by the user
                        .onnx is the only accepted model file extension
                        "example_model.onnx" filename for file in directory.
                        "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
      preprocessor_filepath:  string
                              value - absolute path to preprocessor file 
                              [REQUIRED] to be set by the user
                              "./preprocessor.zip" 
                              searches for an exported zip preprocessor file in the current directory
                              file is generated using export_preprocessor function from the AI Modelshare library  
      model_type :  string 
                    values - [ 'Text' , 'Image' , 'Tabular' , 'Timeseries' ] 
                    type of model data     
      categorical:    bool, default=True
                      True [DEFAULT] if model is of Classification type with categorical target variables
                      False if model is of Regression type with continuous target variables
      y_train : training labels of size of dataset
                value - y values for model
                [REQUIRED] for classification type models
                expects a one hot encoded y train data format
      trainingdata :  training dataset, default='default'
                      expects a pd dataframe 
                      value - x values(inputs) for model
                      [REQUIRED] for tabular data
      private :   bool, default = False
                  True if model and its corresponding data is not public
                  False [DEFAULT] if model and its corresponding data is public    
      y_test :  y labels for test data 
                [REQUIRED] for eval metrics
                expects a one hot encoded y test data format      
      -----------
      Returns
      print_api_info : prints statements with generated live prediction API details
                      also prints steps to update the model submissions by the user/team
                                 

    """
    # Used 2 python functions :
    #  1. take_user_info_and_generate_api : to upload model/preprocessor and generate an api for model submitted by user
    #  2. send_model_data_to_dyndb_and_return_api : to add new record to database with user data, model and api related information

    print("   ")
    print("Creating your prediction API. (Process typically takes less than one minute)...")
    variablename_and_type_data = None
    private = str(private).upper()
    categorical = str(categorical).upper()
    if model_type == "tabular" or "keras_tabular" or 'Tabular':
        variablename_and_type_data = extract_varnames_fromtrainingdata(
            trainingdata)
    if categorical == "TRUE":
        try:
            labels = y_train.columns.tolist()
        except:
            labels = list(set(y_train.to_frame()['tags'].tolist()))
    else:
        labels = "no data"
    api_info = take_user_info_and_generate_api(
        model_filepath, my_credentials, model_type, categorical, labels,preprocessor_filepath,y_test)
    print_api_info = send_model_data_to_dyndb_and_return_api(
        api_info, my_credentials, private, categorical,preprocessor_filepath, variablename_and_type_data)
    return print_api_info


__all__ = [
    take_user_info_and_generate_api,
    send_model_data_to_dyndb_and_return_api,
    model_to_api,
]
