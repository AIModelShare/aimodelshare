import email
import boto3
import botocore
import os
import jwt
from numpy.core.fromnumeric import var
import requests
import uuid
import json
import math
import time
import datetime
import onnx
import tempfile
import sys
import base64
import mimetypes
import numpy as np
import pandas as pd
from aimodelshare.tools import extract_varnames_fromtrainingdata, _get_extension_from_filepath
from aimodelshare.aws import get_s3_iam_client, run_function_on_lambda, get_token, get_aws_token, get_aws_client
from aimodelshare.bucketpolicy import _custom_upload_policy
from aimodelshare.exceptions import AuthorizationError, AWSAccessError, AWSUploadError
from aimodelshare.api import get_api_json
from aimodelshare.modeluser import create_user_getkeyandpassword
from aimodelshare.preprocessormodules import upload_preprocessor
from aimodelshare.model import _get_predictionmodel_key, _extract_model_metadata
from aimodelshare.data_sharing.share_data import share_data_codebuild
from aimodelshare.aimsonnx import _get_metadata

def take_user_info_and_generate_api(model_filepath, model_type, categorical,labels, preprocessor_filepath,
                                    custom_libraries, requirements, exampledata_json_filepath, repo_name, 
                                    image_tag, reproducibility_env_filepath, memory, timeout, pyspark_support=False):
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
    model_type :  string 
                values - [ 'text' , 'image' , 'tabular' , 'timeseries' ]
                Type of model data 
    categorical:  string
                  "TRUE" if model is Classification, categorical type
                  "FALSE" if model is continuous, Regression type
    labels:   list
            value - labels for training data
            can be extracted from columns of y train or can be provided by the user
    custom_libraries:   string
                  "TRUE" if user wants to load custom Python libraries to their prediction runtime
                  "FALSE" if user wishes to use AI Model Share base libraries including latest versions of most common ML libs.
     

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
   
    api_json= get_api_json()
    user_client = boto3.client('apigateway', aws_access_key_id=str(
    os.environ.get("AWS_ACCESS_KEY_ID")), aws_secret_access_key=str(os.environ.get("AWS_SECRET_ACCESS_KEY")), region_name=str(os.environ.get("AWS_REGION")))

    response2 = user_client.import_rest_api(
    failOnWarnings=True,
    parameters={
        'endpointConfigurationTypes': 'REGIONAL'
    },
    body=api_json
    )

    api_id = response2['id']
    now = datetime.datetime.now()
    s3, iam, region = get_s3_iam_client(os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"), os.environ.get("AWS_REGION"))
    
    def create_bucket(s3_client, bucket_name, region):
        try:
            response=s3_client.head_bucket(Bucket=bucket_name)
        except:
            if(region=="us-east-1"):
                response = s3_client.create_bucket(
                    ACL="private",
                    Bucket=bucket_name
                )
            else:
                location={'LocationConstraint': region}
                response=s3_client.create_bucket(
                    ACL="private",
                    Bucket=bucket_name,
                    CreateBucketConfiguration=location
                )
        return response

    create_bucket(s3['client'], os.environ.get("BUCKET_NAME"), region)

    # model upload

    Filepath = model_filepath
    model = onnx.load(model_filepath)
    metadata = _extract_model_metadata(model)
    input_shape = metadata["input_shape"]
    #tab_imports ='./tabular_imports.pkl'
    #img_imports ='./image_imports.pkl'
    file_extension = _get_extension_from_filepath(Filepath)
    unique_model_id = str(api_id)
    file_key, versionfile_key = _get_predictionmodel_key(
        unique_model_id, file_extension)
    try:
        s3["client"].upload_file(exampledata_json_filepath, os.environ.get("BUCKET_NAME"), unique_model_id + "/exampledata.json") 
    except:
        pass
    try:
        s3["client"].upload_file(Filepath, os.environ.get("BUCKET_NAME"),  file_key)
        s3["client"].upload_file(Filepath, os.environ.get("BUCKET_NAME"),  versionfile_key)

        # preprocessor upload
        #s3["client"].upload_file(tab_imports, os.environ.get("BUCKET_NAME"),  'tabular_imports.pkl')
        #s3["client"].upload_file(img_imports, os.environ.get("BUCKET_NAME"),  'image_imports.pkl')
        # preprocessor upload

        # ADD model/Preprocessor VERSION
        response = upload_preprocessor(
            preprocessor_filepath, s3, os.environ.get("BUCKET_NAME"), unique_model_id, 1)
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
            
        #runtime_data = {"runtime_model": {"name": "runtime_model.onnx"},"runtime_preprocessor": runtime_preprocessor_type }
        json_string = json.dumps(runtime_data, sort_keys=False)
        with open(json_path, 'w') as outfile:
            outfile.write(json_string)
        s3["client"].upload_file(
            json_path, os.environ.get("BUCKET_NAME"), unique_model_id + "/runtime_data.json"
        )
        os.remove(json_path)

        # upload model metadata
        upload_model_metadata(model, s3, os.environ.get("BUCKET_NAME"), unique_model_id)

        # upload reproducibility env
        if reproducibility_env_filepath:
            upload_reproducibility_env(reproducibility_env_filepath, s3, os.environ.get("BUCKET_NAME"), unique_model_id)
    except Exception as err:
        raise AWSUploadError(
            "There was a problem with model/preprocessor upload. "+str(err))

    #Delete Legacy exampledata json:
    try:
        os.remove(exampledata_json_filepath)
    except:
        pass

    #headers = {'content-type': 'application/json'}

    ### Progress Update #2/6 {{{
    sys.stdout.write('\r')
    sys.stdout.write("[========                             ] Progress: 30% - Building lambda functions and updating permissions...")
    sys.stdout.flush()
    # }}}
    
    from aimodelshare.api import create_prediction_api
    apiurl = create_prediction_api(model_filepath, unique_model_id,
                                   model_type, categorical, labels,api_id,
                                   custom_libraries, requirements, repo_name, 
                                   image_tag, memory, timeout, pyspark_support=pyspark_support)

    finalresult = [apiurl["body"], apiurl["statusCode"],
                   now, unique_model_id, os.environ.get("BUCKET_NAME"), input_shape]
    return finalresult

def upload_reproducibility_env(reproducibility_env_file, s3, bucket, model_id):
    # Check the reproducibility_env {{{
    with open(reproducibility_env_file) as json_file:
      reproducibility_env = json.load(json_file)
      if "global_seed_code" not in reproducibility_env \
              or "local_seed_code" not in reproducibility_env \
              or "gpu_cpu_parallelism_ops" not in reproducibility_env \
              or "session_runtime_info" not in reproducibility_env:
          raise Exception("reproducibility environment is not complete")

    # Upload the json {{{
    try:
        s3["client"].upload_file(
            reproducibility_env_file, bucket, model_id + "/runtime_reproducibility.json"
        )
    except Exception as err:
        raise err
    # }}}

def upload_model_metadata(model, s3, bucket, model_id):
    meta_dict = _get_metadata(model)
    model_metadata = {
        "model_config": meta_dict["model_config"],
        "ml_framework": meta_dict["ml_framework"],
        "model_type": meta_dict["model_type"]
    }

    temp = tempfile.mkdtemp()
    model_metadata_path = temp + "/" + 'model_metadata.json'
    with open(model_metadata_path, 'w') as outfile:
        json.dump(model_metadata, outfile)

    # Upload the json {{{
    try:
        s3["client"].upload_file(
            model_metadata_path, bucket, model_id + "/runtime_metadata.json"
        )
    except Exception as err:
        raise err

def send_model_data_to_dyndb_and_return_api(api_info, private, categorical, preprocessor_filepath,
                                            aishare_modelname, aishare_modeldescription, aishare_modelevaluation, model_type,
                                            aishare_tags, aishare_apicalls, exampledata_json_filepath,
                                            variablename_and_type_data="default", email_list=[]):
    """
    Updates dynamodb with model data taken as input from user along with already generated api info
    -----------
    Parameters
    api_info :  list  
              length 5
              api and s3 bucket information 
              returned from take_user_info_and_generate_api function
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
                                value- extracted from example_data
                                [variable types,variable columns]
                                'default' when training data info is not available to extract columns
    email_list: list of strings
                values - list including all emails of users who have access the playground.
                list should contain same emails used to sign up for modelshare.org account.
                [OPTIONAL] to be set by the playground owner
    -----------
    Results
    print (api_info) : statements with the generated live prediction API information for the user

    """
    
    # unpack user credentials
    unique_model_id = api_info[3]
    bucket_name = api_info[4]
    input_shape = api_info[5]
    if variablename_and_type_data == "default":
        variablename_and_type_data = ["", ""]

     # needs to use double backslashes and have full filepath
    preprocessor_file_extension = _get_extension_from_filepath(
        preprocessor_filepath)
    if exampledata_json_filepath!="":
        exampledata_addtodatabase={"exampledata":"TRUE"}
    else:
        exampledata_addtodatabase={"exampledata":"FALSE"}
    bodydata = {
        "id": int(math.log(1/((time.time()*1000000)))*100000000000000),
        "unique_model_id": unique_model_id,
        "apideveloper": os.environ.get("username"),  # change this to first and last name
        "apimodeldescription": aishare_modeldescription,
        "apimodelevaluation": aishare_modelevaluation,
        "apimodeltype": model_type,
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
        "input_shape": input_shape,
        "email_list": email_list,
        "useremails": ','.join(email_list),
    }
    bodydata.update(exampledata_addtodatabase)
    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    response = requests.post("https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                              json=bodydata, headers=headers_with_authentication)
    response_string = response.text
    response_string = response_string[1:-1]
    import json
    response_stringfinal = json.loads(response_string)["id"]
    # Build output {{{
    final_message = ("\nYou can now use your Model Playground.\n\n"
                     "Follow this link to explore your Model Playground's functionality\n"
                     "You can make predictions with the Dashboard and access example code from the Programmatic tab.\n")
    web_dashboard_url = ("https://www.modelshare.org/detail/"+ response_stringfinal)
    
    start = api_info[2]
    end = datetime.datetime.now()
    difference = (end - start).total_seconds()
    finalresult2 = "Success! Your Model Playground was created in " + \
        str(int(difference)) + " seconds. \n" + " Playground Url: " + api_info[0] 


    # }}}

    ### Progress Update #6/6 {{{
    sys.stdout.write('\r')
    sys.stdout.write("[=====================================] Progress: 100% - Complete!                                            ")
    sys.stdout.flush()
    # }}}

    return print("\n\n" + finalresult2 + "\n" + final_message + web_dashboard_url)


def model_to_api(model_filepath, model_type, private, categorical, y_train, preprocessor_filepath, 
                custom_libraries="FALSE", example_data=None, image="", 
                base_image_api_endpoint="https://vupwujn586.execute-api.us-east-1.amazonaws.com/dev/copybasetouseracct", 
                update=False, reproducibility_env_filepath=None, memory=None, timeout=None, email_list=[],pyspark_support=False):
    """
      Launches a live prediction REST API for deploying ML models using model parameters and user credentials, provided by the user
      Inputs : 8
      Output : model launched to an API
               detaled API info printed out
     
      -----------
      Parameters 
      
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
      private :   bool, default = False
                  True if model and its corresponding data is not public
                  False [DEFAULT] if model and its corresponding data is public   
      custom_libraries:   string
                  "TRUE" if user wants to load custom Python libraries to their prediction runtime
                  "FALSE" if user wishes to use AI Model Share base libraries including latest versions of most common ML libs.
      example_data:  pandas DataFrame (for tabular & text data) OR filepath as string (image, audio, video data)
                     tabular data - pandas DataFrame in same structure expected by preprocessor function
                     other data types - absolute path to folder containing example data
                                        (first five files with relevent file extensions will be accepted)
                     [REQUIRED] for tabular data
      reproducibility_env_filepath: string
                                value - absolute path to environment environment json file 
                                [OPTIONAL] to be set by the user
                                "./reproducibility.json" 
                                file is generated using export_reproducibility_env function from the AI Modelshare library
      email_list: list of strings
                values - list including all emails of users who have access the playground.
                list should contain same emails used to sign up for modelshare.org account.
                [OPTIONAL] to be set by the playground owner
      -----------
      Returns
      print_api_info : prints statements with generated live prediction API details
                      also prints steps to update the model submissions by the user/team
                                 

    """
    # Used 2 python functions :
    #  1. take_user_info_and_generate_api : to upload model/preprocessor and generate an api for model submitted by user
    #  2. send_model_data_to_dyndb_and_return_api : to add new record to database with user data, model and api related information

    # Get user inputs, pass to other functions  {{{
    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                         aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                         region_name=os.environ.get("AWS_REGION"))

    if all([isinstance(email_list, list)]):
        idtoken = get_aws_token()
        decoded = jwt.decode(idtoken, options={"verify_signature": False})  # works in PyJWT < v2.0
        email=None
        email = decoded['email']
        # Owner has to be the first on the list
        email_list.insert(0, email)
        if any([private==False,private==None]):
          email_list=["publicaimsplayground"]
        else:
          pass
    else:
        return print("email_list argument empty or incorrectly formatted. Please provide a list of emails for authorized playground users formatted as strings.")

    if(image!=""):
        repo_name, image_tag = image.split(':')
    elif model_type=="tabular":
        repo_name, image_tag = "aimodelshare_base_image", "tabular"
    elif model_type=="text":
        repo_name, image_tag = "aimodelshare_base_image", "texttest"
    elif model_type=="image":
        repo_name, image_tag = "aimodelshare_base_image", "v3"
    elif model_type=="video":
        repo_name, image_tag = "aimodelshare_base_image", "v3"
    else:
        repo_name, image_tag = "aimodelshare_base_image", "v3"

    # Pyspark mode
    if pyspark_support:
        repo_name, image_tag = "aimodelshare_base_image", "pyspark"
    
    from aimodelshare.containerization import clone_base_image
    response = clone_base_image(user_session, repo_name, image_tag, "517169013426", base_image_api_endpoint, update)
    if(response["Status"]==0):
        print(response["Success"])
        return

    print("We need some information about your model before we can build your REST API and interactive Model Playground.")
    print("   ")

    requirements = ""
    if(any([custom_libraries=='TRUE',custom_libraries=='true'])):
        requirements = input("Enter all required Python libraries you need at prediction runtime (separated with commas):")
        #_confirm_libraries_exist(requirements)
        
    aishare_modelname = input("Model Name (for AI Model Share Website):")
    aishare_modeldescription = input("Model Description (Explain what your model does and \n why end-users would find your model useful):")
    aishare_modelevaluation = "unverified" # verified metrics added to playground once 1. a model is submitted to a competition leaderboard and 2. playground owner updates runtime
                                           #...model with update_runtime_model()
    aishare_tags = input(
        "Model Key Words (Search categories that describe your model, separated with commas):")
    aishare_apicalls = 0
    print("   ")
    #  }}}

    # Force user to provide example data for tabular models {{{
    if any([model_type.lower() == "tabular", model_type.lower() == "timeseries"]):
        if not isinstance(example_data, pd.DataFrame):
            return print("Error: Example data is required for tabular models. \n Please provide a pandas DataFrame with a sample of your X data (in the format expected by your preprocessor) and try again.")
    else:
        pass
    #}}}
        
    print("Creating your prediction API. (This process may take several minutes.)\n")
    variablename_and_type_data = None
    private = str(private).upper()
    categorical = str(categorical).upper()
    if model_type == "tabular" or "keras_tabular" or 'Tabular':
        variablename_and_type_data = extract_varnames_fromtrainingdata(
            example_data)
    if categorical == "TRUE":
        try:
            labels = y_train.columns.tolist()
        except:
            #labels = list(set(y_train.to_frame()['tags'].tolist()))
            labels = list(set(y_train))
    else:
        labels = "no data"

    # Create Example Data JSON
    exampledata_json_filepath = ""
    if example_data is not None:
        _create_exampledata_json(model_type, example_data)
        exampledata_json_filepath = os.getcwd() + "/exampledata.json"

    ### Progress Update #1/6 {{{
    sys.stdout.write("[===                                  ] Progress: 5% - Accessing Amazon Web Services, uploading resources...")
    sys.stdout.flush()
    # }}}

    api_info = take_user_info_and_generate_api( 
        model_filepath, model_type, categorical, labels, 
        preprocessor_filepath, custom_libraries, requirements, 
        exampledata_json_filepath, repo_name, image_tag, 
        reproducibility_env_filepath, memory, timeout, pyspark_support=pyspark_support)

    ### Progress Update #5/6 {{{
    sys.stdout.write('\r')
    sys.stdout.write("[=================================    ] Progress: 90% - Finishing web dashboard...                           ")
    sys.stdout.flush()
    # }}}
    
    print_api_info = send_model_data_to_dyndb_and_return_api(
        api_info, private, categorical,preprocessor_filepath, aishare_modelname,
        aishare_modeldescription, aishare_modelevaluation, model_type,
        aishare_tags, aishare_apicalls, exampledata_json_filepath,
        variablename_and_type_data, email_list)
    
    return api_info[0]

def create_competition(apiurl, data_directory, y_test, eval_metric_filepath=None, email_list=[], public=False, public_private_split=0.5):
    """
    Creates a model competition for a deployed prediction REST API
    Inputs : 4
    Output : Create ML model competition and allow authorized users to submit models to resulting leaderboard/competition
    
    ---------
    Parameters
    
    apiurl: string
            URL of deployed prediction API 
    
    y_test :  list of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
                [REQUIRED] to generate eval metrics in competition leaderboard
            
    data_directory : folder storing training data and test data (excluding Y test data)
    email_list: [REQUIRED] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
                                          
    ---------
    Returns
    finalmessage : Information such as how to submit models to competition

    """
    if all([isinstance(email_list, list)]):
        if any([len(email_list)>0, public=="True",public=="TRUE",public==True]):
              import jwt
              idtoken=get_aws_token()
              decoded = jwt.decode(idtoken, options={"verify_signature": False})  # works in PyJWT < v2.0
              email=decoded['email']
              email_list.append(email)
    else:
        return print("email_list argument empty or incorrectly formatted.  Please provide a list of emails for authorized competition participants formatted as strings.")

    # create temporary folder
    temp_dir = tempfile.gettempdir()
    
    s3, iam, region = get_s3_iam_client(os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"), os.environ.get("AWS_REGION"))
    
    # Get bucket and model_id subfolder for user based on apiurl {{{
    response, error = run_function_on_lambda(
        apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, api_bucket, model_id = json.loads(response.content.decode("utf-8"))
    # }}} 
    
    # upload y_test data: 
    ytest_path = os.path.join(temp_dir, "ytest.pkl")
    import pickle
    #ytest data to load to s3

    if y_test is not None:
        if type(y_test) is not list:
            y_test=y_test.tolist()
        else: 
            pass

        if all(isinstance(x, (np.float64)) for x in y_test):
              y_test = [float(i) for i in y_test]
        else: 
            pass


    pickle.dump(y_test,open(ytest_path,"wb"))
    s3["client"].upload_file(ytest_path, os.environ.get("BUCKET_NAME"),  model_id + "/competition/ytest.pkl")


    if eval_metric_filepath is not None:
    
        if isinstance(eval_metric_filepath, list):

            for i in eval_metric_filepath: 

                eval_metric_name = i.split('/')[-1]

                s3["client"].upload_file(i, os.environ.get("BUCKET_NAME"),  model_id + '/competition/metrics_' + eval_metric_name)

        else:

            eval_metric_name = eval_metric_filepath.split('/')[-1]

            print(eval_metric_name)

            s3["client"].upload_file(eval_metric_filepath, os.environ.get("BUCKET_NAME"), model_id + '/competition/metrics_' + eval_metric_name)


    # get api_id from apiurl, generate txt file name
    api_url_trim = apiurl.split('https://')[1]
    api_id = api_url_trim.split(".")[0]

    print("\n--INPUT COMPETITION DETAILS--\n")

    aishare_competitionname = input("Enter competition name:")
    aishare_competitiondescription = input("Enter competition description:")

    print("\n--INPUT DATA DETAILS--\n")
    print("Note: (optional) Save an optional LICENSE.txt file in your competition data directory to make users aware of any restrictions on data sharing/usage.\n")

    aishare_datadescription = input(
        "Enter data description (i.e.- filenames denoting training and test data, file types, and any subfolders where files are stored):")
    
    aishare_datalicense = input(
        "Enter optional data license descriptive name (e.g.- 'MIT, Apache 2.0, CC0, Other, etc.'):")    
    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                          aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                         region_name=os.environ.get("AWS_REGION"))
    account_number = user_session.client(
        'sts').get_caller_identity().get('Account')

    datauri=share_data_codebuild(account_number,os.environ.get("AWS_REGION"),data_directory)
    
    #create and upload json file with list of authorized users who can submit to this competition.
    _create_competitionuserauth_json(apiurl, email_list,public,datauri['ecr_uri'], submission_type="competition")
    _create_public_private_split_json(apiurl, public_private_split, "competition")

    bodydata = {"unique_model_id": model_id,
                "bucket_name": api_bucket,
                "apideveloper": os.environ.get("username"),  # change this to first and last name
                "competitionname":aishare_competitionname,                
                "competitiondescription": aishare_competitiondescription,
                # getting rid of extra quotes that screw up dynamodb string search on apiurls
                "apiurl": apiurl,
                "version": 0,
                "Private": "FALSE",
                "delete": "FALSE",
                'datadescription':aishare_datadescription,
                'dataecruri':datauri['ecr_uri'],
                 'datalicense': aishare_datalicense}
    
    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    requests.post("https://o35jwfakca.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)

      
    final_message = ("\n Success! Model competition created. \n\n"
                "You may now update your prediction API runtime model and verify evaluation metrics with the update_runtime_model() function.\n\n"
                "To upload new models and/or preprocessors to this API, team members should use \n"
                "the following credentials:\n\napiurl='" + apiurl+"'"+"\nfrom aimodelshare.aws import set_credentials\nset_credentials(apiurl=apiurl)\n\n"
                "They can then submit models to your competition by using the following code: \n\ncompetition= ai.Competition(apiurl)\n"
                "download_data('"+datauri['ecr_uri']+"') \n"
                 "# Use this data to preprocess data and train model. Write and save preprocessor fxn, save model to onnx file, generate predicted y values\n using X test data, then submit a model below.\n\n"
                "competition.submit_model(model_filepath, preprocessor_filepath, prediction_submission_list)")
  
    return print(final_message)




def create_experiment(apiurl, data_directory, y_test, eval_metric_filepath=None, email_list=[], public=False, public_private_split=0.5):
    """
    Creates a model experiment for a deployed prediction REST API
    Inputs : 4
    Output : Create ML model experiment and allow authorized users to submit models to resulting leaderboard/competition
    
    ---------
    Parameters
    
    apiurl: string
            URL of deployed prediction API 
    
    y_test :  list of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
                [REQUIRED] to generate eval metrics in competition leaderboard
            
    data_directory : folder storing training data and test data (excluding Y test data)
    email_list: [REQUIRED] list of comma separated emails for users who are allowed to submit models to experiment.  Emails should be strings in a list.
                                          
    ---------
    Returns
    finalmessage : Information such as how to submit models to competition

    """
    if all([isinstance(email_list, list)]):
        if any([len(email_list)>0, public=="True",public=="TRUE",public==True]):
              import jwt
              idtoken=get_aws_token()
              decoded = jwt.decode(idtoken, options={"verify_signature": False})  # works in PyJWT < v2.0
              email=decoded['email']
              email_list.append(email)
    else:
        return print("email_list argument empty or incorrectly formatted.  Please provide a list of emails for authorized competition participants formatted as strings.")

    # create temporary folder
    temp_dir = tempfile.gettempdir()
    
    s3, iam, region = get_s3_iam_client(os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"), os.environ.get("AWS_REGION"))
    
    # Get bucket and model_id subfolder for user based on apiurl {{{
    response, error = run_function_on_lambda(
        apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, api_bucket, model_id = json.loads(response.content.decode("utf-8"))
    # }}} 
    
    # upload y_test data: 
    ytest_path = os.path.join(temp_dir, "ytest.pkl")
    import pickle
    #ytest data to load to s3

    if y_test is not None:
        if type(y_test) is not list:
            y_test=y_test.tolist()
        else: 
            pass

        if all(isinstance(x, (np.float64)) for x in y_test):
              y_test = [float(i) for i in y_test]
        else: 
            pass


    pickle.dump(y_test,open(ytest_path,"wb"))
    s3["client"].upload_file(ytest_path, os.environ.get("BUCKET_NAME"),  model_id + "/experiment/ytest.pkl")


    if eval_metric_filepath is not None:
    
        if isinstance(eval_metric_filepath, list):

            for i in eval_metric_filepath: 

                eval_metric_name = i.split('/')[-1]

                s3["client"].upload_file(i, os.environ.get("BUCKET_NAME"),  model_id + '/experiment/metrics_' + eval_metric_name)

        else:

            eval_metric_name = eval_metric_filepath.split('/')[-1]

            print(eval_metric_name)

            s3["client"].upload_file(eval_metric_filepath, os.environ.get("BUCKET_NAME"), model_id + '/experiment/metrics_' + eval_metric_name)


    # get api_id from apiurl, generate txt file name
    api_url_trim = apiurl.split('https://')[1]
    api_id = api_url_trim.split(".")[0]

    print("\n--INPUT COMPETITION DETAILS--\n")

    aishare_competitionname = input("Enter experiment name:")
    aishare_competitiondescription = input("Enter experiment description:")

    print("\n--INPUT DATA DETAILS--\n")
    print("Note: (optional) Save an optional LICENSE.txt file in your experiment data directory to make users aware of any restrictions on data sharing/usage.\n")

    aishare_datadescription = input(
        "Enter data description (i.e.- filenames denoting training and test data, file types, and any subfolders where files are stored):")
    
    aishare_datalicense = input(
        "Enter optional data license descriptive name (e.g.- 'MIT, Apache 2.0, CC0, Other, etc.'):")    
    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                          aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                         region_name=os.environ.get("AWS_REGION"))
    account_number = user_session.client(
        'sts').get_caller_identity().get('Account')

    datauri=share_data_codebuild(account_number,os.environ.get("AWS_REGION"),data_directory)
    
    #create and upload json file with list of authorized users who can submit to this competition.
    _create_competitionuserauth_json(apiurl, email_list,public,datauri['ecr_uri'], submission_type="experiment")
    _create_public_private_split_json(apiurl, public_private_split, "experiment")

    bodydata = {"unique_model_id": model_id,
                "bucket_name": api_bucket,
                "apideveloper": os.environ.get("username"),  # change this to first and last name
                "experiment":"TRUE",
                "competitionname":aishare_competitionname,                
                "competitiondescription": aishare_competitiondescription,
                # getting rid of extra quotes that screw up dynamodb string search on apiurls
                "apiurl": apiurl,
                "version": 0,
                "Private": "FALSE",
                "delete": "FALSE",
                'datadescription':aishare_datadescription,
                'dataecruri':datauri['ecr_uri'],
                 'datalicense': aishare_datalicense}
    
    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    requests.post("https://o35jwfakca.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)

      
    final_message = ("\n Success! Model experiment created. \n\n"
                "You may now update your prediction API runtime model and verify evaluation metrics with the update_runtime_model() function.\n\n"
                "To upload new models and/or preprocessors to this API, team members should use \n"
                "the following credentials:\n\napiurl='" + apiurl+"'"+"\nfrom aimodelshare.aws import set_credentials\nset_credentials(apiurl=apiurl)\n\n"
                "They can then submit models to your experiment by using the following code: \n\nexperiment= ai.Experiment(apiurl)\n"
                "download_data('"+datauri['ecr_uri']+"') \n"
                 "# Use this data to preprocess data and train model. Write and save preprocessor fxn, save model to onnx file, generate predicted y values\n using X test data, then submit a model below.\n\n"
                "experiment.submit_model(model_filepath, preprocessor_filepath, prediction_submission_list)")
  
    return print(final_message)

def _create_public_private_split_json(apiurl, split=0.5, submission_type='competition'): 
      import json
      if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ,
           "username" in os.environ, 
           "password" in os.environ]):
        pass
      else:
          return print("'Set public-private split' unsuccessful. Please provide credentials with set_credentials().")

      # Create user session
      aws_client=get_aws_client(aws_key=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                aws_secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                aws_region=os.environ.get('AWS_REGION'))
      
      user_sess = boto3.session.Session(aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                        region_name=os.environ.get('AWS_REGION'))
      
      s3 = user_sess.resource('s3')

      # Get bucket and model_id for user based on apiurl {{{
      response, error = run_function_on_lambda(
          apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
      )
      if error is not None:
          raise error

      _, api_bucket, model_id = json.loads(response.content.decode("utf-8"))
      # }}}
      
      import json  
      import tempfile
      tempdir = tempfile.TemporaryDirectory()
      with open(tempdir.name+'/public_private_split.json', 'w', encoding='utf-8') as f:
          json.dump({"public_private_split": str(split)}, f, ensure_ascii=False, indent=4)

      aws_client['client'].upload_file(
            tempdir.name+"/public_private_split.json", api_bucket, model_id +"/"+submission_type+"/public_private_split.json"
        )
      
      return

def _create_competitionuserauth_json(apiurl, email_list=[],public=False, datauri=None, submission_type="competition"): 
      import json
      if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ,
           "username" in os.environ, 
           "password" in os.environ]):
        pass
      else:
          return print("'Update Runtime Model' unsuccessful. Please provide credentials with set_credentials().")

      # Create user session
      aws_client=get_aws_client(aws_key=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                aws_secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                aws_region=os.environ.get('AWS_REGION'))
      
      user_sess = boto3.session.Session(aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                        region_name=os.environ.get('AWS_REGION'))
      
      s3 = user_sess.resource('s3')

      # Get bucket and model_id for user based on apiurl {{{
      response, error = run_function_on_lambda(
          apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
      )
      if error is not None:
          raise error

      _, api_bucket, model_id = json.loads(response.content.decode("utf-8"))
      # }}}

      
      import json  
      import tempfile
      tempdir = tempfile.TemporaryDirectory()
      with open(tempdir.name+'/competitionuserdata.json', 'w', encoding='utf-8') as f:
          json.dump({"emaillist": email_list, "public":str(public).upper(),"datauri":str(datauri)}, f, ensure_ascii=False, indent=4)

      aws_client['client'].upload_file(
            tempdir.name+"/competitionuserdata.json", api_bucket, model_id +"/"+submission_type+"/competitionuserdata.json"
        )
      
      return

def update_playground_access_list(apiurl, email_list=[], update_type="Add"): 
    """
    Updates list of authenticated participants who can submit new models to a competition.
    ---------------
    Parameters:
    apiurl: string
            URL of deployed prediction API 
    
    email_list: [REQUIRED] list of comma separated emails for users who are allowed to access model playground.  Emails should be strings in a list.
    update_type:[REQUIRED] options, string: 'Add', 'Remove', 'Replace'. Add appends user emails to original list, Remove deletes users from list, 
                and 'Replace' overwrites the original list with the new list provided.    
    -----------------
    Returns
    response:   "Success" upon successful request
    """
    if update_type not in ['Add', 'Remove', 'Replace']:
        return "Error: update_type must be in the form of 'Add', 'Remove', or 'Replace'"
    
    bodydata = {
        "modifyaccess": "TRUE",
        "apideveloper": os.environ.get("username"),  # change this to first and last name
        "apiurl": apiurl,
        "operation": update_type,
        "email_list": email_list,
    }

    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    response = requests.post("https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                            json=bodydata, headers=headers_with_authentication)
    response_string = response.text
    response_string = response_string[1:-1]

    # Build output {{{
    finalresult = "Success! Your Model Playground email list has been updated. Playground Url: " + apiurl
    return finalresult

def get_playground_access_list(apiurl):
    bodydata = {
        "getaccess": "TRUE",
        "apideveloper": os.environ.get("username"),  # change this to first and last name
        "apiurl": apiurl,
    }

    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    response = requests.post("https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                            json=bodydata, headers=headers_with_authentication)
    response_string = response.text
    response_string = response_string[1:-1]

    return response_string.split(',')

def update_access_list(apiurl, email_list=[],update_type="Add", submission_type="competition"):
      """
      Updates list of authenticated participants who can submit new models to a competition.
      ---------------
      Parameters:
      apiurl: string
              URL of deployed prediction API 
      
      email_list: [REQUIRED] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
      update_type:[REQUIRED] options, string: 'Add', 'Remove', 'Replace_list','Get. Add appends user emails to original list, Remove deletes users from list, 
                  'Replace_list' overwrites the original list with the new list provided, and Get returns the current list.    
      -----------------
      Returns
      response:   "Success" upon successful request
      """
      import json
      import os
      if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ,
           "username" in os.environ, 
           "password" in os.environ]):
        pass
      else:
          return print("'Update unsuccessful. Please provide credentials with set_credentials().")

      if all([isinstance(email_list, list)]):
          if all([len(email_list)>0]):
              pass
      else:
          return print("email_list argument empty or incorrectly formatted.  Please provide a list of emails for authorized competition participants formatted as strings.")

      # Create user session
      aws_client=get_aws_client(aws_key=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                aws_secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                aws_region=os.environ.get('AWS_REGION'))
      
      user_sess = boto3.session.Session(aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                        region_name=os.environ.get('AWS_REGION'))
      
      s3 = user_sess.resource('s3')

      # Get bucket and model_id for user based on apiurl {{{
      response, error = run_function_on_lambda(
          apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
      )
      if error is not None:
          raise error

      _, api_bucket, model_id = json.loads(response.content.decode("utf-8"))
      # }}}

      email_list=json.loads(json.dumps(email_list))
    
       
      if update_type=="Replace_list":
          import json  
          import tempfile
          tempdir = tempfile.TemporaryDirectory()
          content_object = aws_client['resource'].Object(bucket_name=api_bucket, key=model_id +"/"+submission_type+"/competitionuserdata.json")
          file_content = content_object.get()['Body'].read().decode('utf-8')
          json_content = json.loads(file_content)
          json_content['emaillist']=email_list
          with open(tempdir.name+'/competitionuserdata.json', 'w', encoding='utf-8') as f:
              json.dump(json_content, f, ensure_ascii=False, indent=4)

          aws_client['client'].upload_file(
                tempdir.name+"/competitionuserdata.json", api_bucket, model_id +"/"+submission_type+"/competitionuserdata.json"
            )
      elif update_type=="Add":
          import json  
          import tempfile
          tempdir = tempfile.TemporaryDirectory()
          content_object = aws_client['resource'].Object(bucket_name=api_bucket, key=model_id +"/"+submission_type+"/competitionuserdata.json")
          file_content = content_object.get()['Body'].read().decode('utf-8')
          json_content = json.loads(file_content)

          email_list_old=json_content["emaillist"]
          email_list_new=email_list_old+email_list
          print(email_list_new)
            
          json_content["emaillist"]=email_list_new
          with open(tempdir.name+'/competitionuserdata.json', 'w', encoding='utf-8') as f:
              json.dump(json_content, f, ensure_ascii=False, indent=4)

          aws_client['client'].upload_file(
                tempdir.name+"/competitionuserdata.json", api_bucket, model_id +"/"+submission_type+"/competitionuserdata.json"
            )
     
          return "Success: Your competition participant access list is now updated."
      elif update_type=="Remove":
          import json  
          import tempfile
          tempdir = tempfile.TemporaryDirectory()
    
          aws_client['resource']
          content_object = aws_client['resource'].Object(bucket_name=api_bucket, key=model_id +"/"+submission_type+"/competitionuserdata.json")
          file_content = content_object.get()['Body'].read().decode('utf-8')
          json_content = json.loads(file_content)

          email_list_old=json_content["emaillist"]
          email_list_new=list(set(list(email_list_old)) - set(email_list))
          print(email_list_new)
            
          json_content["emaillist"]=email_list_new
          with open(tempdir.name+'/competitionuserdata.json', 'w', encoding='utf-8') as f:
              json.dump(json_content, f, ensure_ascii=False, indent=4)

          aws_client['client'].upload_file(
                tempdir.name+"/competitionuserdata.json", api_bucket, model_id +"/"+submission_type+"/competitionuserdata.json"
            )
          return "Success: Your competition participant access list is now updated."
      elif update_type=="Get":
          import json  
          import tempfile
          tempdir = tempfile.TemporaryDirectory()

          aws_client['resource']
          content_object = aws_client['resource'].Object(bucket_name=api_bucket, key=model_id +"/"+submission_type+"/competitionuserdata.json")
          file_content = content_object.get()['Body'].read().decode('utf-8')
          json_content = json.loads(file_content)
          return json_content['emaillist']
      else:
          return "Error: Check inputs and resubmit."


def _confirm_libraries_exist(requirements):
  requirements = requirements.split(",")
  for i in range(len(requirements)):
      requirements[i] = requirements[i].strip(" ")
      exists = requests.get("https://pypi.org/project/" + requirements[i]) 
      if exists.status_code == 404:
          try_again_message = ("The entered library '" + requirements[i] + "' was not found. "
                                "Please confirm and re-submit library name: ")
          requirements[i] = input(try_again_message)
          exists = requests.get("https://pypi.org/project/" + requirements[i])
  
          if exists.status_code == 404:
              error_message = ("ModuleNotFoundError: No module named '" + requirements[i] + "' found in the Python Package Index (PyPI). \n"
                               "Please double-check library name and try again.")
              return print(error_message)

  return



def _create_exampledata_json(model_type, exampledata_folder_filepath): 
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.avif', 
                        '.svg', '.webp', '.tif', '.bmp', '.jpe', '.jif', '.jfif',
                        '.jfi', 'psd', '.raw', '.arw', '.cr2', '.nrw', '.k25', '.eps']
    video_extensions = ['.avchd', '.avi', '.flv', '.mov', '.mkv', '.mp4', '.wmv']
    audio_extensions = ['.m4a', '.flac', '.mp3', '.mp4', '.wav', '.wma', '.aac']
     
    if any([model_type.lower() == "tabular", model_type.lower() == "timeseries", model_type.lower() == "text"]):
        #confirm data type is data frame, try to convert if not [necessary for front end]
        import pandas as pd
        if isinstance(exampledata_folder_filepath, pd.DataFrame):
            pass
        else:
            exampledata_folder_filepath = pd.DataFrame(exampledata_folder_filepath)
            
        tabularjson = exampledata_folder_filepath.to_json(orient='split', index=False)
        
    
        with open('exampledata.json', 'w', encoding='utf-8') as f:
            json.dump({"exampledata": tabularjson, "totalfiles":1}, f, ensure_ascii=False, indent=4)

            return
        
    else:
        #Check file types & make list to convert 
        data = ""
        file_list = os.listdir(exampledata_folder_filepath)
        files_to_convert = []
        for i in range(len(file_list)):
            file_list[i] = exampledata_folder_filepath + "/" + file_list[i]
            root, ext = os.path.splitext(file_list[i])
            
            if not ext:
                ext = mimetypes.guess_extension(file_list[i])
                
            if model_type.lower() == "image" and ext in image_extensions:
                files_to_convert.append(file_list[i])
                    
            if model_type.lower() == "video" and ext in video_extensions:
                files_to_convert.append(file_list[i])
                
            if model_type.lower() == "audio" and ext in audio_extensions: 
                files_to_convert.append(file_list[i])     
        
            i += 1 
            if len(files_to_convert) == 5:
                break
    
        #base64 encode confirmed file list 
        for i in range(len(files_to_convert)):
            with open(files_to_convert[i], "rb") as current_file: 
                encoded_string = base64.b64encode(current_file.read())
                data = data + encoded_string.decode('utf-8') + ", "
                i += 1
    
        #build json
        with open('exampledata.json', 'w', encoding='utf-8') as f:
            json.dump({"exampledata": data[:-2], "totalfiles": len(files_to_convert)}, f, ensure_ascii=False, indent=4)
        
        return

__all__ = [
    take_user_info_and_generate_api,
    send_model_data_to_dyndb_and_return_api,
    model_to_api,
    create_competition,update_access_list
]
