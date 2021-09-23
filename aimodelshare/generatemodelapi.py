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
import tempfile
import sys
import base64
import mimetypes
import numpy as np
from aimodelshare.tools import extract_varnames_fromtrainingdata, _get_extension_from_filepath
from aimodelshare.aws import get_s3_iam_client, run_function_on_lambda
from aimodelshare.bucketpolicy import _custom_upload_policy
from aimodelshare.exceptions import AuthorizationError, AWSAccessError, AWSUploadError
from aimodelshare.api import create_prediction_api
from aimodelshare.api import get_api_json
from aimodelshare.modeluser import create_user_getkeyandpassword
from aimodelshare.preprocessormodules import upload_preprocessor
from aimodelshare.model import _get_predictionmodel_key, _extract_model_metadata
from aimodelshare.data_sharing.share_data import share_data_codebuild
from aimodelshare.containerization import clone_base_image


def take_user_info_and_generate_api(model_filepath, model_type, categorical,labels, preprocessor_filepath,custom_libraries, requirements, exampledata_json_filepath, repo_name, image_tag):
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
    s3["client"].create_bucket(
        ACL='private',
        Bucket=os.environ.get("BUCKET_NAME"))
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
    
    apiurl = create_prediction_api(model_filepath, unique_model_id,
                                   model_type, categorical, labels,api_id,custom_libraries, requirements, repo_name, image_tag)

    finalresult = [apiurl["body"], apiurl["statusCode"],
                   now, unique_model_id, os.environ.get("BUCKET_NAME"), input_shape]
    return finalresult


def send_model_data_to_dyndb_and_return_api(api_info, private, categorical, preprocessor_filepath,
                                            aishare_modelname, aishare_modeldescription, aishare_modelevaluation, model_type,
                                            aishare_tags, aishare_apicalls, exampledata_json_filepath,variablename_and_type_data="default"):
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
    bodydata = {"id": int(math.log(1/((time.time()*1000000)))*100000000000000),
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
                "input_shape": input_shape
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

    # Build output {{{
    final_message = ("\nYou can now use your Model Playground.\n\n"
                     "Follow this link to explore your Model Playground's functionality\n"
                     "You can make predictions with the Dashboard and access example code from the Programmatic tab.\n")
    web_dashboard_url = ("https://www.modelshare.org/detail/"+ response_string)
    
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


def model_to_api(model_filepath, model_type, private, categorical, y_train, preprocessor_filepath, custom_libraries="FALSE", example_data=None, image="aimodelshare_base_image:v3", base_image_api_endpoint="https://vupwujn586.execute-api.us-east-1.amazonaws.com/dev/copybasetouseracct", update=False):
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

    repo_name, image_tag = image.split(':')
    if model_type=="tabular":
        image_tag="tabular"
    else:
        pass
    
    response = clone_base_image(user_session, repo_name, image_tag, "517169013426", base_image_api_endpoint, update)
    if(response["Status"]==0):
        print(response["Success"])
        return

    print("We need some information about your model before we can build your REST API and interactive Model Playground.")
    print("   ")

    requirements = ""
    if(any([custom_libraries=='TRUE',custom_libraries=='true'])):
        requirements = input("Enter all required Python libraries you need at prediction runtime (separated with commas):")
        _confirm_libraries_exist(requirements)
        
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
        if example_data is None:
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
        model_filepath, model_type, categorical, labels,preprocessor_filepath,custom_libraries, requirements, exampledata_json_filepath, repo_name, image_tag)

    ### Progress Update #5/6 {{{
    sys.stdout.write('\r')
    sys.stdout.write("[=================================    ] Progress: 90% - Finishing web dashboard...                           ")
    sys.stdout.flush()
    # }}}
    
    print_api_info = send_model_data_to_dyndb_and_return_api(
        api_info, private, categorical,preprocessor_filepath, aishare_modelname,
        aishare_modeldescription, aishare_modelevaluation, model_type,
        aishare_tags, aishare_apicalls, exampledata_json_filepath,variablename_and_type_data)
    
    return api_info[0]

def create_competition(apiurl, data_directory, y_test, generate_credentials_file = True):
    """
    Creates a model competition for a deployed prediction REST API
    Inputs : 2
    Output : Submit credentials for model competition
    
    ---------
    Parameters
    
    apiurl: string
            URL of deployed prediction API 
    
    y_test :  y labels for test data 
            [REQUIRED] for eval metrics
            expects a one hot encoded y test data format
            
    data_directory : folder storing training data and test data (excluding Y test data)
    generate_credentials_file (OPTIONAL): Default is True
                                          Function will output .txt file with new credentials
    ---------
    Returns
    finalresultteams3info : Submit_model credentials with access to S3 bucket
    (api_id)_credentials.txt : .txt file with submit_model credentials,
                                formatted for use with set_credentials() function 
    """

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
    s3["client"].upload_file(ytest_path, os.environ.get("BUCKET_NAME"),  model_id + "/ytest.pkl")

    # Reset user policy
    create_user_getkeyandpassword()
    # Detach & Delete current policy  
    policy_response = iam["client"].get_policy(
        PolicyArn=os.environ.get("POLICY_ARN")
    )
    user_policy = iam["resource"].UserPolicy(
        os.environ.get("IAM_USERNAME"), policy_response['Policy']['PolicyName'])
    response = iam["client"].detach_user_policy(
        UserName= os.environ.get("IAM_USERNAME"),
        PolicyArn=os.environ.get("POLICY_ARN")
    )

    # Create & attach new policy that only allows file upload to bucket
    policy = iam["resource"].Policy(os.environ.get("POLICY_ARN"))
    response = policy.delete()
    s3upload_policy = _custom_upload_policy(os.environ.get("BUCKET_NAME"), 
                                            model_id)
    s3uploadpolicy_name = 'temporaryaccessAImodelsharePolicy' + \
        str(uuid.uuid1().hex)
    s3uploadpolicy_response = iam["client"].create_policy(
        PolicyName=s3uploadpolicy_name,
        PolicyDocument=json.dumps(s3upload_policy)
    )
    user = iam["resource"].User(os.environ.get("IAM_USERNAME"))
    response = user.attach_policy(
        PolicyArn=s3uploadpolicy_response['Policy']['Arn']
    )
    
    # get api_id from apiurl, generate txt file name
    api_url_trim = apiurl.split('https://')[1]
    api_id = api_url_trim.split(".")[0]
    txt_file_name = api_id+"_credentials.txt"

    aishare_competitionname = input("Enter competition name:")
    aishare_competitiondescription = input("Enter competition description:")
    aishare_datadescription = input(
        "Enter data description (i.e.- filenames denoting training and test data, file types, and any subfolders where files are stored):")
    
    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                          aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                         region_name=os.environ.get("AWS_REGION"))
    account_number = user_session.client(
        'sts').get_caller_identity().get('Account')

    datauri=share_data_codebuild(account_number,os.environ.get("AWS_REGION"),data_directory)

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
                'dataecruri':datauri['ecr_uri']}
    
    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    requests.post("https://o35jwfakca.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)

    
    #Format output text
    formatted_userpass = ('[aimodelshare_creds] \n'
                'username = "Your_Username_Here" \n'
                'password = "Your_Password_Here"\n\n')

    formatted_new_creds = ("#Credentials for Competition: " + api_id + "\n"
                '[submit_model:"' + apiurl + '"]\n'
                'AWS_ACCESS_KEY_ID = "' + os.environ.get("AI_MODELSHARE_ACCESS_KEY_ID") + '"\n'
                'AWS_SECRET_ACCESS_KEY = "' + os.environ.get("AI_MODELSHARE_SECRET_ACCESS_KEY") +'"\n'
                'AWS_REGION = "' + os.environ.get("AWS_REGION") + '"\n')
    
    final_message = ("\n Success! Model competition created. \n\n"
                "Your team members can now make use of the following functions: \n"
                "submit_model() to submit new models to the competition leaderboard. \n"
                "download_data('"+datauri['ecr_uri']+"') to download your competition data.  \n\n"
                "You may update your prediction API runtime model with the update_runtime_model() function.\n\n"
                "To upload new models and/or preprocessors to this API, team members should use \n"
                "the following credentials:\n\n" + formatted_new_creds + "\n"
                "(This aws key/password combination limits team members to file upload access only.)\n\n")
  
    file_generated_message = ("These credentials have been saved as: " + txt_file_name + ".")

    # Generate .txt file with new credentials 
    if generate_credentials_file == True:
        final_message = final_message + file_generated_message

        f= open(txt_file_name,"w+")
        f.write(formatted_userpass + formatted_new_creds)
        f.close()



    return print(final_message)



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
    create_competition,
]
