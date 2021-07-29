import boto3
import os
import requests
import uuid
import json
import math
import time
import datetime
import onnx
import tempfile
import shutil
import sys
import pickle
import tempfile

from aimodelshare.tools import extract_varnames_fromtrainingdata, _get_extension_from_filepath
from aimodelshare.aws import get_s3_iam_client, run_function_on_lambda
from aimodelshare.bucketpolicy import _custom_upload_policy
from aimodelshare.exceptions import AuthorizationError, AWSAccessError, AWSUploadError
from aimodelshare.api import create_prediction_api
from aimodelshare.api import get_api_json

from aimodelshare.preprocessormodules import upload_preprocessor
from aimodelshare.model import _get_predictionmodel_key, _extract_model_metadata

# what is private here
# what if deployment_dir is not present? Not handled yet
# Return section and private parameter documentation left
# output_list_exampledata list being accessed by which key in response?
# create file_objects in temp_dir, abstract from user
# convert example output from list to json, in frontend display value of key of result

def deploy_custom_lambda(input_json_exampledata, output_json_exampledata, lambda_filepath, deployment_dir, private, custom_libraries):

    """
        Deploys an AWS Lambda function based on the predict() function specified in the lambda_filepath .py file
        Inputs : 6
        Output : Information about the deployed API

        -----------

        Parameters :

        input_json_exampledata : JSON object [REQUIRED]
                                 JSON object representing the structure of input that Lambda expects to receive

        output_json_exampledata : List of element(s) [REQUIRED]
                                  List of element(s) representing the output that the Lambda will return
        
        lambda_filepath : String [REQUIRED]
                          Expects relative/absolute path to the .py file containing the predict() function,
                          imports to all other custom libraries that are defined by the user and used by the
                          predict() function can be placed in the deployment_dir directory
        
        deployment_dir : String
                         Expects relative/absolute path to the directory containing all the files being used by the
                         predict() function in the lambda_filepath .py file

        private : string

        custom_libraries : String of libraries ("library_1,library_2")
                           Expects a list of strings denoting libraries required for Lambda to work
                           Strings must be libraries present in PyPi
                           Installation will follow pattern - pip install <library_name>

        -----------

        Returns

        api_info : prints statements with generated live prediction API details
                   also prints steps to update the model submissions by the user/team
    """

    temp_dir = tempfile.gettempdir()

    # if 'file_objects' is not the name of deployment directory deployment_dir, 'file_objects' directory
    # is created and contents of the deployment_dir directory are copied to 'file_objects' directory
    if deployment_dir != 'file_objects':
        if os.path.exists('file_objects'):
            shutil.rmtree('file_objects')
        shutil.copytree(deployment_dir, 'file_objects')

    # if 'custom_lambda.py' is not the name of the custom lambda .py file lambda_filepath, 'custom_lambda.py' file
    # is created and contents of lambda_filepath .py file is written into 'custom_lambda.py'
    if lambda_filepath != 'custom_lambda.py':
        with open(lambda_filepath, 'r') as in_f:
            with open('custom_lambda.py', 'w') as out_f:
                out_f.write(in_f.read())

    # create json and upload to API folder in S3 for displaying
    json_exampledata = {
        "input_json_exampledata": input_json_exampledata,
        "output_json_exampledata": output_json_exampledata
    }

    with open(os.path.join(temp_dir, 'exampledata.json'), 'w') as f:
        json.dump(json_exampledata, f)

    aws_access_key_id = str(os.environ.get("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = str(os.environ.get("AWS_SECRET_ACCESS_KEY"))
    region_name = str(os.environ.get("AWS_REGION"))

    ### COMMENTS - TO DO
    api_json= get_api_json()        # why is this required
    user_client = boto3.client(     # creating apigateway client
        'apigateway',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    response2 = user_client.import_rest_api(        # what is being imported
        failOnWarnings = True,
        parameters = {'endpointConfigurationTypes': 'REGIONAL'},
        body = api_json
    )
    ###

    start = datetime.datetime.now()

    api_id = response2['id']
    now = datetime.datetime.now()
    s3, iam, region = get_s3_iam_client(aws_access_key_id, aws_secret_access_key, region_name)

    s3["client"].create_bucket(
        Bucket=os.environ.get("BUCKET_NAME")
    )

    apiurl = create_prediction_api(None, str(api_id), 'custom', 'FALSE', [], api_id, "TRUE", custom_libraries)

    print("We need some information about your model before we can generate your API.\n")
    aishare_modelname = input("Name your model: ")
    aishare_modeldescription = input("Describe your model: ")
    aishare_modelevaluation = input("Describe your model's performance: ")
    aishare_tags = input("Enter comma-separated search categories for your model: ")
    aishare_apicalls = 0
    print('')
    # unpack user credentials
    unique_model_id = str(api_id)
    bucket_name = os.environ.get("BUCKET_NAME")

    # why is this being done here, can it not be abstracted

    categorical="FALSE" # categorical is being used where, for what
    bodydata = {
        "id": int(math.log(1/((time.time()*1000000)))*100000000000000),
        "unique_model_id": unique_model_id,
        "apideveloper": os.environ.get("username"),  # change this to first and last name
        "apimodeldescription": aishare_modeldescription,
        "apimodelevaluation": aishare_modelevaluation,
        "apimodeltype": 'custom',
        # getting rid of extra quotes that screw up dynamodb string search on apiurls
        "apiurl": apiurl['body'].strip('\"'),
        "bucket_name": bucket_name,
        "version": 1,
        "modelname": aishare_modelname,
        "tags": aishare_tags,
        "Private": private,
        "Categorical": categorical,
        "delete": "FALSE",
    }

    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    requests.post("https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)

    end = datetime.datetime.now()    # end timer
    difference = (end - start).total_seconds()
    finalresult2 = "Your AI Model Share API was created in " + \
        str(int(difference)) + " seconds." + " API Url: " + apiurl['body']
    s3, iam, region = get_s3_iam_client(os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"), os.environ.get("AWS_REGION"))
    policy_response = iam["client"].get_policy(
        PolicyArn=os.environ.get("POLICY_ARN")
    )
    user_policy = iam["resource"].UserPolicy(
        os.environ.get("IAM_USERNAME"), policy_response['Policy']['PolicyName'])
    response = iam["client"].detach_user_policy(
        UserName= os.environ.get("IAM_USERNAME"),
        PolicyArn=os.environ.get("POLICY_ARN")
    )
    # add new policy that only allows file upload to bucket
    policy = iam["resource"].Policy(os.environ.get("POLICY_ARN"))
    response = policy.delete()
    s3upload_policy = _custom_upload_policy(bucket_name, unique_model_id)
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
    finalresultteams3info = "Your team members can submit improved models to your prediction api using the update_model_version() function. \nTo upload new models and/or preprocessors to this model team members should use the following awskey/password/region:\n\n aws_key = " + \
        os.environ.get("AI_MODELSHARE_ACCESS_KEY_ID") + ", aws_password = " + os.environ.get("AI_MODELSHARE_SECRET_ACCESS_KEY") + " region = " + \
        os.environ.get("AWS_REGION") +".  \n\nThis aws key/password combination limits team members to file upload access only."
    api_info = finalresult2+"\n"+finalresultteams3info
    
    return print(api_info)

__all__ = [
    deploy_custom_lambda
]