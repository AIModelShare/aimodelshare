import boto3
import botocore
import os
import requests
import uuid
import json
import math
import time
import datetime

from aimodelshare.tools import form_timestamp
from aimodelshare.exceptions import AuthorizationError, AWSAccessError, AWSUploadError
from aimodelshare.aws import get_s3_iam_client
from aimodelshare.bucketpolicy import _custom_s3_policy



def get_jwt_token(username, password):

    config = botocore.config.Config(signature_version=botocore.UNSIGNED)

    provider_client = boto3.client(
      "cognito-idp", region_name="us-east-2", config=config
    )

    try:
      # Get JWT token for the user
      response = provider_client.initiate_auth(
        ClientId='57tnil9teheh8ic5ravno5c7ln',
        AuthFlow='USER_PASSWORD_AUTH',
        AuthParameters={'USERNAME': username,'PASSWORD': password})

      jwt_aws_token={"username": username, "authorizationToken": response["AuthenticationResult"]["IdToken"]}

    except :
      err = "Username or password does not exist.  Please enter new username or password."+"\n"
      err += "Sign up at AImodelshare.com/register."
      raise AuthorizationError(err)

    return jwt_aws_token



def create_user_getkeyandpassword(jwt_aws_token, aws_key, aws_password, region):

    username = jwt_aws_token["username"]
    returned_jwt_token = jwt_aws_token["authorizationToken"]
    s3, iam, region = get_s3_iam_client(aws_key, aws_password, region)

    #create s3 bucket and iam user
    now = datetime.datetime.now()
    year = datetime.date.today().year
    ts = form_timestamp(time.time())
    bucket_name = 'aimodelshare' + username.lower()
    master_name = 'aimodelshare' + username.lower()
    try:
      #check if bucket exist and you have access
      s3.meta.client.head_bucket(Bucket=bucket_name)
      
    except :
    
      try :
        #bucket doesnot exist then create it
        bucket = s3["client"].create_bucket(ACL ='private',Bucket=bucket_name)

      except :
        #bucket exists but you have no access
        #add versioning
        s3_resource = boto3.resource('s3')
        version =0 
        
        for bucket in s3_resource.buckets.all(): 
          if bucket.name.startswith(bucket_name+'-'):
             version+=1

        for i in range(1,version):
          #check if any version of bucket name exists in the user's buckets
          if bucket_name+'-'+str(i) in s3["client"].buckets.all():
            bucket_name = bucket_name+'-'+str(i)

        if bucket_name == master_name:
          bucket_name= bucket_name+'-'+str(version)

      my_policy = _custom_s3_policy(bucket_name)
    #sub_bucket = 'aimodelshare' + username.lower() + ts.replace("_","")
    iam_username = 'AI_MODELSHARE_' + ts

    try:
      
      iam["client"].create_user(
        UserName = iam_username
      )
      iam_response = iam["client"].create_access_key(
        UserName=iam_username
      )
    except Exception as err:
      raise err
    AI_MODELSHARE_AccessKeyId = iam_response['AccessKey']['AccessKeyId']
    AI_MODELSHARE_SecretAccessKey = iam_response['AccessKey']['SecretAccessKey']
    
    #create and attach policy for the s3 bucket
    my_managed_policy = _custom_s3_policy(bucket_name)
    policy_name = 'temporaryaccessAImodelsharePolicy' + str(uuid.uuid1().hex)
    policy_response = iam["client"].create_policy(
      PolicyName = policy_name,
      PolicyDocument = json.dumps(my_managed_policy)
    )
    policy_arn = policy_response['Policy']['Arn']
    user = iam["resource"].User(iam_username)
    user.attach_policy(
          PolicyArn=policy_arn
      )
    
    return {"now":now,"iamusername":iam_username, "username":username,"AI_MODELSHARE_AccessKeyId":AI_MODELSHARE_AccessKeyId,
            "AI_MODELSHARE_SecretAccessKey":AI_MODELSHARE_SecretAccessKey,"returned_jwt_token":returned_jwt_token,"aws_key":aws_key,
            "aws_password":aws_password,"region":region,"policy_arn":policy_arn,"policy_name":policy_name,"bucket_name":bucket_name}




__all__ = [
    get_jwt_token,
    create_user_getkeyandpassword,
]
