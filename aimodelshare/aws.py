import os
import boto3
import botocore
import requests
import json
from aimodelshare.exceptions import AuthorizationError, AWSAccessError


def set_credentials(credential_file=None, type="submit_model", apiurl="apiurl",manual=False):
  #TODO:
  #1. When user runs set_credentials() with no args the function should immediately use the manual entry approach.
  #2.
  
  import os
  import getpass
  flag = False
  set_creds = []

  if any([manual == True,credential_file==None]):
    user = getpass.getpass(prompt="AI Modelshare Username:")
    os.environ["username"] = user
    pw = getpass.getpass(prompt="AI Modelshare Password:")
    os.environ["password"] = pw
    set_creds.extend(["username", "password"])
  
  else: 
    f = open(credential_file)

    for line in f:
      if "aimodelshare_creds" in line or "AIMODELSHARE_CREDS" in line:
        for line in f:
          if line == "\n":
            break
          try:
            value = line.split("=", 1)[1].strip()
            value = value[1:-1]
            key = line.split("=", 1)[0].strip()
            os.environ[key.lower()] = value
            set_creds.append(key.lower())
    
          except LookupError: 
            print(* "Warning: Review format of", credential_file, ". Format should be variablename = 'variable_value'")
            break
  
  if any([manual == True,credential_file==None]):
    flag = True
    access_key = getpass.getpass(prompt="AWS_ACCESS_KEY_ID:")
    os.environ["AWS_ACCESS_KEY_ID"] = access_key

    secret_key = getpass.getpass(prompt="AWS_SECRET_ACCESS_KEY:")
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key

    region = getpass.getpass(prompt="AWS_REGION:")
    os.environ["AWS_REGION"] = region
    set_creds.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])

  else:  
    f = open(credential_file)
    for line in f:
      if (apiurl in line) and ((type in line) or (type.upper() in line)):
        flag = True
        for line in f:
          if line == "\n":
            break
          try:
            value = line.split("=", 1)[1].strip()
            value = value[1:-1]
            key = line.split("=", 1)[0].strip()
            os.environ[key.upper()] = value
            set_creds.append(key.upper())
          except LookupError: 
            print(* "Warning: Review format of", credential_file, ". Format should be variablename = 'variable_value'.")
            break
  if not flag: 
    return "Error: apiurl and/or type not found in "+str(credential_file)+". Please correct entries and resubmit."

  try:
    f.close()    
  except:
    pass
  success = "Your "+type+" credentials for "+apiurl+"have been set successfully."
  return success



def get_aws_token(user_name, user_pass):
    config = botocore.config.Config(signature_version=botocore.UNSIGNED)

    provider_client = boto3.client(
        "cognito-idp", region_name="us-east-2", config=config
    )

    try:
        response = provider_client.initiate_auth(
            ClientId="7ptv9f8pt36elmg0e4v9v7jo9t",
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": user_name, "PASSWORD": user_pass},
        )

    except Exception as err:
        raise AuthorizationError("Could not authorize user. " + str(err))

    return {"username": user_name,"password": user_pass, "token": response["AuthenticationResult"]["IdToken"]}


def get_aws_client(aws_key=None, aws_secret=None, aws_region=None):
    key = aws_key if aws_key is not None else os.environ.get("AWS_ACCESS_KEY_ID")
    secret = (
        aws_secret
        if aws_secret is not None
        else os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    region = (
        aws_region if aws_region is not None else os.environ.get("AWS_DEFAULT_REGION")
    )

    if any([key is None, secret is None, region is None]):
        raise ValueError("Invalid arguments")

    usersession = boto3.session.Session(
        aws_access_key_id=key, aws_secret_access_key=secret, region_name=region,
    )

    s3client = usersession.client("s3")
    s3resource = usersession.resource("s3")

    return {"client": s3client, "resource": s3resource}


def get_s3_iam_client(aws_key=None,aws_password=None, aws_region=None):

  key = aws_key if aws_key is not None else os.environ.get("AWS_ACCESS_KEY_ID")
  password = (
        aws_password
        if aws_password is not None
        else os.environ.get("AWS_SECRET_ACCESS_KEY"))
  region = (
        aws_region if aws_region is not None else os.environ.get("AWS_DEFAULT_REGION"))

  if any([key is None, password is None, region is None]):
        raise AuthorizationError("Please set your aws credentials before creating your prediction API.")

  usersession = boto3.session.Session(
        aws_access_key_id=key, aws_secret_access_key=password, region_name=region,
    )

  s3_client = boto3.client('s3')
  s3_resource = boto3.resource('s3')
  iam_client = boto3.client('iam')
  iam_resource = boto3.resource('iam')

  s3 = {"client":s3_client,"resource":s3_resource}
  iam = {"client":iam_client,"resource":iam_resource}

  return s3,iam,region


def run_function_on_lambda(url, token, **kwargs):
    kwargs["apideveloper"] = token["username"]
    kwargs["apiurl"] = url

    headers_with_authentication = {
        "content-type": "application/json",
        "authorizationToken": token["token"],
    }

    response = requests.post(
        "https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
        json=kwargs,
        headers=headers_with_authentication,
    )

    if response.status_code != 200:
        return (
            None,
            AWSAccessError(
                "Error:"
                + "Please make sure your api url and token are correct."
            ),
        )

    return response, None


def get_token(username, password):
      #get token for access to prediction lambas or to submit predictions to generate model evaluation metrics
      tokenstring = '{\"username\": \"$usernamestring\", \"password\": \"$passwordstring\"}'
      from string import Template
      t = Template(tokenstring)
      newdata = t.substitute(
          usernamestring=username, passwordstring=password)
      api_url='https://xgwe1d6wai.execute-api.us-east-1.amazonaws.com/dev' 
      headers={ 'Content-Type':'application/json'}
      token =requests.post(api_url,headers=headers,data=json.dumps({"action": "login", "request":newdata}))
      return token.text



__all__ = [
    get_aws_token,
    get_aws_client,
    run_function_on_lambda,
    get_s3_iam_client,
]
