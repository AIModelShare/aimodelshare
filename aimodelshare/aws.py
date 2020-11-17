import os
import boto3
import botocore
import requests

from aimodelshare.exceptions import AuthorizationError, AWSAccessError


def get_aws_token(user_name, user_pass):
    config = botocore.config.Config(signature_version=botocore.UNSIGNED)

    provider_client = boto3.client(
        "cognito-idp", region_name="us-east-2", config=config
    )

    try:
        response = provider_client.initiate_auth(
            ClientId="57tnil9teheh8ic5ravno5c7ln",
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
        "Authorization": token["token"],
    }

    response = requests.post(
        "https://bbfgxopv21.execute-api.us-east-1.amazonaws.com/dev/todos",
        json=kwargs,
        headers=headers_with_authentication,
    )

    if response.status_code != 200:
        return (
            None,
            AWSAccessError(
                "Could not execute function on the lambda."
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
