import json
import os
import random
import boto3
import botocore
import tempfile
import zipfile
import shutil
import time
import functools
from zipfile import ZipFile, ZIP_STORED, ZipInfo



def create_prediction_api(my_credentials, model_filepath, unique_model_id, model_type, fileextension, categorical, labels, preprocessor_fileextension="default" ):
    AI_MODELSHARE_AccessKeyId = my_credentials["AI_MODELSHARE_AccessKeyId"]
    AI_MODELSHARE_SecretAccessKey = my_credentials["AI_MODELSHARE_SecretAccessKey"]
    region = my_credentials["region"]
    username = my_credentials["username"]
    bucket_name = my_credentials["bucket_name"]

  # Wait for 5 seconds to ensure aws iam user on user account has time to load into aws's system
    time.sleep(5)
    user_session = boto3.session.Session(aws_access_key_id=AI_MODELSHARE_AccessKeyId, aws_secret_access_key=AI_MODELSHARE_SecretAccessKey, region_name=region)
    cloud_layer = "arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
    #dill_layer ="arn:aws:lambda:us-east-1:517169013426:layer:dill:3"

  ## Update note:  dyndb data to add.  apiname. (include username too)


    api_name = 'modapi'+str(random.randint(1,1000000))
    account_number = user_session.client('sts').get_caller_identity().get('Account')

  # Update note:  change apiname in apijson from modapi890799 to randomly generated apiname?  or aimodelshare generic name?
    api_json = get_api_json()
    user_client = boto3.client('apigateway', aws_access_key_id=str(AI_MODELSHARE_AccessKeyId), aws_secret_access_key=str(AI_MODELSHARE_SecretAccessKey), region_name=str(region))

    response2 = user_client.import_rest_api(
        failOnWarnings=True,
    parameters={
        'endpointConfigurationTypes': 'REGIONAL'
    },
        body=api_json
    )
    api_id=response2['id'] 
    
 ## Update note:  dyndb data to add.  api_id and resourceid "Resource": "arn:aws:execute-api:us-east-1:517169013426:iu3q9io652/prod/OPTIONS/m"

    
    response3 = user_client.get_resources(restApiId=api_id)
    if str(response3['items'][0]).find("OPTIONS")>0:
        resource_id=response3['items'][0]['id']
    else:
        resource_id=response3['items'][1]['id']

    #write main handlers
    if  model_type=='Text' or model_type =='text':
  
      with open('./aimodelshare/main/1.txt', 'r') as txt_file: #this is for keras_image_color
        data = txt_file.read()
      with open('/tmp/main.py', 'w') as file:
        file.write(data.format(bucket_name,unique_model_id))
      
    elif model_type =='Image' and categorical== 'TRUE':
        with open('./aimodelshare/main/2.txt', 'r') as txt_file: #this is for keras_image_color
          data = txt_file.read()
        with open('/tmp/main.py', 'w') as file:
          file.write(data.format(bucket_name,unique_model_id,labels))
    elif model_type =='Image' and categorical== 'FALSE':
        with open('./aimodelshare/main/3.txt', 'r') as txt_file: #this is for keras_image_color
          data = txt_file.read()
        with open('/tmp/main.py', 'w') as file:
          file.write(data.format(bucket_name,unique_model_id))
    elif all([  model_type =='Tabular',categorical =='TRUE']):
        with open('./aimodelshare/main/4.txt', 'r') as txt_file: #this is for keras_image_color
          data = txt_file.read()
        with open('/tmp/main.py', 'w') as file:
          file.write(data.format(bucket_name,unique_model_id,labels))
    elif all([model_type=='Tabular', categorical =='FALSE']):
        with open('./aimodelshare/main/5.txt', 'r') as txt_file: #this is for keras_image_color
          data = txt_file.read()
        with open('/tmp/main.py', 'w') as file:
          file.write(data.format(bucket_name,unique_model_id))

    with zipfile.ZipFile('/tmp/archive.zip', 'a') as z:
        z.write('/tmp/main.py','main.py')

    os.remove("/tmp/main.py")
    #preprocessor upload


  # Upload lambda function zipfile to user's model file folder on s3
    try:
        s3_client = user_session.client('s3')  # This should go to developer's account from my account
        s3_client.upload_file('/tmp/archive.zip', bucket_name,  unique_model_id+"/"+'archivetest.zip')

    except Exception as e:
        print(e)

    os.remove("/tmp/archive.zip")
  # Create and/or update roles for lambda function you will create below
    lambdarole1={u'Version': u'2012-10-17', u'Statement': [{u'Action': u'sts:AssumeRole', u'Effect': u'Allow', u'Principal': {u'Service': u'lambda.amazonaws.com'}}]}
    lambdarolename='myService-dev-us-east-1-lambdaRole'

    roles = user_session.client('iam').list_roles()

    lambdafxnname='modfunction'+str(random.randint(1,1000000))
    lambdaauthfxnname='redisAccess'
    if str(roles['Roles']).find("myService-dev-us-east-1-lambdaRole")>0:

        response6_2 =  user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname+':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname+':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::'+bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
            RoleName=lambdarolename,
            )
    else:
        response6 = user_session.resource('iam').create_role(
      AssumeRolePolicyDocument=json.dumps(lambdarole1),
      Path='/',
      RoleName=lambdarolename,
        )
        response6_2 =  user_session.client('iam').put_role_policy(
      PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname+':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname+':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::'+bucket_name+'/*"],"Effect": "Allow"}]}',
      PolicyName='S3AccessandcloudwatchlogPolicy',
      RoleName=lambdarolename,
    )
    if str(roles['Roles']).find("myService-dev-us-east-1-lambdaRole")>0:

        response6_2 =  user_session.client('iam').put_role_policy(
        PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname+':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname+':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::'+bucket_name+'/*"],"Effect": "Allow"}]}',
        PolicyName='S3AccessandcloudwatchlogPolicy',
         RoleName=lambdarolename,
        )
    else:
        response6 = user_session.resource('iam').create_role(
        AssumeRolePolicyDocument=json.dumps(lambdarole1),
        Path='/',
        RoleName=lambdarolename,
        )
        response6_2 =  user_session.client('iam').put_role_policy(
        PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname+':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname+':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::'+bucket_name+'/*"],"Effect": "Allow"}]}',
        PolicyName='S3AccessandcloudwatchlogPolicy',
        RoleName=lambdarolename,
        )
    lambdaclient = user_session.client('lambda')
    layers =[cloud_layer]
    
    #if model_type=='sklearn_text' or  model_type=='keras_text' or model_type=='flubber_text' or model_type =='text':
      #layers.append(keras_layer)
      
    
    response6 = lambdaclient.create_function(FunctionName=lambdafxnname, Runtime='python3.6', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
    Code={
        'S3Bucket': bucket_name,
        'S3Key':  unique_model_id+"/"+'archivetest.zip'
          }, Timeout=10, MemorySize=512,Layers=layers) #ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE
    fxn_list = lambdaclient.list_functions()
    stmt_id='apigateway-prod-'+str(random.randint(1,1000000))
    if str(fxn_list.items()).find("redisAccess")>0:
        response7 = lambdaclient.add_permission(
          FunctionName=lambdaauthfxnname,
          StatementId=stmt_id,
          Action='lambda:InvokeFunction',
          Principal='apigateway.amazonaws.com',
          SourceArn='arn:aws:execute-api:us-east-1:'+account_number+":"+api_id+'/*/*',
        )
    else:
        #upload authfxn code first
        response7 = lambdaclient.add_permission(
        FunctionName=lambdaauthfxnname,
        StatementId=stmt_id,
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:'+account_number+":"+api_id+'/*/*',
      )
    ## Update note:  dyndb data to add.  lambdafxnname

    # change api name below?

    response7 = lambdaclient.add_permission(
        FunctionName=lambdafxnname,
        StatementId='apigateway-prod-2',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:'+account_number+":"+api_id+'/*/POST/m',
        )

    response8 = lambdaclient.add_permission(
        FunctionName=lambdafxnname,
        StatementId='apigateway-test-2',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:'+account_number+":"+api_id+'/*/POST/m',
        )

    # Create and or update lambda and apigateway gateway roles

    lambdarole2={u'Version': u'2012-10-17', u'Statement': [{u'Action': u'sts:AssumeRole', u'Principal': {u'Service': [u'lambda.amazonaws.com', u'apigateway.amazonaws.com']}, u'Effect': u'Allow', u'Sid': u''}]}
    if str(roles['Roles']).find("lambda_invoke_function_assume_apigw_role")>0:
        response11 = user_session.client('apigateway').put_integration(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='POST',
          type='AWS_PROXY',
          integrationHttpMethod='POST',
          uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:'+account_number+':function:'+lambdafxnname+'/invocations',
          credentials='arn:aws:iam::'+account_number+':role/lambda_invoke_function_assume_apigw_role')
        response11_1 = user_session.client('apigateway').put_integration(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='OPTIONS',
          type='MOCK',
          requestTemplates={
            'application/json': '{"statusCode": 200}'
          },
          integrationHttpMethod='OPTIONS',
          uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:'+account_number+':function:'+lambdafxnname+'/invocations',
          credentials='arn:aws:iam::'+account_number+':role/lambda_invoke_function_assume_apigw_role')
        response11_1B = user_session.client('apigateway').put_integration(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='OPTIONS',
          type='MOCK',
          requestTemplates={
            'application/json': '{"statusCode": 200}'
          }
          )
        response11_1C = user_session.client('apigateway').put_integration_response(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='OPTIONS',
          statusCode='200',
          responseParameters={
            'method.response.header.Access-Control-Allow-Headers': '\'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token\'',
            'method.response.header.Access-Control-Allow-Methods': '\'POST,OPTIONS\'',
            'method.response.header.Access-Control-Allow-Origin': '\'*\''
          },
          responseTemplates={
            'application/json': '{"statusCode": 200}'
          }
          )
        response11_2 =  user_session.client('iam').put_role_policy(
          PolicyDocument='{"Version":"2012-10-17","Statement":{"Effect":"Allow","Action":"lambda:InvokeFunction","Resource":"*"}}',
          PolicyName='invokelambda',
          RoleName='lambda_invoke_function_assume_apigw_role',
          )
    else:

        response10 = user_session.resource('iam').create_role(
          AssumeRolePolicyDocument=json.dumps(lambdarole2),
          Path='/',
          RoleName='lambda_invoke_function_assume_apigw_role',
        )
        response11 = user_session.client('apigateway').put_integration(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='POST',
          type='AWS_PROXY',
          integrationHttpMethod='POST',
          uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:'+account_number+':function:'+lambdafxnname+'/invocations',
          credentials='arn:aws:iam::'+account_number+':role/lambda_invoke_function_assume_apigw_role')
        response11_1 = user_session.client('apigateway').put_integration(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='OPTIONS',
          type='MOCK',
          requestTemplates={
            'application/json': '{"statusCode": 200}'
          },
          integrationHttpMethod='OPTIONS',
          uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:'+account_number+':function:'+lambdafxnname+'/invocations',
          credentials='arn:aws:iam::'+account_number+':role/lambda_invoke_function_assume_apigw_role')
        response11_1B = user_session.client('apigateway').put_integration(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='OPTIONS',
          type='MOCK',
          requestTemplates={
            'application/json': '{"statusCode": 200}'
          }
          )
        response11_1C = user_session.client('apigateway').put_integration_response(
          restApiId=api_id,
          resourceId=resource_id,
          httpMethod='OPTIONS',
          statusCode='200',
          responseParameters={
            'method.response.header.Access-Control-Allow-Headers': '\'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token\'',
            'method.response.header.Access-Control-Allow-Methods': '\'POST,OPTIONS\'',
            'method.response.header.Access-Control-Allow-Origin': '\'*\''
          },
          responseTemplates={
            'application/json': '{"statusCode": 200}'
          }
          )
        response11_2 =  user_session.client('iam').put_role_policy(
          PolicyDocument='{"Version":"2012-10-17","Statement":{"Effect":"Allow","Action":"lambda:InvokeFunction","Resource":"*"}}',
          PolicyName='invokelambda',
          RoleName='lambda_invoke_function_assume_apigw_role',
          )
    response12_1 = user_client.update_rest_api(
      restApiId =api_id,
      patchOperations=[{
      "op":"replace",
      "path":"/policy",
      "value": '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": "*","Action": "execute-api:Invoke","Resource": "arn:aws:execute-api:'+region+':'+account_number+':'+api_id+'/prod/OPTIONS/m"}]}'
      },]
      )
    response12 = user_session.client('apigateway').create_deployment(
        restApiId=api_id,
        stageName='prod')
    


    result='https://'+api_id+ '.execute-api.'+region+'.amazonaws.com/prod/m'

    return {"statusCode": 200,
    "headers": {
    "Access-Control-Allow-Origin" : "*",
    "Access-Control-Allow-Credentials": True
    },
    "body": json.dumps(result)}



def get_api_json():
  apijson='''{
                "openapi": "3.0.1",
                "info": {
                  "title": "modapi36146",
                  "description": "This is a copy of my first API",
                  "version": "2020-05-12T21:25:38Z"
                },
                "servers": [
                  {
                    "url": "https://zm2yl9jj88.execute-api.us-east-1.amazonaws.com/{basePath}",
                    "variables": {
                      "basePath": {
                        "default": "/prod"
                      }
                    }
                  }
                ],
                "paths": {
                  "/m": {
                    "post": {
                      "responses": {
                        "200": {
                          "description": "200 response",
                          "headers": {
                            "Access-Control-Allow-Origin": {
                              "schema": {
                                "type": "string"
                              }
                            }
                          },
                          "content": {
                            "application/json": {
                              "schema": {
                                "$ref": "#/components/schemas/outputmodel"
                              }
                            }
                          }
                        }
                      },
                "security": [
                  {
                    "redisAccessauth": []
                  }
                ]
                    },
                    "options": {
                      "responses": {
                        "200": {
                          "description": "200 response",
                          "headers": {
                            "Access-Control-Allow-Origin": {
                              "schema": {
                                "type": "string"
                              }
                            },
                            "Access-Control-Allow-Methods": {
                              "schema": {
                                "type": "string"
                              }
                            },
                            "Access-Control-Allow-Headers": {
                              "schema": {
                                "type": "string"
                              }
                            }
                          },
                          "content": {
                            "application/json": {
                              "schema": {
                                "$ref": "#/components/schemas/Empty"
                              }
                            }
                          }
                        }
                      },
                "security": [
                  {
                    "testauth2frombotonext8": []
                  }
                ]
                    }
                  }
                },
                "components": {
                  "schemas": {
                    "Empty": {
                      "title": "Empty Schema",
                      "type": "object"
                    },
                    "outputmodel": {
                      "title": "Output",
                      "type": "object",
                      "properties": {
                        "body": {
                          "type": "string"
                        }
                      }
                    }
                  },
            "securitySchemes": {
              "redisAccessauth": {
                "type": "apiKey",
                "name": "authorizationToken",
                "in": "header",
                "x-amazon-apigateway-authtype": "custom",
                "x-amazon-apigateway-authorizer": {
                  "authorizerUri": "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:517169013426:function:redisAccess/invocations",
                  "authorizerResultTtlInSeconds": 0,
                  "type": "token"
                }
              }
            }
                },
           "x-amazon-apigateway-policy": {
            "Version": "2012-10-17",
            "Statement": [
              {
                "Effect": "Deny",
                "Principal": "*",
                "Action": "execute-api:Invoke",
                "Resource": "arn:aws:execute-api:us-east-1:517169013426:0eq0w2nmmb/*",
                "Condition": {
                  "IpAddress": {
                    "aws:SourceIp": "11"
                  }
                }
              }
            ]
          }
              }'''
  return apijson

__all__ = [
    create_prediction_api,
  get_api_json
  ,
]
