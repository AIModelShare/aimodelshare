import time
import sys
import os
import shutil
import random
import tempfile
import functools
import json
import requests
import math
from zipfile import ZipFile
from string import Template

import shortuuid
import boto3
import botocore

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from . import main  # relative-import the *package* containing the templates

from .utils import *

class create_prediction_api_class():

    def __init__(self, model_filepath, unique_model_id, model_type, categorical, labels, apiid, custom_libraries, requirements, repo_name="", image_tag="", memory=None, timeout=90, pyspark_support=False):

        #####
        self.user_session = boto3.session.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
            region_name=os.environ.get("AWS_REGION")
        )
        self.sts_client = self.user_session.client("sts")
        self.account_id = self.sts_client.get_caller_identity()["Account"]
        #####

        from .aws_client import AWSClient
        self.aws_client = AWSClient(self.user_session)
                
        #####
        self.account_id = self.account_id
        self.unique_model_id = unique_model_id
        self.categorical = categorical
        self.model_type = model_type
        self.labels = labels
        self.apiid = apiid
        self.custom_libraries = custom_libraries
        self.requirements = requirements
        self.repo_name = repo_name
        self.image_tag = image_tag
        self.memory = memory
        self.timeout = timeout
        self.pyspark_support = pyspark_support
        self.region = os.environ.get("AWS_REGION")
        self.bucket_name = os.environ.get("BUCKET_NAME")
        self.python_runtime = 'python3.7'
        #####

        self.model_type = self.model_type.lower()

        if(self.categorical == "TRUE"):
            self.categorical = True
        elif(self.categorical == "FALSE"):
            self.categorical = False

        if(self.custom_libraries == "TRUE"):
            self.custom_libraries = True
        elif(self.custom_libraries == "FALSE"):
            self.custom_libraries = False

        self.memory_model_mapping = {
            "tabular": 1024,
            "text": 1024,
            "image": 1024,
            "video": 1024,
            "custom":1024
        }

        self.timeout_model_mapping = {
            "tabular": 90,
            "text": 90,
            "image": 90,
            "video": 90,
            "custom": 90
        }

        self.eval_layer_map = {
            "us-east-1": "arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6",
            "us-east-2": "arn:aws:lambda:us-east-2:517169013426:layer:eval_layer_test:5",
            "us-west-1": "arn:aws:lambda:us-west-1:517169013426:layer:eval_layer_test:1",
            "us-west-2": "arn:aws:lambda:us-west-2:517169013426:layer:eval_layer_test:1",
            "eu-west-1": "arn:aws:lambda:eu-west-1:517169013426:layer:eval_layer_test:1",
            "eu-west-2": "arn:aws:lambda:eu-west-2:517169013426:layer:eval_layer_test:1",
            "eu-west-3": "arn:aws:lambda:eu-west-3:517169013426:layer:eval_layer_test:1"
        }

        self.auth_layer_map = {
            "us-east-1": "arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2",
            "us-east-2": "arn:aws:lambda:us-east-2:517169013426:layer:aimsauth_layer:9",
            "us-west-1": "arn:aws:lambda:us-west-1:517169013426:layer:aimsauth_layer:1",
            "us-west-2": "arn:aws:lambda:us-west-2:517169013426:layer:aimsauth_layer:1",
            "eu-west-1": "arn:aws:lambda:eu-west-1:517169013426:layer:aimsauth_layer:1",
            "eu-west-2": "arn:aws:lambda:eu-west-2:517169013426:layer:aimsauth_layer:1",
            "eu-west-3": "arn:aws:lambda:eu-west-3:517169013426:layer:aimsauth_layer:1"
        }
        
        try:
            onnx_size = math.ceil(os.path.getsize(model_filepath)/(1024*1024))
        except:
            onxx_size = 500

        self.temp_dir_file_deletion_list = ['archive2.zip', 'archive3.zip', 'archive.zip', 'archivetest.zip', 'archiveeval.zip', 'archiveauth.zip', 'main.py', 'ytest.pkl']
        self.memory = self.memory_model_mapping[self.model_type] if self.memory==None else memory
        self.timeout = self.timeout_model_mapping[self.model_type] if self.timeout==None else timeout

        self.eval_layer = self.eval_layer_map[self.region]
        self.auth_layer = self.auth_layer_map[self.region]

        self.temp_dir = tempfile.gettempdir()
        self.file_objects_folder_path = os.path.join(self.temp_dir, 'file_objects')

        self.memory_lambda = {
            "main": 1024,
            "eval": 2048,
            "auth": 512
        }

        if self.categorical == True:
            self.task_type="classification"
        elif self.categorical == False:
            self.task_type="regression"
        else:
            self.task_type="custom"
        
        if(self.model_type=="custom"):
            self.model_class="custom"
        elif(self.model_type=="neural style transfer"):
            self.model_class="generative"
        elif(self.categorical==True):
            self.model_class="classification"
        elif(self.categorical==False):
            self.model_class="regression"

        self.model_template_mapping = {
            "classification": {
                "text": "1.txt",
                "image": "2.txt",
                "tabular": "4.txt",
                "audio": "7.txt",
                "video": "8.txt"
            },
            "regression": {
                "text": "1B.txt",
                "image": "3.txt",
                "tabular": "5.txt",
                "timeseries": "6.txt"
            },
            "generative": {
                "neural style transfer": "nst.txt"
            }
        }

    def create_prediction_api(self):
        global main
        if self.pyspark_support:
            try:
                from . import pyspark as main
            except:
                raise("Error: Please install pyspark to enable pyspark features")
        
        delete_files_from_temp_dir(self.temp_dir_file_deletion_list)

        if self.model_type != "custom":
            delete_folder(self.file_objects_folder_path)
            make_folder(self.file_objects_folder_path)

        if self.model_type == 'custom':
            with open("custom_lambda.py", 'r') as in_file:     
                newdata = in_file.read()
        else:
            data = pkg_resources.read_text(main, self.model_template_mapping[self.model_class][self.model_type])
            t = Template(data)
            if(self.model_class=="classification"):
                newdata = t.substitute(
                    bucket_name=os.environ.get("BUCKET_NAME"),
                    unique_model_id=self.unique_model_id,
                    labels=self.labels
                )
            else:
                newdata = t.substitute(
                    bucket_name=os.environ.get("BUCKET_NAME"),
                    unique_model_id=self.unique_model_id
                )

        with open(os.path.join(self.file_objects_folder_path, 'model.py'), 'w') as file:
            file.write(newdata)

        if(self.model_type == 'custom'):
            from . import custom_approach
            data = pkg_resources.read_text(custom_approach, 'lambda_function.py')
        else:
            data = pkg_resources.read_text(main, 'lambda_function.txt')

        with open(os.path.join(self.file_objects_folder_path, 'lambda_function.py'), 'w') as file:
            file.write(data)
        
        ###
        t = Template(pkg_resources.read_text(main, 'eval_lambda.txt'))
        data = t.substitute(bucket_name = self.bucket_name, unique_model_id = self.unique_model_id, task_type = self.task_type)
        with open(os.path.join(self.temp_dir, 'main.py'), 'w') as file:
            file.write(data)
        with ZipFile(os.path.join(self.temp_dir, 'archive2.zip'), 'a') as z:
            z.write(os.path.join(self.temp_dir, 'main.py'), 'main.py')
        self.aws_client.upload_file_to_s3(os.path.join(self.temp_dir, 'archive2.zip'), os.environ.get("BUCKET_NAME"), self.unique_model_id+"/"+'archiveeval.zip')

        data2 = pkg_resources.read_text(main, 'authorization.txt')
        with open(os.path.join(self.temp_dir, 'main.py'), 'w') as file:
            file.write(data2)
        with ZipFile(os.path.join(self.temp_dir, 'archive3.zip'), 'a') as z:
            z.write(os.path.join(self.temp_dir, 'main.py'), 'main.py')
        self.aws_client.upload_file_to_s3(os.path.join(self.temp_dir, 'archive3.zip'), os.environ.get("BUCKET_NAME"), self.unique_model_id+"/"+'archiveauth.zip')
        ###

        if self.model_type.lower() == 'custom':
            self.aws_client.upload_file_to_s3(os.path.join(self.temp_dir, 'exampledata.json'), os.environ.get("BUCKET_NAME"), self.unique_model_id+"/"+"exampledata.json")
    
        delete_files_from_temp_dir(self.temp_dir_file_deletion_list)

        ####################

        short_uuid = str(shortuuid.uuid())

        lambdarolename = 'myService-dev-us-' + self.region + '-lambdaRole'+short_uuid
        lambdapolicyname = 'myService-dev-' + self.region + '-lambdaPolicy'+short_uuid
        lambdafxnname = 'modfunction'+short_uuid
        lambdaauthfxnname = 'redisAccess'+short_uuid
        lambdaevalfxnname = 'evalfunction'+short_uuid

        from . import json_templates

        lambdarole1 = json.loads(pkg_resources.read_text(json_templates, 'lambda_role_1.txt'))
        lambdapolicy1 = json.loads(pkg_resources.read_text(json_templates, 'lambda_policy_1.txt'))

        self.aws_client.delete_iam_role(lambdarolename) # delete role for CodeBuild if role with same name exists
        self.aws_client.create_iam_role(lambdarolename, lambdarole1) # creating role for CodeBuild
        self.aws_client.delete_iam_policy(lambdapolicyname) # delete policy for CodeBuild if policy with same name exists
        self.aws_client.create_iam_policy(lambdapolicyname, lambdapolicy1) # creating policy for CodeBuild
        self.aws_client.attach_policy_to_role(lambdarolename, lambdapolicyname)

        sys.stdout.write('\r')
        sys.stdout.write("[============                         ] Progress: 40% - Creating custom containers...                        ")
        sys.stdout.flush()

        #########

        if(self.custom_libraries == False):
            from aimodelshare.containerization import create_lambda_using_base_image
            response6 = create_lambda_using_base_image(self.user_session, os.getenv("BUCKET_NAME"), self.file_objects_folder_path, lambdafxnname, self.apiid, self.repo_name, self.image_tag, self.memory, self.timeout)
        elif(self.custom_libraries == True):
            requirements = self.requirements.split(",")
            for i in range(len(requirements)):
                requirements[i] = requirements[i].strip(" ")
            with open(os.path.join(self.file_objects_folder_path, 'requirements.txt'), 'a') as f:
                for lib in requirements:
                    f.write('%s\n' % lib)
            requirements_file_path = os.path.join(self.file_objects_folder_path, 'requirements.txt')
            from aimodelshare.containerisation import deploy_container
            response6 = deploy_container(self.account_id, os.environ.get("AWS_REGION"), self.user_session, lambdafxnname, self.file_objects_folder_path,requirements_file_path,self.apiid, pyspark_support=self.pyspark_support)

        ##########

        lambdaclient = self.user_session.client('lambda')
        role_arn = 'arn:aws:iam::' + self.account_id + ':role/' + lambdarolename
        handler = 'main.handler'

        def create_lambda_function(function_name, python_runtime, role_arn, handler, code_source, timeout, memory_size, layers):
            response = self.aws_client.lambda_client.create_function(
                FunctionName = function_name,
                Runtime = python_runtime,
                Role = role_arn,
                Handler = handler,
                Code = code_source,
                Timeout = timeout,
                MemorySize = memory_size,
                Layers = layers
            )

        eval_code_source = {'S3Bucket': self.bucket_name, 'S3Key':  self.unique_model_id + "/" + "archiveeval.zip"}
        eval_layers = [self.eval_layer, self.auth_layer]
        create_lambda_function(lambdaevalfxnname, self.python_runtime, role_arn, handler, eval_code_source, 90, 2048, eval_layers)

        auth_code_source = {'S3Bucket': self.bucket_name, 'S3Key':  self.unique_model_id + "/" + "archiveauth.zip"}
        auth_layers = [self.auth_layer]
        create_lambda_function(lambdaauthfxnname, self.python_runtime, role_arn, handler, auth_code_source, 90, 512, auth_layers)

        sys.stdout.write('\r')
        sys.stdout.write("[==========================           ] Progress: 75% - Deploying prediction API...                          ")
        sys.stdout.flush()
        # }}}

        ##############################

        api_id = self.apiid
        stmt_id = 'apigateway-prod-'+str(shortuuid.uuid())

        resourceidlist = self.aws_client.get_api_resources(api_id)
        api_path_id = {}
        for i in resourceidlist:
            api_path_id.update({i['path']: i['id']})

        resource_id_parent = api_path_id['/']
        resource_id_lambda = api_path_id['/m']
        resource_id_eval = api_path_id['/eval']

        fxn_list = self.aws_client.lambda_client.list_functions()

        arn_prefix = "arn:aws:execute-api:" + self.region + ":" + self.account_id + ":" + api_id
        self.aws_client.add_invoke_resource_policy_to_lambda(lambdaauthfxnname, stmt_id, arn_prefix + "/*/*")
        #self.aws_client.add_invoke_resource_policy_to_lambda(lambdafxnname, 'apigateway-prod-2', arn_prefix + "/*/POST/m")
        #self.aws_client.add_invoke_resource_policy_to_lambda(lambdafxnname, 'apigateway-test-2', arn_prefix + "/*/POST/m")
        self.aws_client.add_invoke_resource_policy_to_lambda(lambdaevalfxnname, 'apigateway-prod-3', arn_prefix + "/*/POST/eval")
        self.aws_client.add_invoke_resource_policy_to_lambda(lambdaevalfxnname, 'apigateway-test-3', arn_prefix + "/*/POST/eval")

        integration_response = json.loads(pkg_resources.read_text(json_templates, 'integration_response.txt'))

        lambdarole2 = json.loads(pkg_resources.read_text(json_templates, 'lambda_role_2.txt'))
        lambdarolename2 = 'lambda_invoke_function_assume_apigw_role_2'
        lambdapolicy2 = json.loads(pkg_resources.read_text(json_templates, 'lambda_policy_2.txt'))
        lambdapolicyname2 = 'invokelambda'

        self.aws_client.delete_iam_role(lambdarolename2) # delete role for CodeBuild if role with same name exists
        self.aws_client.create_iam_role(lambdarolename2, lambdarole2) # creating role for CodeBuild
        self.aws_client.delete_iam_policy(lambdapolicyname2) # delete policy for CodeBuild if policy with same name exists
        self.aws_client.create_iam_policy(lambdapolicyname2, lambdapolicy2) # creating policy for CodeBuild
        self.aws_client.attach_policy_to_role(lambdarolename2, lambdapolicyname2)

        uri_str = "arn:aws:apigateway:" + self.region + ":lambda:path/2015-03-31/functions/arn:aws:lambda:" + self.region + ":" + self.account_id + ':function:' + lambdafxnname + '/invocations'
        credentials = 'arn:aws:iam::'+self.account_id+':role/' + lambdarolename2
        self.aws_client.integration_setup(api_id, resource_id_lambda, uri_str, credentials, integration_response)

        uri_str_2 = "arn:aws:apigateway:" + self.region + ":lambda:path/2015-03-31/functions/arn:aws:lambda:" + self.region + ":" + self.account_id + ':function:' + lambdaevalfxnname + '/invocations'
        credentials_2 = 'arn:aws:iam::'+self.account_id+':role/' + lambdarolename2
        self.aws_client.integration_setup(api_id, resource_id_eval, uri_str_2, credentials_2, integration_response)

        response = self.aws_client.apigateway_client.update_rest_api(
            restApiId=api_id,
            patchOperations=[
                {
                    "op": "replace",
                    "path": "/policy",
                    "value": '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": "*","Action": "execute-api:Invoke","Resource": "arn:aws:execute-api:'+os.environ.get("AWS_REGION")+':'+self.account_id+':'+api_id+'/prod/OPTIONS/*"}]}'
                }
            ]
        )

        auth_uri_str = "arn:aws:apigateway:"+ os.environ.get("AWS_REGION") +":lambda:path/2015-03-31/functions/arn:aws:lambda:"+os.environ.get("AWS_REGION")+":"+self.account_id+":function:"+lambdaauthfxnname+"/invocations"

        responseauthfxnapigateway = self.aws_client.apigateway_client.create_authorizer(
            restApiId=api_id,
            name='aimscustomauthfxn',
            type='TOKEN',
            authorizerUri=auth_uri_str,
            identitySource="method.request.header.authorizationToken",
            authorizerResultTtlInSeconds=0
        )

        responseauthfxnapigateway = self.aws_client.apigateway_client.get_authorizers(
            restApiId=api_id
        )

        authorizerid=responseauthfxnapigateway['items'][0]['id']

        stmt_idauth = 'apigateway-prod-'+str(shortuuid.uuid())
        response70 = self.user_session.client('lambda').add_permission(
            FunctionName=lambdaauthfxnname,
            StatementId=stmt_idauth,
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn='arn:aws:execute-api:'+self.region+':' + self.account_id+':'+api_id + '/authorizers/' + authorizerid
        )
        
        response_modmthd_addauth = self.user_session.client('apigateway').update_method(
            restApiId=api_id,
            resourceId=resource_id_lambda,
            httpMethod='POST',
            patchOperations=[
                {
                    'op': 'replace',
                    'path': '/authorizationType',
                    'value': 'CUSTOM',
                    'from': 'NONE'
                },
                {
                    'op': 'replace',
                    'path': '/authorizerId',
                    'value': authorizerid
                }
            ]
        )

        response_evalmthd_addauth = self.user_session.client('apigateway').update_method(
            restApiId=api_id,
            resourceId=resource_id_eval,
            httpMethod='POST',
            patchOperations=[
                {
                    'op': 'replace',
                    'path': '/authorizationType',
                    'value': 'CUSTOM',
                    'from': 'NONE'
                },
                {
                    'op': 'replace',
                    'path': '/authorizerId',
                    'value': authorizerid
                }])
        response12 = self.user_session.client('apigateway').create_deployment(
            restApiId=api_id,
            stageName='prod'
        )

        result = 'https://'+ api_id + '.execute-api.' + os.environ.get("AWS_REGION") + '.amazonaws.com/prod/m'

        if self.model_type=='custom':
            ### Progress Update #6/6 {{{
            sys.stdout.write('\r')
            sys.stdout.write("[=====================================] Progress: 100% - API deployment completed!                          ")
            sys.stdout.flush()
            # }}}

        return {"statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps(result)}



def create_prediction_api(model_filepath, unique_model_id, model_type, 
                          categorical, labels, apiid, custom_libraries, 
                          requirements, repo_name="", image_tag="", 
                          memory=None, timeout=None, pyspark_support=False):
    api_class = create_prediction_api_class(model_filepath, unique_model_id, 
                                           model_type, categorical, labels, apiid, 
                                           custom_libraries, requirements, repo_name, 
                                           image_tag, memory, timeout, pyspark_support)
    return api_class.create_prediction_api()

def get_api_json():
    region = "us-east-2"
    apijson = '''
    {
      "openapi": "3.0.1",
      "info": {
        "title": "modapi36146",
        "description": "This is a copy of my first API okay",
        "version": "2020-05-12T21:25:38Z"
      },
      "servers": [
        {
          "url": "https://8nee9nskdb.execute-api.%s.amazonaws.com/{basePath}",
          "variables": {
            "basePath": {
              "default": "/prod"
            }
          }
        }
      ],
      "paths": {
        "/eval": {
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
            }
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
            }
          }
        },
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
            }
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
            }
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
        }
      },
      "x-amazon-apigateway-policy": {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "execute-api:Invoke",
            "Resource": "arn:aws:execute-api:%s:517169013426:8nee9nskdb/prod/OPTIONS/*"
          }
        ]
      }
    }
            ''' % (region, region)
    return apijson

def delete_deployment(apiurl):
    """
    apiurl: string of API URL the user wishes to delete
    WARNING: User must supply high-level credentials in order to delete an API. 
    """
    from aimodelshare.aws import run_function_on_lambda

    # Provide Warning & Have user confirm deletion 
    print("Running this function will permanently delete all resources tied to this deployment, \n including the eval lambda and all models submitted to the model competition.\n")
    confirmation = input(prompt="To confirm, type 'permanently delete':")
    if confirmation.lower() == "permanently delete" or confirmation.lower() == "'permanently delete'":
        pass
    else:
        return print("'Delete Deployment' unsuccessful: operation cancelled by user.")

    # Confirm that creds are loaded, print warning if not
    if all(["AWS_ACCESS_KEY_ID" in os.environ, 
            "AWS_SECRET_ACCESS_KEY" in os.environ,
            "AWS_REGION" in os.environ,
           "username" in os.environ, 
           "password" in os.environ]):
        pass
    else:
        return print("'Delete Deployment' unsuccessful. Please provide credentials with set_credentials().")
    
    # get api_id from apiurl
    api_url_trim = apiurl.split('https://')[1]
    api_id = api_url_trim.split(".")[0]

    # Create User Session
    user_sess = boto3.session.Session(aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                                      aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), 
                                      region_name=os.environ.get('AWS_REGION'))
    
    s3 = user_sess.resource('s3')

    # Get bucket and model_id subfolder for user based on apiurl {{{
    response, error = run_function_on_lambda(
        apiurl, **{"delete": "FALSE", "versionupdateget": "TRUE"}
    )
    if error is not None:
        raise error

    _, api_bucket, model_id = json.loads(response.content.decode("utf-8"))
    # }}} 

    #Confirm username in bucket name
    ## TODO: Update this check to more secure process
    if os.environ.get("username").lower() in api_bucket:
        pass
    else:
        print("Permission denied. Please provide credentials that allow administrator access to this api.")
        return


    # get api resources
    api = user_sess.client('apigateway')
    resources = api.get_resources(
        restApiId=api_id
        )

    #get lambda arns 
    lambda_arns = list()
    for i in range(len(resources['items'])):
        if len(resources['items'][i]) > 2: 
            resource_id = resources['items'][i]['id']
            integration = api.get_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST'
            )
            uri=integration['uri']
            ans1=uri.split('functions/')
            lambda_arn = ans1[1].split('/invocations')[0]
            lambda_arns.append(lambda_arn)
    else: 
        pass 

    # get authorizer arn 
    authorizers = api.get_authorizers(
                  restApiId=api_id
                  )

    authorizer_full_arn=authorizers['items'][0]['authorizerUri']
    ans1=authorizer_full_arn.split('functions/')
    auth_arn = ans1[1].split('/invocations')[0]
    lambda_arns.append(auth_arn)

    # delete lambdas & authorizer
    client = boto3.client('lambda', region_name=os.environ.get("AWS_REGION"))
    for arn in lambda_arns:
        lambda_response = client.delete_function(
            FunctionName = arn
        )

    # delete api
    client = boto3.client('apigateway', region_name=os.environ.get("AWS_REGION"))
    api_response = client.delete_rest_api(
        restApiId=api_id
    )

    # delete api page on front end
    bodydata = {'apiurl': apiurl,
                'delete': "TRUE"
                }
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}

    requests.post("https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)
    
    # delete competition posting
    bodydata = {"apiurl": apiurl,
                "delete":"TRUE",
                "experiment":"FALSE"
                                }

    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("AWS_TOKEN"), 'Access-Control-Allow-Headers':
                                    'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # competitiondata lambda function invoked through below url to update model submissions and contributors
    requests.post("https://o35jwfakca.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)

    # delete experiment posting
    bodydata = {"apiurl": apiurl,
                "delete":"TRUE",
                "experiment":"TRUE"
                                }

    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("AWS_TOKEN"), 'Access-Control-Allow-Headers':
                                    'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # competitiondata lambda function invoked through below url to update model submissions and contributors
    requests.post("https://o35jwfakca.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)    
    
    # Delete competition data container image
    try:
            content_object = s3.Object(bucket_name=api_bucket, key=model_id + "/competitionuserdata.json")
            file_content = content_object.get()['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)
            ecr_uri=json_content['datauri']      

            ecr_client = user_sess.client('ecr-public')

            repository_image = ecr_uri.split('/')[2]

            repository = repository_image.split(':')[0]
            image = repository_image.split(':')[1]

            response = ecr_client.batch_delete_image(
                repositoryName=repository,
                imageIds=[
                    {
                        'imageTag': image
                    }
                ]
            )

            image_details = ecr_client.describe_images(
                repositoryName=repository
            )

            if len(image_details['imageDetails'])==0:
                response = ecr_client.delete_repository(
                    repositoryName=repository
                )
    except:
        pass
    # delete s3 folder
    bucket = s3.Bucket(api_bucket)
    bucket.objects.filter(Prefix= model_id+'/').delete() 

    return "Deployment deleted successfully."

__all__ = [
    get_api_json,
    create_prediction_api,
    delete_deployment
]
