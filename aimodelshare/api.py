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
import requests
import sys
from zipfile import ZipFile, ZIP_STORED, ZipInfo
import shutil
import shortuuid
from aimodelshare.containerization import create_lambda_using_base_image
from aimodelshare.containerisation import deploy_container

def create_prediction_api(model_filepath, unique_model_id, model_type, categorical, labels, apiid, custom_libraries, requirements, repo_name="", image_tag=""):

    from zipfile import ZipFile
    import zipfile
    import tempfile
    # create temporary folder
    temp_dir = tempfile.gettempdir()
    import os
    if os.path.exists(os.path.join(temp_dir, 'archive.zip')):
      os.remove(os.path.join(temp_dir, 'archive.zip'))
    else:
      pass
    if os.path.exists(os.path.join(temp_dir, 'archivetest.zip')):
      os.remove(os.path.join(temp_dir, 'archivetest.zip'))
    else:
      pass 	

    if os.path.exists(os.path.join(temp_dir, 'archive2.zip')):
      os.remove(os.path.join(temp_dir, 'archive2.zip'))
    else:
      pass         

    if os.path.exists(os.path.join(temp_dir, 'archive3.zip')):
      os.remove(os.path.join(temp_dir, 'archive3.zip'))
    else:
      pass 
    if os.path.exists(os.path.join(temp_dir,'main.py')):
      os.remove(os.path.join(temp_dir,'main.py'))
    else:
      pass 
    if os.path.exists(os.path.join(temp_dir,'ytest.pkl')):
      os.remove(os.path.join(temp_dir,'ytest.pkl'))
    else:
      pass 
    model_type = model_type.lower()
    categorical = categorical.upper()
    # Wait for 5 seconds to ensure aws iam user on user account has time to load into aws's system
    #time.sleep(5)

    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                          aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                          region_name=os.environ.get("AWS_REGION"))
    if(model_type=="neural style transfer"):
            model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:keras_image:1"
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    elif model_type=='image' :
            model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:keras_image:1"
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    elif model_type=='text':
            model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_layer:2"
            keras_layer ='arn:aws:lambda:us-east-1:517169013426:layer:keras_preprocesor:1'
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    elif model_type == 'tabular' or model_type =='timeseries':
            model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    elif model_type.lower() == 'audio':
            model_layer = "arn:aws:lambda:us-east-1:517169013426:layer:librosa_nosklearn:9"
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    elif model_type.lower() == 'video':
            model_layer = "arn:aws:lambda:us-east-1:517169013426:layer:videolayer:3"
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    elif model_type.lower() == 'custom':
            model_layer = "arn:aws:lambda:us-east-1:517169013426:layer:videolayer:3"
            eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:eval_layer_test:6"
            auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:aimsauth_layer:2"
    else :
        print("no matching model data type to load correct python package zip file (lambda layer)")

    #cloud_layer = "arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
    # dill_layer ="arn:aws:lambda:us-east-1:517169013426:layer:dill:3"

    # Update note:  dyndb data to add.  apiname. (include username too)


    account_number = user_session.client(
        'sts').get_caller_identity().get('Account')

    import tempfile
    from zipfile import ZipFile
    import zipfile
    import os

    # create temporary folder
    temp_dir = tempfile.gettempdir()

    try:
      import importlib.resources as pkg_resources

    except ImportError:
      # Try backported to PY<37 `importlib_resources`.
      import importlib_resources as pkg_resources

    from . import main  # relative-import the *package* containing the templates

    from . import custom_approach

    file_objects_folder_path = os.path.join(temp_dir, 'file_objects')

    if model_type.lower() != "custom":  # file_objects already initialized if custom
        if os.path.exists(file_objects_folder_path):
            shutil.rmtree(file_objects_folder_path)
        os.mkdir(file_objects_folder_path)


    # write main handlers
    if(model_type == "neural style transfer"):
            data = pkg_resources.read_text(main, 'nst.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id)
    elif model_type == 'text' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '1.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id, labels=labels)
    elif model_type == 'text' and categorical == 'FALSE':
            data = pkg_resources.read_text(main, '1B.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id)
    elif model_type == 'image' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '2.txt')
            from string import Template
            t = Template(data)            
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id, labels=labels)
    elif model_type == 'image' and categorical == 'FALSE':
            data = pkg_resources.read_text(main, '3.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id)
    elif all([model_type == 'tabular', categorical == 'TRUE']):
            data = pkg_resources.read_text(main, '4.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id, labels=labels)
    elif all([model_type == 'tabular', categorical == 'FALSE']):
            data = pkg_resources.read_text(main, '5.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id)
    elif model_type.lower() == 'timeseries' and categorical == 'FALSE':
            data = pkg_resources.read_text(main, '6.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id)
    elif model_type.lower() == 'audio' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '7.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id, labels=labels)
    elif model_type.lower() == 'video' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '8.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id, labels=labels)
    elif model_type.lower() == 'custom':
         with open("custom_lambda.py", 'r') as in_file:     
             newdata = in_file.read()

    with open(os.path.join(file_objects_folder_path, 'model.py'), 'w') as file:
        file.write(newdata)

    if(model_type.lower() == 'custom'):
         data = pkg_resources.read_text(custom_approach, 'lambda_function.py')
         with open(os.path.join(file_objects_folder_path, 'lambda_function.py'), 'w') as file:
             file.write(data)
    else:
        data = pkg_resources.read_text(main, 'lambda_function.txt')
        with open(os.path.join(file_objects_folder_path, 'lambda_function.py'), 'w') as file:
            file.write(data)
        
    #with zipfile.ZipFile(os.path.join(temp_dir, 'archive.zip'), 'a') as z:
    #    z.write(os.path.join(temp_dir, 'main.py'), 'main.py')

    # preprocessor upload

    # Upload lambda function zipfile to user's model file folder on s3
    # try:
    #     # This should go to developer's account from my account
    #     s3_client = user_session.client('s3')
    #     s3_client.upload_file(os.path.join(
    #          temp_dir, 'archive.zip'), os.environ.get("BUCKET_NAME"),  unique_model_id+"/"+'archivetest.zip')

    # except Exception as e:
    #     print(e)

    if os.path.exists(os.path.join(temp_dir,'main.py')):
      os.remove(os.path.join(temp_dir,'main.py'))
    else:
      pass  

    if os.path.exists(os.path.join(temp_dir,'archive.zip')):
      os.remove(os.path.join(temp_dir,'archive.zip'))
    else:
      pass   
    
    if any([categorical=="TRUE",categorical=="True",categorical=="true"]):
         task_type="classification"
    elif any([categorical=="FALSE", categorical=="False", categorical=="false"]):
         task_type="regression"
    else:
         task_type="custom"
    # Upload model eval lambda function zipfile to user's model file folder on s3
    # use task_type
    data = pkg_resources.read_text(main, 'eval_lambda.txt')
    from string import Template
    t = Template(data)
    newdata = t.substitute(
        bucket_name=os.environ.get("BUCKET_NAME"), unique_model_id=unique_model_id, task_type=task_type)
    with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
        file.write(newdata)

    with zipfile.ZipFile(os.path.join(temp_dir, 'archive2.zip'), 'a') as z:
        z.write(os.path.join(temp_dir, 'main.py'), 'main.py')

    # preprocessor upload

    # Upload lambda function zipfile to user's model file folder on s3
    try:
        # This should go to developer's account from my account
        s3_client = user_session.client('s3')
        s3_client.upload_file(os.path.join(
            temp_dir, 'archive2.zip'), os.environ.get("BUCKET_NAME"),  unique_model_id+"/"+'archiveeval.zip')

    except Exception as e:
        print(e)

    import os

    if os.path.exists(os.path.join(temp_dir,'main.py')):
      os.remove(os.path.join(temp_dir,'main.py'))
    else:
      pass         

    if os.path.exists(os.path.join(temp_dir,'archive2.zip')):
      os.remove(os.path.join(temp_dir,'archive2.zip'))
    else:
      pass  

    # Upload model eval lambda function zipfile to user's model file folder on s3
    if categorical == 'TRUE':
            newdata = pkg_resources.read_text(main, 'authorization.txt')
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif categorical == 'FALSE':
            newdata = pkg_resources.read_text(main, 'authorization.txt')
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    with zipfile.ZipFile(os.path.join(temp_dir, 'archive3.zip'), 'a') as z:
        z.write(os.path.join(temp_dir, 'main.py'), 'main.py')

    # preprocessor upload

    # Upload lambda function zipfile to user's model file folder on s3
    try:
        # This should go to developer's account from my account
        s3_client = user_session.client('s3')
        s3_client.upload_file(os.path.join(
            temp_dir, 'archive3.zip'), os.environ.get("BUCKET_NAME"),  unique_model_id+"/"+'archiveauth.zip')

    except Exception as e:
        print(e)

    if model_type.lower() == 'custom':
        s3_client.upload_file(os.path.join(
            temp_dir, 'exampledata.json'), os.environ.get("BUCKET_NAME"),  unique_model_id+"/"+"exampledata.json")
 
    import os
    if os.path.exists(os.path.join(temp_dir, 'archive.zip')):
      os.remove(os.path.join(temp_dir, 'archive.zip'))
    else:
      pass
    if os.path.exists(os.path.join(temp_dir, 'archivetest.zip')):
      os.remove(os.path.join(temp_dir, 'archivetest.zip'))
    else:
      pass 	

    if os.path.exists(os.path.join(temp_dir, 'archive2.zip')):
      os.remove(os.path.join(temp_dir, 'archive2.zip'))
    else:
      pass         

    if os.path.exists(os.path.join(temp_dir, 'archive3.zip')):
      os.remove(os.path.join(temp_dir, 'archive3.zip'))
    else:
      pass 
    if os.path.exists(os.path.join(temp_dir,'main.py')):
      os.remove(os.path.join(temp_dir,'main.py'))
    else:
      pass 


    # Create and/or update roles for lambda function you will create below
    lambdarole1 = {u'Version': u'2012-10-17', u'Statement': [
        {u'Action': u'sts:AssumeRole', u'Effect': u'Allow', u'Principal': {u'Service': u'lambda.amazonaws.com'}}]}

    roles = user_session.client('iam').list_roles()
    
    lambdarolename = 'myService-dev-us-east-1-lambdaRole'+str(shortuuid.uuid())
    lambdafxnname = 'modfunction'+str(shortuuid.uuid())
    lambdaauthfxnname = 'redisAccess'+str(shortuuid.uuid())
    lambdaevalfxnname = 'evalfunction'+str(shortuuid.uuid())

    response6 = user_session.resource('iam').create_role(
            AssumeRolePolicyDocument=json.dumps(lambdarole1),
            Path='/',
            RoleName=lambdarolename,
        )
    response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogGroup"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':*"],"Effect": "Allow"},{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdafxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            os.environ.get("BUCKET_NAME")+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy'+str(shortuuid.uuid()),
            RoleName=lambdarolename,
        )

    response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdaauthfxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            os.environ.get("BUCKET_NAME")+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy'+str(shortuuid.uuid()),
            RoleName=lambdarolename,
        )

    response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaevalfxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdaevalfxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:ListBucket"],"Resource": ["arn:aws:s3:::' +
            os.environ.get("BUCKET_NAME")+'"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            os.environ.get("BUCKET_NAME")+'/*"],"Effect": "Allow"},{"Action": ["s3:PutObject"],"Resource": ["arn:aws:s3:::' +
            os.environ.get("BUCKET_NAME")+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy'+str(shortuuid.uuid()),
            RoleName=lambdarolename,
        )

    lambdaclient = user_session.client('lambda')

    ##!!!  this is f'd.  looks like aishwarya hasn't fixed the reference to other layers!  It's a good start, but we should add a subfunction to return
    ## the correct layer from an externally stored list (save arns to a github repo and allow them to be imported here at some point.)
    layers = []
    layers.append(model_layer)
    if model_type =='text':
        layers.append(keras_layer)

    # if model_type=='sklearn_text' or  model_type=='keras_text' or model_type=='flubber_text' or model_type =='text':
    # layers.append(keras_layer)

    #response6 = lambdaclient.create_function(FunctionName=lambdafxnname, Runtime='python3.6', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
    #                                          Code={
    #                                              'S3Bucket': os.environ.get("BUCKET_NAME"),
    #                                              'S3Key':  unique_model_id+"/"+'archivetest.zip'
    #                                          }, Timeout=10, MemorySize=512, Layers=layers)  # ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE


    ### Progress Update #3/6 {{{
    sys.stdout.write('\r')
    sys.stdout.write("[============                         ] Progress: 40% - Creating custom containers...                        ")
    sys.stdout.flush()
    # }}}
    
    if(any([custom_libraries=='FALSE',custom_libraries=='false'])):
        response6 = create_lambda_using_base_image(user_session, os.getenv("BUCKET_NAME"), file_objects_folder_path, lambdafxnname, apiid, repo_name, image_tag, 3072, 90)
    elif(any([custom_libraries=='TRUE',custom_libraries=='true'])):

        requirements = requirements.split(",")
        for i in range(len(requirements)):
            requirements[i] = requirements[i].strip(" ")

        with open(os.path.join(file_objects_folder_path, 'requirements.txt'), 'a') as f:
            for lib in requirements:
                f.write('%s\n' % lib)
        
        requirements_file_path = os.path.join(file_objects_folder_path, 'requirements.txt')


        #response6 = deploy_lambda_using_sam(user_session, os.getenv("BUCKET_NAME"), requirements, file_objects_folder_path, lambdafxnname, apiid, 1024, 90, "3.7")
        response6 = deploy_container(account_number, os.environ.get("AWS_REGION"), user_session, lambdafxnname, file_objects_folder_path,requirements_file_path,apiid)

    response6evalfxn = lambdaclient.create_function(FunctionName=lambdaevalfxnname, Runtime='python3.7', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
                                          Code={
                                              'S3Bucket': os.environ.get("BUCKET_NAME"),
                                              'S3Key':  unique_model_id+"/"+'archiveeval.zip'
                                          }, Timeout=90, MemorySize=2048, Layers=[eval_layer,auth_layer])  # ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE

    response6authfxn = lambdaclient.create_function(FunctionName=lambdaauthfxnname, Runtime='python3.7', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
                                          Code={
                                              'S3Bucket': os.environ.get("BUCKET_NAME"),
                                              'S3Key':  unique_model_id+"/"+'archiveauth.zip'
                                          }, Timeout=10, MemorySize=512, Layers=[auth_layer])  # ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE

    ### Progress Update #4/6 {{{
    sys.stdout.write('\r')
    sys.stdout.write("[==========================           ] Progress: 75% - Deploying prediction API...                          ")
    sys.stdout.flush()
    # }}}


    #add create api about here
    #TODO: 
    api_name = 'modapi'+str(shortuuid.uuid())	

    # Update note:  change apiname in apijson from modapi890799 to randomly generated apiname?  or aimodelshare generic name?

    user_client = boto3.client('apigateway', aws_access_key_id=str(
        os.environ.get("AWS_ACCESS_KEY_ID")), aws_secret_access_key=str(os.environ.get("AWS_SECRET_ACCESS_KEY")), region_name=str(os.environ.get("AWS_REGION")))

    api_id = apiid

    # Update note:  dyndb data to add.  api_id and resourceid "Resource": "arn:aws:execute-api:us-east-1:517169013426:iu3q9io652/prod/OPTIONS/m"

    # Update note:  dyndb data to add.  api_id and resourceid "Resource": "arn:aws:execute-api:us-east-1:517169013426:iu3q9io652/prod/OPTIONS/m"


    response3 = user_client.get_resources(restApiId=api_id)


    resourceidlist=response3['items']

    # Python3 code to iterate over a list 
    api_id_data = {}   
    # Using for loop 

    for i in resourceidlist: 
        api_id_data.update({i['path']: i['id']})

    resource_id_parent=api_id_data['/']

    resource_id=api_id_data['/m']

    resource_id_eval=api_id_data['/eval']


    # NEXT: update permissions

    fxn_list = lambdaclient.list_functions()

    stmt_id = 'apigateway-prod-'+str(shortuuid.uuid())
    # upload authfxn code first

    response7_1 = lambdaclient.add_permission(
            FunctionName=lambdaauthfxnname,
            StatementId=stmt_id,
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn='arn:aws:execute-api:us-east-1:'+account_number+":"+api_id+'/*/*',
    )
    # Update note:  dyndb data to add.  lambdafxnname

    # change api name below?

    time.sleep(15)

    response7 = lambdaclient.add_permission(
        FunctionName=lambdafxnname,
        StatementId='apigateway-prod-2',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:' +
        account_number+":"+api_id+'/*/POST/m',
    )

    response8 = lambdaclient.add_permission(
        FunctionName=lambdafxnname,
        StatementId='apigateway-test-2',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:' +
        account_number+":"+api_id+'/*/POST/m',
    )

    response7 = lambdaclient.add_permission(
        FunctionName=lambdaevalfxnname,
        StatementId='apigateway-prod-3',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:' +
        account_number+":"+api_id+'/*/POST/eval',
    )

    response8 = lambdaclient.add_permission(
        FunctionName=lambdaevalfxnname,
        StatementId='apigateway-test-3',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:us-east-1:' +
        account_number+":"+api_id+'/*/POST/eval',
    )

    # Create and or update lambda and apigateway gateway roles


    lambdarole2 = {u'Version': u'2012-10-17', u'Statement': [{u'Action': u'sts:AssumeRole', u'Principal': {
        u'Service': [u'lambda.amazonaws.com', u'apigateway.amazonaws.com']}, u'Effect': u'Allow', u'Sid': u''}]}

    if str(roles['Roles']).find("lambda_invoke_function_assume_apigw_role") > 0:
        None
    else:
        response10 = user_session.resource('iam').create_role(
            AssumeRolePolicyDocument=json.dumps(lambdarole2),
            Path='/',
            RoleName='lambda_invoke_function_assume_apigw_role',
        )
        time.sleep(10)

    response11 = user_session.client('apigateway').put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:' +
        account_number+':function:'+lambdafxnname+'/invocations',
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
        uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:' +
        account_number+':function:'+lambdafxnname+'/invocations',
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
    response11_2 = user_session.client('iam').put_role_policy(
        PolicyDocument='{"Version":"2012-10-17","Statement":{"Effect":"Allow","Action":"lambda:InvokeFunction","Resource":"*"}}',
        PolicyName='invokelambda',
        RoleName='lambda_invoke_function_assume_apigw_role',
    )
    response12_1 = user_client.update_rest_api(
        restApiId=api_id,
        patchOperations=[{
            "op": "replace",
            "path": "/policy",
            "value": '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": "*","Action": "execute-api:Invoke","Resource": "arn:aws:execute-api:'+os.environ.get("AWS_REGION")+':'+account_number+':'+api_id+'/prod/OPTIONS/*"}]}'
        }, ]
    )

    # start here to update eval fxn integration with api resource_id_eval, lambdaevalfxnname
    response11 = user_session.client('apigateway').put_integration(
        restApiId=api_id,
        resourceId=resource_id_eval,
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:' +
        account_number+':function:'+lambdaevalfxnname+'/invocations',
        credentials='arn:aws:iam::'+account_number+':role/lambda_invoke_function_assume_apigw_role')
    response11_1 = user_session.client('apigateway').put_integration(
        restApiId=api_id,
        resourceId=resource_id_eval,
        httpMethod='OPTIONS',
        type='MOCK',
        requestTemplates={
            'application/json': '{"statusCode": 200}'
        },
        integrationHttpMethod='OPTIONS',
        uri='arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:' +
        account_number+':function:'+lambdaevalfxnname+'/invocations',
        credentials='arn:aws:iam::'+account_number+':role/lambda_invoke_function_assume_apigw_role')
    response11_1B = user_session.client('apigateway').put_integration(
        restApiId=api_id,
        resourceId=resource_id_eval,
        httpMethod='OPTIONS',
        type='MOCK',
        requestTemplates={
            'application/json': '{"statusCode": 200}'
        }
    )
    response11_1C = user_session.client('apigateway').put_integration_response(
        restApiId=api_id,
        resourceId=resource_id_eval,
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
    response11_2 = user_session.client('iam').put_role_policy(
        PolicyDocument='{"Version":"2012-10-17","Statement":{"Effect":"Allow","Action":"lambda:InvokeFunction","Resource":"*"}}',
        PolicyName='invokelambda',
        RoleName='lambda_invoke_function_assume_apigw_role',
    )
        
    response12_1 = user_client.update_rest_api(
        restApiId=api_id,
        patchOperations=[{
            "op": "replace",
            "path": "/policy",
            "value": '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": "*","Action": "execute-api:Invoke","Resource": "arn:aws:execute-api:'+ os.environ.get("AWS_REGION") +':'+account_number+':'+api_id+'/prod/OPTIONS/*"}]}'
        }, ]
    )

    responseauthfxnapigateway = user_session.client('apigateway').create_authorizer(
        restApiId=api_id,
        name='aimscustomauthfxn',
        type='TOKEN',
        authorizerUri="arn:aws:apigateway:"+ os.environ.get("AWS_REGION") +":lambda:path/2015-03-31/functions/arn:aws:lambda:"+os.environ.get("AWS_REGION")+":"+account_number+":function:"+lambdaauthfxnname+"/invocations",
        identitySource="method.request.header.authorizationToken",
        authorizerResultTtlInSeconds=0
    )

    responseauthfxnapigateway = user_session.client('apigateway').get_authorizers(
        restApiId=api_id
    )

    authorizerid=responseauthfxnapigateway['items'][0]['id']

    stmt_idauth = 'apigateway-prod-'+str(shortuuid.uuid())
    response70 = user_session.client('lambda').add_permission(
        FunctionName=lambdaauthfxnname,
        StatementId=stmt_idauth,
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn='arn:aws:execute-api:'+os.environ.get("AWS_REGION")+':' +
        account_number+':'+api_id+'/authorizers/'+authorizerid,
    )
    
    response_modmthd_addauth = user_session.client('apigateway').update_method(
        restApiId=api_id,
        resourceId=resource_id,
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

    response_evalmthd_addauth = user_session.client('apigateway').update_method(
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
    response12 = user_session.client('apigateway').create_deployment(
        restApiId=api_id,
        stageName='prod')


    result = 'https://'+api_id + '.execute-api.'+os.environ.get("AWS_REGION")+'.amazonaws.com/prod/m'

    return {"statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps(result)}



def get_api_json():
    apijson = '''
        {
          "openapi": "3.0.1",
          "info": {
            "title": "modapi36146",
            "description": "This is a copy of my first API",
            "version": "2020-05-12T21:25:38Z"
          },
          "servers": [
            {
              "url": "https://8nee9nskdb.execute-api.us-east-1.amazonaws.com/{basePath}",
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
                "Resource": "arn:aws:execute-api:us-east-1:517169013426:8nee9nskdb/prod/OPTIONS/*"
              }
            ]
          }
        }
                '''
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

    # delete s3 folder
    bucket = s3.Bucket(api_bucket)
    bucket.objects.filter(Prefix= model_id+'/').delete() 

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
                "delete":"TRUE"
                                }

    # Get the response
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("AWS_TOKEN"), 'Access-Control-Allow-Headers':
                                    'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # competitiondata lambda function invoked through below url to update model submissions and contributors
    requests.post("https://o35jwfakca.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)

    return "API deleted successfully."


__all__ = [
    get_api_json,
    create_prediction_api,
    delete_deployment
]
