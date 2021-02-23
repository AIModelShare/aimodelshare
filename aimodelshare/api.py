# code version #2: build on #1, without testing above yet.  create eval lambda, add policy updates
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


def create_prediction_api(my_credentials, model_filepath, unique_model_id, model_type,categorical, labels):
    from zipfile import ZipFile
    import zipfile

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
    AI_MODELSHARE_AccessKeyId = my_credentials["AI_MODELSHARE_AccessKeyId"]
    AI_MODELSHARE_SecretAccessKey = my_credentials["AI_MODELSHARE_SecretAccessKey"]
    region = my_credentials["region"]
    username = my_credentials["username"]
    bucket_name = my_credentials["bucket_name"]
    model_type = model_type.lower()
    categorical = categorical.upper()
  # Wait for 5 seconds to ensure aws iam user on user account has time to load into aws's system
    time.sleep(5)
    user_session = boto3.session.Session(aws_access_key_id=AI_MODELSHARE_AccessKeyId,
                                         aws_secret_access_key=AI_MODELSHARE_SecretAccessKey, region_name=region)
    if model_type=='image' :
           model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:keras_image:1"
           eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
           auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:redissearch:1"
    elif model_type=='text':
           model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_layer:2"
           keras_layer ='arn:aws:lambda:us-east-1:517169013426:layer:keras_preprocesor:1'
           eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
           auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:redissearch:1"
    elif model_type == 'tabular' or model_type =='timeseries':
           model_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
           eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
           auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:redissearch:1"
    elif model_type.lower() == 'audio':
           model_layer = "arn:aws:lambda:us-east-1:517169013426:layer:librosa_nosklearn:9"
           eval_layer ="arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
           auth_layer ="arn:aws:lambda:us-east-1:517169013426:layer:redissearch:1"
    else :
        print("no matching model data type to load correct python package zip file (lambda layer)")

    #cloud_layer = "arn:aws:lambda:us-east-1:517169013426:layer:tabular_cloudpicklelayer:1"
    # dill_layer ="arn:aws:lambda:us-east-1:517169013426:layer:dill:3"

  # Update note:  dyndb data to add.  apiname. (include username too)


    account_number = user_session.client(
        'sts').get_caller_identity().get('Account')

    try:
    	import importlib.resources as pkg_resources

    except ImportError:
    	# Try backported to PY<37 `importlib_resources`.
    	import importlib_resources as pkg_resources

    from . import main  # relative-import the *package* containing the templates


    # write main handlers
    if model_type == 'text' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '1.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id, labels=labels)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif model_type == 'text' and categorical == 'FALSE':
            data = pkg_resources.read_text(main, '1B.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif model_type == 'image' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '2.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id, labels=labels)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif model_type == 'image' and categorical == 'FALSE':
            data = pkg_resources.read_text(main, '3.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif all([model_type == 'tabular', categorical == 'TRUE']):
            data = pkg_resources.read_text(main, '4.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id, labels=labels)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif all([model_type == 'tabular', categorical == 'FALSE']):
            data = pkg_resources.read_text(main, '5.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif model_type.lower() == 'timeseries' and categorical == 'FALSE':
            data = pkg_resources.read_text(main, '6.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)

    elif model_type.lower() == 'audio' and categorical == 'TRUE':
            data = pkg_resources.read_text(main, '7.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id, labels=labels)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)

    with zipfile.ZipFile(os.path.join(temp_dir, 'archive.zip'), 'a') as z:
        z.write(os.path.join(temp_dir, 'main.py'), 'main.py')

	
    # preprocessor upload

# Upload lambda function zipfile to user's model file folder on s3
    try:
        # This should go to developer's account from my account
        s3_client = user_session.client('s3')
        s3_client.upload_file(os.path.join(
            temp_dir, 'archive.zip'), bucket_name,  unique_model_id+"/"+'archivetest.zip')

    except Exception as e:
        print(e)
    
    if os.path.exists(os.path.join(temp_dir,'main.py')):
      os.remove(os.path.join(temp_dir,'main.py'))
    else:
      pass  

    if os.path.exists(os.path.join(temp_dir,'archive.zip')):
      os.remove(os.path.join(temp_dir,'archive.zip'))
    else:
      pass 
# Upload model eval lambda function zipfile to user's model file folder on s3
    if categorical == 'TRUE':
            data = pkg_resources.read_text(main, 'eval_classification.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id)
            with open(os.path.join(temp_dir, 'main.py'), 'w') as file:
                file.write(newdata)
    elif categorical == 'FALSE':
            data = pkg_resources.read_text(main, 'eval_regression.txt')
            from string import Template
            t = Template(data)
            newdata = t.substitute(
                bucket_name=bucket_name, unique_model_id=unique_model_id)
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
            temp_dir, 'archive2.zip'), bucket_name,  unique_model_id+"/"+'archiveeval.zip')

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
            temp_dir, 'archive3.zip'), bucket_name,  unique_model_id+"/"+'archiveauth.zip')

    except Exception as e:
        print(e)
    
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

#!!! 2. update lambda creation code and iam policy attachements/apigateway integrations next.

    # Create and/or update roles for lambda function you will create below
    lambdarole1 = {u'Version': u'2012-10-17', u'Statement': [
        {u'Action': u'sts:AssumeRole', u'Effect': u'Allow', u'Principal': {u'Service': u'lambda.amazonaws.com'}}]}
    lambdarolename = 'myService-dev-us-east-1-lambdaRole'

    roles = user_session.client('iam').list_roles()

    lambdafxnname = 'modfunction'+str(random.randint(1, 1000000))
    lambdaauthfxnname = 'redisAccess'+str(random.randint(1, 1000000))
    lambdaevalfxnname = 'evalfunction'+str(random.randint(1, 1000000))

    if str(roles['Roles']).find("myService-dev-us-east-1-lambdaRole") > 0:
        response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdafxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
            RoleName=lambdarolename,
        )
    else:
        response6 = user_session.resource('iam').create_role(
            AssumeRolePolicyDocument=json.dumps(lambdarole1),
            Path='/',
            RoleName=lambdarolename,
        )
        response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdafxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdafxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
            RoleName=lambdarolename,
        )
    if str(roles['Roles']).find("myService-dev-us-east-1-lambdaRole") > 0:

        response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdaauthfxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
            RoleName=lambdarolename,
        )
    else:
        response6 = user_session.resource('iam').create_role(
            AssumeRolePolicyDocument=json.dumps(lambdarole1),
            Path='/',
            RoleName=lambdarolename,
        )
        response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaauthfxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdaauthfxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
            RoleName=lambdarolename,
        )
    if str(roles['Roles']).find("myService-dev-us-east-1-lambdaRole") > 0:

        response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaevalfxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdaevalfxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
            RoleName=lambdarolename,
        )
    else:
        response6 = user_session.resource('iam').create_role(
            AssumeRolePolicyDocument=json.dumps(lambdarole1),
            Path='/',
            RoleName=lambdarolename,
        )
        response6_2 = user_session.client('iam').put_role_policy(
            PolicyDocument='{"Version":"2012-10-17","Statement":[{"Action": ["logs:CreateLogStream"], "Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/'+lambdaevalfxnname +
            ':*"],"Effect": "Allow"},{"Action": ["logs:PutLogEvents"],"Resource": ["arn:aws:logs:us-east-1:'+account_number+':log-group:/aws/lambda/' +
            lambdaevalfxnname +
            ':*:*"],"Effect": "Allow"},{"Action": ["s3:GetObject"],"Resource": ["arn:aws:s3:::' +
            bucket_name+'/*"],"Effect": "Allow"}]}',
            PolicyName='S3AccessandcloudwatchlogPolicy',
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

    response6 = lambdaclient.create_function(FunctionName=lambdafxnname, Runtime='python3.6', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
                                             Code={
                                                 'S3Bucket': bucket_name,
                                                 'S3Key':  unique_model_id+"/"+'archivetest.zip'
                                             }, Timeout=10, MemorySize=512, Layers=layers)  # ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE

    response6evalfxn = lambdaclient.create_function(FunctionName=lambdaevalfxnname, Runtime='python3.6', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
                                          Code={
                                              'S3Bucket': bucket_name,
                                              'S3Key':  unique_model_id+"/"+'archiveeval.zip'
                                          }, Timeout=10, MemorySize=512, Layers=[eval_layer])  # ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE
    response6authfxn = lambdaclient.create_function(FunctionName=lambdaauthfxnname, Runtime='python3.6', Role='arn:aws:iam::'+account_number+':role/'+lambdarolename, Handler='main.handler',
                                          Code={
                                              'S3Bucket': bucket_name,
                                              'S3Key':  unique_model_id+"/"+'archiveauth.zip'
                                          }, Timeout=10, MemorySize=512, Layers=[auth_layer])  # ADD ANOTHER LAYER ARN .. THE ONE SPECIFIC TO MODEL TYPE


#add create api about here
#TODO: 
    api_name = 'modapi'+str(random.randint(1, 1000000))	
  # Update note:  change apiname in apijson from modapi890799 to randomly generated apiname?  or aimodelshare generic name?
    data = get_api_json()
    from string import Template
    t = Template(data)
    api_json = t.substitute(
                lambda_arn=response6authfxn['FunctionArn'])



    user_client = boto3.client('apigateway', aws_access_key_id=str(
        AI_MODELSHARE_AccessKeyId), aws_secret_access_key=str(AI_MODELSHARE_SecretAccessKey), region_name=str(region))

    response2 = user_client.import_rest_api(
        failOnWarnings=True,
        parameters={
            'endpointConfigurationTypes': 'REGIONAL'
        },
        body=api_json
    )
    api_id = response2['id']

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
    stmt_id = 'apigateway-prod-'+str(random.randint(1, 1000000))
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
            "value": '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": "*","Action": "execute-api:Invoke","Resource": "arn:aws:execute-api:'+region+':'+account_number+':'+api_id+'/prod/OPTIONS/*"}]}'
        }, ]
    )

# start here to update eval fxn integration with api resource_id_eval, lambdaevalfxnname
    if str(roles['Roles']).find("lambda_invoke_function_assume_apigw_role") > 0:
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
    else:

        response10 = user_session.resource('iam').create_role(
            AssumeRolePolicyDocument=json.dumps(lambdarole2),
            Path='/',
            RoleName='lambda_invoke_function_assume_apigw_role',
        )
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
            "value": '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": "*","Action": "execute-api:Invoke","Resource": "arn:aws:execute-api:'+region+':'+account_number+':'+api_id+'/prod/OPTIONS/*"}]}'
        }, ]
    )

    response12 = user_session.client('apigateway').create_deployment(
        restApiId=api_id,
        stageName='prod')

    result = 'https://'+api_id + '.execute-api.'+region+'.amazonaws.com/prod/m'

    return {"statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps(result)}


def get_api_json():
    apijson = '''{
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
				  },
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
				  "authorizerUri":lambda_arn,
				  "authorizerResultTtlInSeconds": 0,
				  "type": "token"
				}
			  }
			}
				}},
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
    get_api_json,
]
