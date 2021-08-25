import json
import boto3
import os
import shutil
import time
import tempfile
import zipfile
import importlib.resources as pkg_resources
from string import Template

def lambda_using_base_image(account_id, region, session, project_name, model_dir, requirements_file_path, apiid, memory_size='3000', timeout='90', python_version='3.7'):

    codebuild_bucket_name=os.environ.get("BUCKET_NAME") # s3 bucket name to create  #TODO: use same bucket and subfolder we used previously to store this data
                                                                                    #Why?  AWS limits users to 100 total buckets!  Our old code only creates one per user per acct.

    repository=project_name+'-repository' # repository name to create
    
    template_folder=tempfile.gettempdir()+'/'+project_name # folder to create for sam

    stack_name=project_name+'-stack' # stack name to be created in cloudformation

    docker_tag='latest'
    function_name=project_name
    role_name=project_name+'-lambda-role'
    policy_name=project_name+'-lambda-policy'
    
    codebuild_role_name=project_name+'-codebuild-role'
    codebuild_policies_name=project_name+'-codebuild-policies'

    codebuild_project_name=project_name+'-project'

    s3_client = session.resource('s3', region_name=region)

    s3_resource = session.resource('s3', region_name=region)

    bucket_versioning = s3_resource.BucketVersioning(codebuild_bucket_name)
    response = bucket_versioning.enable()

    ecr = session.client('ecr')

    response = ecr.create_repository(
        repositoryName=repository
    )

    iam = session.client('iam')

    import importlib_resources as pkg_resources
    #from . import sam

    os.mkdir(template_folder)

    response = shutil.copytree(model_dir, '/'.join([template_folder, model_dir]))

    def get_all_file_paths(directory):
        file_paths = []
        for root, directories, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        return file_paths

    file_paths = get_all_file_paths(template_folder)

    template_folder_len = len(template_folder)

    with zipfile.ZipFile(''.join([template_folder, '.zip']),'w') as zip:
        for file in file_paths:
            zip.write(file, file[template_folder_len:])

    s3_client = session.client('s3')
    s3_client.upload_file(''.join([template_folder, '.zip']),
                          codebuild_bucket_name,
                          ''.join([apiid, '/', project_name, '.zip']))

    iam_client = session.client('iam')
    
    response = iam_client.create_policy(
        PolicyName=policy_name,
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "logs:CreateLogStream",
                    "Resource": "*",
                    "Effect": "Allow"
                },
                {
                    "Action": "logs:PutLogEvents",
                    "Resource": "*",
                    "Effect": "Allow"
                },
                {
                    "Action": "s3:*",
                    "Resource": "*",
                    "Effect": "Allow"
                }
            ]
        })
    )
    
    time.sleep(5)

    response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        })
    )
    
    time.sleep(5)
        
    response = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::' + account_id + ':policy/' + policy_name
    )
    
    time.sleep(5)
    
    lambda_client = session.client('lambda')

    response = lambda_client.create_function(
        FunctionName=project_name,
        Role='arn:aws:iam::' + account_id + ':role/' + role_name,
        Code={
            'ImageUri': account_id + '.dkr.ecr.' + region + '.amazonaws.com/aimodelshare-base-image:latest'
        },
        PackageType="Image",
        Timeout=int(timeout),
        MemorySize=int(memory_size),
        Environment={
            'Variables': {
                'bucket': codebuild_bucket_name,
                'key': apiid + '/' + project_name + '.zip',
                'final_location': '/tmp/' + model_dir + '.zip'
            }
        }
    )
    
    s3_client.delete_object(Bucket=codebuild_bucket_name, Key=apiid)
    
    shutil.rmtree(template_folder)
    
__all__ = [
    lambda_using_base_image
]