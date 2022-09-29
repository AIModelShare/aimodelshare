import json
import boto3
import os
import shutil
import tempfile
import time
from aimodelshare.aws import get_s3_iam_client, run_function_on_lambda, get_token, get_aws_token, get_aws_client

import importlib.resources as pkg_resources
from string import Template

def create_bucket(s3_client, bucket_name, region):
        try:
            response=s3_client.head_bucket(Bucket=bucket_name)
        except:
            if(region=="us-east-1"):
                response = s3_client.create_bucket(
                    ACL="private",
                    Bucket=bucket_name
                )
            else:
                location={'LocationConstraint': region}
                response=s3_client.create_bucket(
                    ACL="private",
                    Bucket=bucket_name,
                    CreateBucketConfiguration=location
                )
        return response
    
def deploy_container(account_id, region, session, project_name, model_dir, requirements_file_path, apiid, memory_size='1024', timeout='120', python_version='3.7', pyspark_support=False):

    codebuild_bucket_name=os.environ.get("BUCKET_NAME") # s3 bucket name to create  #TODO: use same bucket and subfolder we used previously to store this data
                                                        # Why? AWS limits users to 100 total buckets!  Our old code only creates one per user per acct.

    repository=project_name.lower()+'repository' # repository name to create

    template_folder=tempfile.gettempdir()+'/'+project_name # folder to create for sam

    stack_name=project_name+'-stack' # stack name to be created in cloudformation

    docker_tag='latest'
    function_name=project_name
    role_name=project_name+'-lambda-role'
    policy_name=project_name+'-lambda-policy'
    
    codebuild_role_name=project_name+'-codebuild-role'
    codebuild_policies_name=project_name+'-codebuild-policies'

    codebuild_project_name=project_name+'-project'

    aws_access_key_id = str(os.environ.get("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = str(os.environ.get("AWS_SECRET_ACCESS_KEY"))
    region_name = str(os.environ.get("AWS_REGION"))
        
    s3, iam, region = get_s3_iam_client(aws_access_key_id, aws_secret_access_key, region_name)
    create_bucket(s3['client'], codebuild_bucket_name, region)

    s3_resource = session.resource('s3', region_name=region)

    bucket_versioning = s3_resource.BucketVersioning(codebuild_bucket_name)
    response = bucket_versioning.enable()

    ecr = session.client('ecr')
    
    #check repo name for issues
    os.environ["repository"] = repository.lower()
    response = ecr.create_repository(
        repositoryName=repository.lower()
    )

    iam = session.client('iam')

    import importlib_resources as pkg_resources
    from . import sam

    codebuild_trust_relationship = json.loads(pkg_resources.read_text(sam, 'codebuild_trust_relationship.txt'))
    #with open(os.path.join('sam', 'codebuild_trust_relationship.txt'), 'r') as file:
    #    codebuild_trust_relationship = json.load(file)

    response = iam.create_role(
        RoleName=codebuild_role_name,
        AssumeRolePolicyDocument=json.dumps(codebuild_trust_relationship)
    )

    codebuild_policies = json.loads(pkg_resources.read_text(sam, 'codebuild_policies.txt'))
    #with open(os.path.join('sam', 'codebuild_policies.txt'), 'r') as file:
    #    codebuild_policies = json.load(file)

    response = iam.create_policy(
        PolicyName=codebuild_policies_name,
        PolicyDocument=json.dumps(codebuild_policies),
    )

    response = iam.attach_role_policy(
        RoleName=codebuild_role_name,
        PolicyArn=''.join(['arn:aws:iam::', account_id, ':policy/', codebuild_policies_name])
    )

    os.mkdir(template_folder)
    os.mkdir('/'.join([template_folder, 'app']))

    #####

    data = pkg_resources.read_text(sam, 'buildspec.txt')
    #with open(os.path.join('sam', 'buildspec.txt'), 'r') as file:
    #    data = file.read()

    template = Template(data)
    newdata = template.substitute(
        account_id=account_id,
        region=region, #os.environ.get("region"),
        repository_name=repository,
        stack_name=stack_name)
    with open(os.path.join(template_folder, 'buildspec.yml'), 'w') as file:
        file.write(newdata)

    #####

    data = pkg_resources.read_text(sam, 'template.txt')
    #with open(os.path.join('sam', 'template.txt'), 'r') as file:
    #    data = file.read()

    #modfunction756350

    template = Template(data)
    newdata = template.substitute(
        image_tag=docker_tag, #os.environ.get("docker_tag"),
        role_name=role_name,
        policy_name=policy_name,
        function_name=function_name,
        memory_size=memory_size,
        timeout=timeout)
    with open(os.path.join(template_folder, 'template.yml'), 'w') as file:
        file.write(newdata)

    #####
    if pyspark_support:
        data = pkg_resources.read_text(sam, 'Dockerfile_PySpark.txt')
        #with open(os.path.join('sam', 'Dockerfile.txt'), 'r') as file:
        #    data = file.read()

        template = Template(data)
        newdata = template.substitute(
            python_version=python_version,
            directory=model_dir,
            requirements_file_path=requirements_file_path,
            PATH="$PATH",
            SPARK_HOME="$SPARK_HOME",
            PYTHONPATH="$PYTHONPATH",
            JAVA_HOME="$JAVA_HOME"
        )

        with open(os.path.join('/'.join([template_folder, 'app']), 'Dockerfile'), 'w') as file:
            file.write(newdata)

        data = pkg_resources.read_text(sam, 'spark-class.txt')
        
        template = Template(data)
        newdata = template.substitute(
            python_version=python_version,
            var="$@"
        )
        
        with open(os.path.join('/'.join([template_folder, 'app']), 'spark-class'), 'w') as file:
            file.write(newdata)
    else:
        data = pkg_resources.read_text(sam, 'Dockerfile.txt')
        #with open(os.path.join('sam', 'Dockerfile.txt'), 'r') as file:
        #    data = file.read()

        template = Template(data)
        newdata = template.substitute(
            python_version=python_version,
            directory=model_dir,
            requirements_file_path=requirements_file_path)
        with open(os.path.join('/'.join([template_folder, 'app']), 'Dockerfile'), 'w') as file:
            file.write(newdata)
        
    response = shutil.copytree(model_dir, '/'.join([template_folder, 'app', model_dir]))

    import zipfile

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
                          
    codebuild = session.client('codebuild')
    time.sleep(15)
    response = codebuild.create_project(
        name=codebuild_project_name,
        source={
            'type': 'S3',
            'location': codebuild_bucket_name + '/'  + apiid + '/' + project_name + '.zip'
        },
        artifacts={
            'type': 'S3',
            'location': codebuild_bucket_name 
        },
        environment={
            'type': 'LINUX_CONTAINER',
            'image': 'aws/codebuild/standard:5.0',
            'computeType': 'BUILD_GENERAL1_SMALL',
            'privilegedMode': True
        },
        serviceRole=codebuild_role_name,
    )

    response = codebuild.start_build(
        projectName=codebuild_project_name
    )

    while(True):
        theBuild = codebuild.batch_get_builds(ids=[response['build']['id']])
        buildStatus = theBuild['builds'][0]['buildStatus']
        if buildStatus == 'SUCCEEDED':
            buildSucceeded = True
            break
        elif buildStatus == 'FAILED' or buildStatus == 'FAULT' or buildStatus == 'STOPPED' or buildStatus == 'TIMED_OUT':
            print("container failed to build on codebuild "+buildStatus)
            break
        time.sleep(10)

    s3_client = session.client('s3')
    s3_client.delete_object(Bucket=codebuild_bucket_name,
                            Key=apiid)

    shutil.rmtree(template_folder)
