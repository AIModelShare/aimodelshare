#import docker
import os
import shutil
import importlib.resources as pkg_resources
#import importlib_resources as pkg_resources
from . import data_sharing_templates
from string import Template
import zipfile
import time
import json
import boto3
import tempfile
import requests
import uuid

def create_docker_folder_local(dataset_dir, dataset_name, python_version):

    tmp_dataset_dir = tempfile.gettempdir() + '/' + '/'.join(['tmp_dataset_dir', dataset_name])

    tmp_dataset = tempfile.gettempdir() + '/' + 'tmp_dataset_dir'

    os.mkdir(tmp_dataset)

    shutil.copytree(dataset_dir, tmp_dataset_dir)

    #data = pkg_resources.read_text(data_sharing_templates, 'Dockerfile.txt')
    with open(os.path.join('data_sharing_templates', 'Dockerfile.txt'), 'r') as file:
        data = file.read()

    template = Template(data)
    newdata = template.substitute(
        python_version=python_version)
    with open(os.path.join(tmp_dataset, 'Dockerfile'), 'w') as file:
        file.write(newdata)

def create_docker_folder_codebuild(dataset_dir, dataset_name, template_folder, region, registry_uri, repository, dataset_tag, python_version):

    tmp_dataset_dir = tempfile.gettempdir() + '/' + '/'.join(['tmp_dataset_dir', dataset_name])

    tmp_dataset = tempfile.gettempdir() + '/' + 'tmp_dataset_dir'

    os.mkdir(tmp_dataset)

    """
    if not os.path.exists(tmp_dataset):
        os.makedirs(tmp_dataset)
    else:
        shutil.rmtree(tmp_dataset)
        os.makedirs(tmp_dataset)
    """

    shutil.copytree(dataset_dir, tmp_dataset_dir)

    os.mkdir(template_folder)

    data = pkg_resources.read_text(data_sharing_templates, 'Dockerfile.txt')
    #with open(os.path.join('data_sharing_templates', 'Dockerfile.txt'), 'r') as file:
    #    data = file.read()

    template = Template(data)
    newdata = template.substitute(
        python_version=python_version)
    with open(os.path.join(template_folder, 'Dockerfile'), 'w') as file:
        file.write(newdata)
    
    data = pkg_resources.read_text(data_sharing_templates, 'buildspec.txt')
    #with open(os.path.join('data_sharing_templates', 'buildspec.txt'), 'r') as file:
    #    data = file.read()

    template = Template(data)
    newdata = template.substitute(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region=region,
        registry_uri=registry_uri,
        repository=repository,
        dataset_tag=dataset_tag)
    with open(os.path.join(template_folder, 'buildspec.yml'), 'w') as file:
        file.write(newdata)

    response = shutil.copytree(tmp_dataset, '/'.join([template_folder, 'tmp_dataset_dir']))

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
            try:
                zip.write(file, file[template_folder_len:])
            except:
                pass

    shutil.rmtree(tmp_dataset)
    
    shutil.rmtree(template_folder)

#def share_data_local(dataset_dir, tag='latest', python_version='3.8'):

    #create_docker_folder_local(dataset_dir)

    #client = docker.from_env()

    #client.images.build(path='./tmp_dataset_folder', tag=tag)

    #shutil.rmtree('tmp_dataset_dir')

    # send to ecr
    
    # client.images.

    # client.push(repository,

    # ecr_client.create_repository

    # docker tag aimodelshare-base-image:latest 517169013426.dkr.ecr.us-east-1.amazonaws.com/aimodelshare-base-image:latest
    # aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
    # docker push public.ecr.aws/y2e2a1d6/aimodelshare-image-classification-public

def share_data_codebuild(account_id, region, dataset_dir, dataset_tag='latest', python_version='3.8'):

    print('Uploading your data. Please wait for a confirmation message.')
    
    session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                    region_name=os.environ.get("AWS_REGION"))

    flag = 0

    dataset_name = dataset_dir.replace(" ", "_")

    repository=dataset_name+'-repository'

    template_folder=tempfile.gettempdir() + '/' + dataset_name+'_'+dataset_tag

    codebuild_role_name=dataset_name+'-codebuild-role'
    codebuild_policies_name=dataset_name+'-codebuild-policies'

    codebuild_dataset_name=dataset_name+'-upload'

    s3_client = session.resource('s3', region_name=region)

    bucket_name = "aimodelshare"+str(account_id)+"sharedata"
    
    try:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration = {
                'LocationConstraint': region
            }
        )
    except:
        s3_client.create_bucket(
            Bucket=bucket_name
        )
    
    s3_resource = session.resource('s3', region_name=region)

    bucket_versioning = s3_resource.BucketVersioning(bucket_name)
    response = bucket_versioning.enable()

    # ecr = session.client('ecr')

    ecr = session.client('ecr-public')

    registry_uri = ecr.describe_registries()['registries'][0]['registryUri']

    try:
        response = ecr.create_repository(
            repositoryName=repository
        )
    except:
        pass

    create_docker_folder_codebuild(dataset_dir, dataset_name, template_folder, region, registry_uri, repository, dataset_tag, python_version)

    iam = session.client('iam')
    
    codebuild_trust_relationship = json.loads(pkg_resources.read_text(data_sharing_templates, 'codebuild_trust_relationship.txt'))
    #with open(os.path.join('data_sharing_templates', 'codebuild_trust_relationship.txt'), 'r') as file:
    #    codebuild_trust_relationship = json.load(file)

    try:
        response = iam.create_role(
            RoleName=codebuild_role_name,
            AssumeRolePolicyDocument=json.dumps(codebuild_trust_relationship)
        )
    except:
        flag = 1
        pass

    codebuild_policies = json.loads(pkg_resources.read_text(data_sharing_templates, 'codebuild_policies.txt'))
    #with open(os.path.join('data_sharing_templates', 'codebuild_policies.txt'), 'r') as file:
    #    codebuild_policies = json.load(file)

    try:
        response = iam.create_policy(
            PolicyName=codebuild_policies_name,
            PolicyDocument=json.dumps(codebuild_policies),
        )
    except:
        flag = 1
        pass

    response = iam.attach_role_policy(
        RoleName=codebuild_role_name,
        PolicyArn=''.join(['arn:aws:iam::', account_id, ':policy/', codebuild_policies_name])
    )

    s3_client = session.client('s3')
    s3_client.upload_file(''.join([template_folder, '.zip']),
                          bucket_name,
                          ''.join([dataset_name+'_'+dataset_tag, '.zip']))

    if(flag==0):
        time.sleep(15)

    codebuild = session.client('codebuild')
    
    try:
        response = codebuild.create_project(
            name=codebuild_dataset_name,
            source={
                'type': 'S3',
                'location': bucket_name + '/' + dataset_name+'_'+dataset_tag + '.zip'
            },
            artifacts={
                'type': 'NO_ARTIFACTS',
            },
            environment={
                'type': 'LINUX_CONTAINER',
                'image': 'aws/codebuild/standard:5.0',
                'computeType': 'BUILD_GENERAL1_SMALL',
                'privilegedMode': True
            },
            serviceRole=codebuild_role_name
        )
    except:
        response = codebuild.delete_project(
            name=codebuild_dataset_name
        )
        response = codebuild.create_project(
            name=codebuild_dataset_name,
            source={
                'type': 'S3',
                'location': bucket_name + '/' + dataset_name+'_'+dataset_tag + '.zip'
            },
            artifacts={
                'type': 'NO_ARTIFACTS',
            },
            environment={
                'type': 'LINUX_CONTAINER',
                'image': 'aws/codebuild/standard:5.0',
                'computeType': 'BUILD_GENERAL1_SMALL',
                'privilegedMode': True
            },
            serviceRole=codebuild_role_name
        )

    response = codebuild.start_build(
        projectName=codebuild_dataset_name
    )

    os.remove(template_folder+'.zip')

    return {"ecr_uri":registry_uri + '/' + repository + ':' + dataset_tag}

def share_dataset(data_directory="folder_file_path",classification="default", private="FALSE"):
    data_directory=str(data_directory).lower()
    aishare_datasetname = input("Enter dataset name:")
    aishare_datadescription = input(
        "Enter data description (i.e.- filenames denoting training and test data, file types, and any subfolders where files are stored):")
    aishare_datatags = input(
        "Enter tags to help users find your data (i.e.- flower dataset, image, supervised learning, classification")   
    datalicense=input("Insert license (Optional): ")
    datacitation=input("Insert citation (Optional): ")
    modelplaygroundurl=input("Insert AI Model Share model playground url (Optional): ")
    problemdomain=input("Enter a number signifying your dataset problem domain or data type: 1 = Image 2 = Video 3 = Text 4 = Tabular 5 = Neural Style Transfer 6 = Object Detection 7 = Other \n")

    optiondict={"1":"Image", "2":"Video","3":"Text","4":"Tabular", "5": "Audio","6":"Neural Style Transfer", "7": "Object Detection", "8":"Other"}
    problemdomainfinal=optiondict.get(problemdomain,"Other")


    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                          aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"), 
                                         region_name=os.environ.get("AWS_REGION"))
    account_number = user_session.client(
        'sts').get_caller_identity().get('Account')

    datauri=share_data_codebuild(account_number,os.environ.get("AWS_REGION"),data_directory)


    #TODO: Replace redis data code with new api post call, see helper code below from competition api
    #TODO: add "dataset:id" as models are ingested on the backend
    bodydata = {"dataowner": os.environ.get("username"),  # change this to first and last name
                "dataname":aishare_datasetname ,                
                'datadescription':aishare_datadescription,
                'datatags':aishare_datatags,
                'dataecruri':datauri['ecr_uri'],
                'datalicense':datalicense,
                'datacitation':datacitation,
                'classification':classification,
                "modelplaygroundurl": modelplaygroundurl,
                "Private": private,
                "delete": "FALSE",
                "problemdomain":problemdomainfinal}

    # datasets api
    headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                   'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
    # modeltoapi lambda function invoked through below url to return new prediction api in response
    response=requests.post("https://jyz9nn0joe.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                  json=bodydata, headers=headers_with_authentication)
    return "Your dataset has been shared to modelshare.org."
