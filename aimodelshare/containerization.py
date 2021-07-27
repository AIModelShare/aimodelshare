import json
import os
import shutil
import time
import tempfile
import zipfile
from string import Template
import importlib_resources as pkg_resources
from . import iam
from . import containerization_templates

# abstraction to return list of strings of paths of all files present in a given directory
def get_all_file_paths_in_directory(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

# abstraction to get the details of a repository
def get_repository_details(user_session, repo_name):
    ecr_client = user_session.client("ecr")
    try:
        repo_details = ecr_client.describe_repositories(repositoryNames=[repo_name])['repositories']
    except:
        repo_details = []   # no details
    return repo_details

# abstraction to get the details of an image
def get_image_details(user_session, repo_name, image_tag):
    ecr_client = user_session.client('ecr')
    try:
        image_details = ecr_client.describe_images(repositoryName=repo_name, imageIds=[{'imageTag': image_tag}])['imageDetails']
    except:
        image_details = []
    return image_details

# abstraction to create repository with name repo_name
def create_repository(user_session, repo_name):
    ecr_client = user_session.client('ecr')
    response = ecr_client.create_repository(
        repositoryName=repo_name
    )

# abstraction to upload file to S3 bucket
def upload_file_to_s3(user_session, local_file_path, bucket_name, bucket_file_path):
    s3_client = user_session.client("s3")
    s3_client.upload_file(
        local_file_path,   # path to file in the local environment
        bucket_name,    # S3 bucket name
        bucket_file_path   # path to file in the S3 bucket
    )

# abstraction to delete file from S3 bucket
def delete_file_from_s3(user_session, bucket_name, bucket_file_path):
    s3_client = user_session.client("s3")
    s3_client.delete_object(
        Bucket=bucket_name,    # S3 bucket name
        Key=bucket_file_path   # path to file in the S3 bucket
    )

# abstraction to delete IAM role
def delete_iam_role(user_session, role_name):
    iam_client = user_session.client("iam")
    response = iam_client.delete_role(
        RoleName=role_name
    )
    # keep running loop till role existence is erased
    while(True):
        try:
            response = iam_client.get_role(role_name)
        except:
            break
        time.sleep(5)

# abstraction to delete IAM policy
def delete_iam_policy(user_session, policy_name):
    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    iam_client = user_session.client("iam")
    policy_arn = "arn:aws:iam::" + account_id + ":policy/" + policy_name
    response = iam_client.delete_policy(
        PolicyArn=policy_arn
    )
    # keep running loop till policy existence is erased
    while(True):
        try:
            response = iam_client.get_policy(policy_arn)
        except:
            break
        time.sleep(5)

# abstraction to create IAM role
def create_iam_role(user_session, role_name, trust_relationship):
    try:
        delete_iam_role(user_session, role_name)
    except:
        None
    iam_client = user_session.client("iam")
    response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_relationship)     # convert JSON to string
    )
    # keep running loop till policy existence reflects
    while(True):
        try:
            response = iam_client.get_role(role_name)
            break
        except:
            None
        time.sleep(5)

# abstraction to create IAM policy
def create_iam_policy(user_session, policy_name, policy):
    try:
        delete_iam_policy(user_session, policy_name)
    except:
        None
    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    iam_client = user_session.client("iam")
    policy_arn = "arn:aws:iam::" + account_id + ":policy/" + policy_name
    response = iam_client.create_policy(
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy)     # convert JSON to string
    )
    # keep running loop till policy existence reflects
    while(True):
        try:
            response = iam_client.get_policy(policy_arn)
            break
        except:
            None
        time.sleep(5)

# abstraction to attach IAM policy to IAM role
def attach_policy_to_role(user_session, role_name, policy_name):
    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    iam_client = user_session.client("iam")
    policy_arn = "arn:aws:iam::" + account_id + ":policy/" + policy_name
    response = iam_client.attach_role_policy(
        RoleName = role_name,
        PolicyArn = policy_arn
    )
    time.sleep(5)

# build image using CodeBuild from files in zip file
def build_image(user_session, bucket_name, zip_file, image_name):

    # upload zip file to S3 bucket
    upload_file_to_s3(user_session, zip_file, bucket_name, image_name+'.zip')

    # reading JSON of the trust relationship required to create role and authorize it to use CodeBuild to build Docker image
    role_name = "codebuild_role"
    trust_relationship = json.loads(pkg_resources.read_text(iam, "codebuild_trust_relationship.txt"))
    
    # creating role for CodeBuild
    create_iam_role(user_session, role_name, trust_relationship)

    # reading JSON of all the policies that CodeBuild requires for accessing AWS services 
    policy_name = "codebuild_policy"
    policy = json.loads(pkg_resources.read_text(iam, "codebuild_policy.txt"))

    # creating policy for CodeBuild
    create_iam_policy(user_session, policy_name, policy)

    # attaching policies to role to execute CodeBuild to build Docker image
    attach_policy_to_role(user_session, role_name, policy_name)

    # creating CodeBuild project
    # specify which zip to be sourced from S3 that contains all the files to create the image
    # and specify the Linux environment that will be used to build the image
    codebuild_project_name = 'codebuild_' + image_name + '_project'
    codebuild_client = user_session.client("codebuild")
    response = codebuild_client.create_project(
        name = codebuild_project_name,
        source = {
            "type": "S3",   # where to fetch the source files from
            "location": bucket_name + "/" + image_name + ".zip"    # exact location of the zip in S3 bucket
        },
        artifacts = {
            "type": "S3",   # where to store the artifacts
            "location": bucket_name   # which bucket to store artifacts in
        },
        environment = {
            "type": "LINUX_CONTAINER",    # using a Linux environment to build the Docker image
            "image": "aws/codebuild/standard:5.0",    # type of image to use to build Docker image
            "computeType": "BUILD_GENERAL1_SMALL",    # compute type to use based on the user's choice
            "privilegedMode": True    # so that a Docker image can be built inside the image
        },
        serviceRole=role_name    # role that CodeBuild will use to build project
    )

    response = codebuild_client.start_build(
        projectName = codebuild_project_name
    )

    while(True):
        build_response = codebuild_client.batch_get_builds(ids=[response['build']['id']])
        build_status = build_response['builds'][0]['buildStatus']
        if build_status == 'SUCCEEDED':
            #response = codebuild_client.delete_project(
            #    name = codebuild_project_name
            #)
            print("Image successfully built.")
            print("CodebBuild finished with status " + build_status)
            break
        elif build_status == 'FAILED' or build_status == 'FAULT' or build_status == 'STOPPED' or build_status == 'TIMED_OUT':
            #response = codebuild_client.delete_project(
            #    name = codebuild_project_name
            #)
            print("Image not successfully built.")
            print("CodeBuild finished with status " + build_status)
            break
        time.sleep(5)

    # delete zip file from S3 bucket
    delete_file_from_s3(user_session, bucket_name, image_name+'.zip')

# create a base image containing a particular set of libraries in repository with specific image tag
def build_new_base_image(user_session, bucket_name, libraries, repository, image_tag, python_version):

    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    region = user_session.region_name

    folder_name = "base_image_folder"
    #label=",".join(libraries)      # label of image will be all string of all libraries

    # temporary folder path where we will create all files and folder
    temp_dir = tempfile.gettempdir() + "/" + folder_name

    if(os.path.isdir(temp_dir)):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    # list of all Python libraries (with their versions if required) required to be downloaded from PyPI into Docker image
    with open(os.path.join(temp_dir, "requirements.txt"), "a") as f:
        for lib in libraries:
            f.write('%s\n' % lib)

    # Dockerfile.txt template being read and appropriate variables being assigned to generate Dockerfile
    data = pkg_resources.read_text(containerization_templates, "Dockerfile.txt")   # read template from containerization folder
    template = Template(data)
    newdata = template.substitute(
        python_version=python_version)  # AWS maintained images with speicific python versions
    with open(os.path.join(temp_dir, "Dockerfile"), "w") as file:
        file.write(newdata)

    # buildspec.txt template being read and appropriate variables being assigned to generate buildspec.yml
    data = pkg_resources.read_text(containerization_templates, "buildspec.txt")   # read template from containerization folder
    template = Template(data)
    newdata = template.substitute(
        account_id=account_id,      # AWS account id
        region=region,      # region in which the repository is / should be created
        repository=repository,      # name of the repository
        image_tag=image_tag,        # version / tag to be given to the image
        label="test")     #label of the library
    with open(os.path.join(temp_dir, "buildspec.yml"), "w") as file:
        file.write(newdata)

    # lambda_function.py being generated which has the handler that will be called when Docker image is invoked
    data = pkg_resources.read_text(containerization_templates, "lambda_function.txt")   # read template from containerization folder
    with open(os.path.join(temp_dir, "lambda_function.py"), "w") as file:
        file.write(data)

    file_paths = get_all_file_paths_in_directory(temp_dir)    # getting list of strings containing paths of all files

    # zipping all files in the temporary folder to be uploaded to the S3 bucket
    with zipfile.ZipFile(temp_dir + ".zip", "w") as zip:
        for file in file_paths:
            zip.write(file, file.replace(temp_dir, ""))      # ignore temporary file path when copying to zip file

    build_image(user_session, bucket_name, temp_dir + ".zip", repository + "_" + image_tag + "_base_image")

# create lambda function using a base image from a specific repository having a specific tag
def create_lambda_using_base_image(user_session, bucket_name, directory, lambda_name, api_id, repository, image_tag, memory_size, timeout):

    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    region = user_session.region_name

    temp_dir = tempfile.gettempdir() + "/" + lambda_name

    if(os.path.isdir(temp_dir)):
        shutil.rmtree(temp_dir)

    shutil.copytree(directory, temp_dir)     # copying files from the local directory to tmp folder directory

    temp_path_directory_file_paths = get_all_file_paths_in_directory(temp_dir)    # getting list of strings containing paths of all files

    # zipping all files in a temporary folder to be uploaded to the S3 bucket
    with zipfile.ZipFile(temp_dir + ".zip", "w") as zip:    # remove temporary path from directory if present
        for file in temp_path_directory_file_paths:
            zip.write(file, file.replace(temp_dir, ""))    # ignore temporary path when copying to zip file

    upload_file_to_s3(user_session, temp_dir + ".zip", bucket_name, lambda_name + ".zip")        # upload zip file to S3 bucket

    # reading JSON of the trust relationship required to create role and authorize it to use CodeBuild to build Docker image
    role_name = "lambda_role"
    trust_relationship = json.loads(pkg_resources.read_text(iam, "lambda_trust_relationship.txt"))
    
    # creating role for CodeBuild
    create_iam_role(user_session, role_name, trust_relationship)

    # reading JSON of all the policies that CodeBuild requires for accessing AWS services 
    policy_name = "lambda_policy"
    policy = json.loads(pkg_resources.read_text(iam, "lambda_policy.txt"))

    # creating policy for CodeBuild    
    create_iam_policy(user_session, policy_name, policy)

    # attaching policies to role to execute CodeBuild to build Docker image
    attach_policy_to_role(user_session, role_name, policy_name)

    lambda_client = user_session.client('lambda')
    response = lambda_client.create_function(
        FunctionName=lambda_name,
        Role = 'arn:aws:iam::' + account_id + ':role/' + role_name,
        Code = {
            'ImageUri': account_id + '.dkr.ecr.' + region + '.amazonaws.com/' + repository + ":" + image_tag
        },
        PackageType = "Image",
        Timeout = int(timeout),
        MemorySize = int(memory_size),
        Environment = {
            'Variables': {
                'bucket': bucket_name,     # bucket where zip file is located
                'api_id': api_id,     # api_id in the bucket in which zip file is stored
                'directory': lambda_name        # directory in which all files exist
            }
        }
    )

    os.remove(temp_dir + ".zip")     # delete the zip file created in tmp directory
    shutil.rmtree(temp_dir)      # delete the temporary folder created in tmp directory