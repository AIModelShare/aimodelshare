import json
import os
import shutil
import time
import tempfile
import zipfile
from string import Template
import sys
import boto3
import importlib_resources as pkg_resources

import uuid
import requests

time_delay=2

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
    #print("Fetching details of repository \"" + repo_name + "\".")
    ecr_client = user_session.client("ecr")
    try:
        repo_details = ecr_client.describe_repositories(repositoryNames=[repo_name])['repositories']
    except:
        repo_details = []   # no details
    return repo_details

# abstraction to get the details of an image
def get_image_details(user_session, repo_name, image_tag):
    #print("Fecthing details of image \"" + repo_name + ":" + image_tag + "\".")
    ecr_client = user_session.client('ecr')
    try:
        image_details = ecr_client.describe_images(repositoryName=repo_name, imageIds=[{'imageTag': image_tag}])['imageDetails']
        #print("Fetched details of image \"" + repo_name + ":" + image_tag + "\" successfully.")
    except:
        #print("No such image \"" + repo_name + ":" + image_tag + "\" exists.")
        image_details = []
    return image_details

# abstraction to create repository with name repo_name
def create_repository(user_session, repo_name):
    #print("Creating repository \"" + repo_name + "\".")
    ecr_client = user_session.client('ecr')
    response = ecr_client.create_repository(
        repositoryName=repo_name
    )
    #print("Created repository  \"" + repo_name + "\" successfully.")

# abstraction to upload file to S3 bucket
def upload_file_to_s3(user_session, local_file_path, bucket_name, bucket_file_path):
    #print("Uploading function files to \"" + bucket_file_path +"\".")
    s3_client = user_session.client("s3")
    s3_client.upload_file(
        local_file_path,   # path to file in the local environment
        bucket_name,    # S3 bucket name
        bucket_file_path   # path to file in the S3 bucket
    )
    #print("Uploaded function files to \"" + bucket_file_path +"\" successfully.")

# abstraction to delete file from S3 bucket
def delete_file_from_s3(user_session, bucket_name, bucket_file_path):
    #print("Deleting file \"" + bucket_file_path +"\".")
    s3_client = user_session.client("s3")
    s3_client.delete_object(
        Bucket=bucket_name,    # S3 bucket name
        Key=bucket_file_path   # path to file in the S3 bucket
    )
    #print("Deleted file \"" + bucket_file_path +"\" successfully.")

# abstraction to delete IAM role
def delete_iam_role(user_session, role_name):
    #print("Deleting IAM role \"" + role_name + "\".")
    iam_client = user_session.client("iam")
    # see if role exists
    try:
        response = iam_client.get_role(RoleName=role_name)
    except:
        return
    # once role's existence is verified, fetch all attached policies
    response = iam_client.list_attached_role_policies(
        RoleName=role_name
    )
    # detach the policy from the role
    policies = response['AttachedPolicies']
    for policy in policies:
        response = iam_client.detach_role_policy(
            RoleName=role_name,
            PolicyArn=policy['PolicyArn']
        )
    # delete role
    response = iam_client.delete_role(
        RoleName=role_name
    )
    # give time to reflect in IAM
    time.sleep(time_delay)
    #print("Deleted IAM role \"" + role_name + "\" successfully.")

# abstraction to delete IAM policy
def delete_iam_policy(user_session, policy_name):

    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    
    #print("Deleting IAM policy \"" + policy_name + "\".")
    iam_client = user_session.client("iam")
    policy_arn = "arn:aws:iam::" + account_id + ":policy/" + policy_name
    # see if policy exists
    try:
        response = iam_client.get_policy(PolicyArn=policy_arn)
    except:
        return
    # once policy's existence is verified, delete policy
    response = iam_client.delete_policy(
        PolicyArn=policy_arn
    )
    # give time to reflect in IAM
    time.sleep(time_delay)
    #print("Deleted IAM policy \"" + policy_name + "\" successfully.")

# abstraction to create IAM role
def create_iam_role(user_session, role_name, trust_relationship):
    #print("Creating IAM role \"" + role_name + "\".")
    iam_client = user_session.client("iam")
    response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_relationship)     # convert JSON to string
    )
    # give time to reflect in IAM
    time.sleep(time_delay)
    #print("Created IAM role \"" + role_name + "\" successfully.")
    
# abstraction to create IAM policy
def create_iam_policy(user_session, policy_name, policy):

    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    
    #print("Creating IAM policy \"" + policy_name + "\".")
    iam_client = user_session.client("iam")
    policy_arn = "arn:aws:iam::" + account_id + ":policy/" + policy_name
    response = iam_client.create_policy(
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy)     # convert JSON to string
    )
    # give time to reflect in IAM
    time.sleep(time_delay)
    #print("Created IAM policy \"" + policy_name + "\" successfully.")

# abstraction to attach IAM policy to IAM role
def attach_policy_to_role(user_session, role_name, policy_name):

    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    
    #print("Attaching IAM policy \"" + policy_name +"\" to IAM role \"" + role_name + "\".")
    iam_client = user_session.client("iam")
    policy_arn = "arn:aws:iam::" + account_id + ":policy/" + policy_name
    response = iam_client.attach_role_policy(
        RoleName = role_name,
        PolicyArn = policy_arn
    )
    # give time to reflect in IAM
    time.sleep(time_delay)
    #print("Attached IAM policy \"" + policy_name +"\" to IAM role \"" + role_name + "\" successfully.")

# build image using CodeBuild from files in zip file
def build_image(user_session, bucket_name, zip_file, image_name, image):

    from . import iam

    # upload zip file to S3 bucket
    upload_file_to_s3(user_session, zip_file, bucket_name, image_name+'.zip')

    # reading JSON of the trust relationship required to create role and authorize it to use CodeBuild to build Docker image
    role_name = "codebuild_role"
    trust_relationship = json.loads(pkg_resources.read_text(iam, "codebuild_trust_relationship.txt"))
    
    # delete role for CodeBuild if role with same name exists
    delete_iam_role(user_session, role_name)

    # creating role for CodeBuild
    create_iam_role(user_session, role_name, trust_relationship)

    # reading JSON of all the policies that CodeBuild requires for accessing AWS services 
    policy_name = "codebuild_policy"
    policy = json.loads(pkg_resources.read_text(iam, "codebuild_policy.txt"))

    # delete policy for CodeBuild if policy with same name exists
    delete_iam_policy(user_session, policy_name)

    # creating policy for CodeBuild
    create_iam_policy(user_session, policy_name, policy)

    # attaching policies to role to execute CodeBuild to build Docker image
    attach_policy_to_role(user_session, role_name, policy_name)

    time.sleep(5)

    # creating CodeBuild project
    # specify which zip to be sourced from S3 that contains all the files to create the image
    # and specify the Linux environment that will be used to build the image
    codebuild_project_name = 'codebuild_' + image_name.replace("/", "_") + '_project'
    codebuild_client = user_session.client("codebuild")
    counter=1
    while(counter<=3):
        try:
            print("Attempt " + str(counter) + " to create CodeBuild project.")
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
            break
        except:
            counter+=1
            if(counter<=3):
                print("CodeBuild project creation failed. Waiting for dependent resources to reflect. Retrying again in 10 seconds.")
                time.sleep(time_delay)
            else:
                print("CodeBuild project creation failed.")
                delete_file_from_s3(user_session, bucket_name, image_name+'.zip')   # delete zip file from S3 bucket
                return

    # running through loop while build status shows termination/successful completion
    counter=0
    while(True):
        build_response = codebuild_client.batch_get_builds(ids=[response['build']['id']])
        build_status = build_response['builds'][0]['buildStatus']
        if build_status == 'SUCCEEDED':
            response = codebuild_client.delete_project(     # delete CodeBuild project after process completes
                name = codebuild_project_name
            )
            print("Image successfully built.")
            print("CodeBuild finished with status " + build_status)
            print("\nNew Base Image URI: " + image)
            break
        elif build_status == 'FAILED' or build_status == 'FAULT' or build_status == 'STOPPED' or build_status == 'TIMED_OUT':
            response = codebuild_client.delete_project(     # delete CodeBuild project after process completes
                name = codebuild_project_name
            )
            print("Image not successfully built.")
            print("CodeBuild finished with status " + build_status)
            break        
        sys.stdout.write('\r')
        sys.stdout.write("Waiting" + "."*counter)
        sys.stdout.flush()
        counter=(counter+1)%4
        time.sleep(1)

    # delete zip file from S3 bucket
    delete_file_from_s3(user_session, bucket_name, image_name+'.zip')

# create a base image containing a particular set of libraries in repository with specific image tag
def build_new_base_image(libraries, repository, image_tag, python_version):

    from . import containerization_templates

    user_session = boto3.session.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                         aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"),
                                         region_name=os.environ.get("AWS_REGION"))

    bucket_name = os.environ.get("BUCKET_NAME")

    s3_client = user_session.client("s3")
    
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

    create_bucket(s3_client, bucket_name, os.environ.get("AWS_REGION"))

    print("S3 Bucket \"" + bucket_name + "\" used for all storage purposes.")
    
    unique_name = repository + "_" + image_tag

    try:
        create_repository(user_session, repository)
    except:
        print("Repository already exists.")

    sts_client = user_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    region = user_session.region_name

    print("Building new base image.")

    folder_name = "unique_name"
    ##################################################
    #label = "libraries=" + ",".join(libraries)      # label of image will be all string of all libraries
    label = "libraries=test"      # label of image will be all string of all libraries
    ##################################################

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
        label=label)     #label of the library
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

    build_image(user_session, bucket_name, temp_dir + ".zip", unique_name + "_base_image", repository+":"+image_tag)

    if(os.path.isdir(temp_dir)):
        shutil.rmtree(temp_dir)

    return

# create lambda function using a base image from a specific repository having a specific tag
def create_lambda_using_base_image(user_session, bucket_name, directory, lambda_name, api_id, repository, image_tag, memory_size, timeout):

    from . import iam

    cloudformation_client = user_session.client('cloudformation')

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

    upload_file_to_s3(user_session, temp_dir + ".zip", bucket_name, api_id + "/" + lambda_name + ".zip")        # upload zip file to S3 bucket

    # reading JSON of the trust relationship required to create role and authorize it to use CodeBuild to build Docker image
    role_name = "lambda_role_" + api_id
    trust_relationship = json.loads(pkg_resources.read_text(iam, "lambda_trust_relationship.txt"))
    
    # creating role for CodeBuild
    create_iam_role(user_session, role_name, trust_relationship)

    # reading JSON of all the policies that CodeBuild requires for accessing AWS services 
    policy_name = "lambda_policy_" + api_id
    policy = json.loads(pkg_resources.read_text(iam, "lambda_policy.txt"))

    # creating policy for CodeBuild    
    create_iam_policy(user_session, policy_name, policy)

    # attaching policies to role to execute CodeBuild to build Docker image
    attach_policy_to_role(user_session, role_name, policy_name)

    #print("Creating Lambda function \"" + lambda_name + "\".")

    role = 'arn:aws:iam::' + account_id + ':role/' + role_name
    code = {
        'ImageUri': account_id + '.dkr.ecr.' + region + '.amazonaws.com/' + repository + ":" + image_tag
    }
    package_type = "Image"
    timeout = int(timeout)
    memory_size = int(memory_size)
    environment = {
        'Variables': {
            'bucket': bucket_name,     # bucket where zip file is located
            'api_id': api_id,     # api_id in the bucket in which zip file is stored
            'function_name': lambda_name,        # Lambda function name
            'NUMBA_CACHE_DIR': '/tmp'
        }
    }

    lambda_client = user_session.client("lambda")
    counter=1
    while(counter<=3):
        try:
            #print("Attempt " + str(counter) + " to create Lambda function.")
            response_lambda = lambda_client.create_function(
                FunctionName=lambda_name,
                Role = role,
                Code = code,
                PackageType = package_type,
                Timeout = timeout,
                MemorySize = memory_size,
                Environment = environment
            )
            break
        except:
            counter+=1
            if(counter<=3):
                time.sleep(5)
            else:
                print("Lambda function creation failed.")

    # running through loop until function reflects
    counter=1
    while(counter<=3):
        try:
            response = lambda_client.get_function(
                FunctionName=lambda_name
            )
            #print("Created Lambda function " + lambda_name + "\" successfully.")
            break
        except:
            counter+=1
            if(counter<=3):
                #print("Lambda function did not reflected. Waiting for Lambda function to reflect.")
                time.sleep(5)
            else:
                None
                #print("Lambda function did not reflected.")

    # stack_name = 'stack-' + api_id
    # role_name = "lambda_role_" + api_id
    # trust_policy = json.loads(pkg_resources.read_text(iam, "lambda_trust_relationship.txt"))
    # policy_name = "lambda_policy_" + api_id
    # policy = json.loads(pkg_resources.read_text(iam, "lambda_policy.txt"))
    
    # role = 'arn:aws:iam::' + account_id + ':role/' + role_name
    # code = {
    #     'ImageUri': account_id + '.dkr.ecr.' + region + '.amazonaws.com/' + repository + ":" + image_tag
    # }
    # package_type = "Image"
    # timeout = int(timeout)
    # memory_size = int(memory_size)
    # environment = {
    #     'Variables': {
    #         'bucket': bucket_name,     # bucket where zip file is located
    #         'api_id': api_id,     # api_id in the bucket in which zip file is stored
    #         'function_name': lambda_name,        # Lambda function name
    #         'NUMBA_CACHE_DIR': '/tmp'
    #     }
    # }
    # arn_prefix = "arn:aws:execute-api:" + region + ":" + account_id + ":" + api_id
    # template_body = get_cloudformation_template()
    # template = Template(template_body)
    # new_template = template.substitute(
    #     PolicyDocument = json.dumps(policy),
    #     PolicyName = policy_name,
    #     TrustPolicy = json.dumps(trust_policy),
    #     Policies = json.dumps([policy_name]),
    #     RoleName = role_name,
    #     Code = json.dumps(code),
    #     Environment = json.dumps(environment),
    #     FunctionName = lambda_name,
    #     MemorySize = memory_size,
    #     PackageType = package_type,
    #     Timeout = timeout,
    #     Action1 = 'lambda:InvokeFunction',
    #     Principal1 = 'apigateway.amazonaws.com',
    #     SourceArn1 = arn_prefix + "/*/POST/m"
    # )
    # template_body = new_template

    # response = cloudformation_client.create_stack(
    #     StackName = stack_name,
    #     TemplateBody = template_body,
    #     DisableRollback=False,
    #     Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM', 'CAPABILITY_AUTO_EXPAND']
    # )

    os.remove(temp_dir + ".zip")    # delete the zip file created in tmp directory
    if(os.path.isdir(temp_dir)):    # delete the temporary folder created in tmp directory
        shutil.rmtree(temp_dir)

    return response

# check if the image exists in the specified repository with specified image tag
def check_if_image_exists(user_session, repo_name, image_tag):
    #print("Checking if image \"" + repo_name + ":" + image_tag +"\" exists.")
    import datetime
    ecr_client = user_session.client('ecr')
    try:
        image_details = ecr_client.describe_images(
            repositoryName=repo_name,
            imageIds=[{'imageTag': image_tag}]
        )
        result = True
        #adding date check of last image updates for any image tags.
        #if user has old image then we update it using this approach
        #result=image_details['imageDetails'][0]['imagePushedAt'].date()<=datetime.date(2021, 10, 6)

        #print("The image " + "\"" + repo_name + ":" + image_tag + "\"" + " exists.")
        # print details of image
        #print(image_details)
        return result
    except:
        #print("The image " + "\"" + repo_name + ":" + image_tag + "\"" + " does not exist.")
        return False

# check if repo exists
def check_if_repo_exists(user_session, repo_name):
    #print("Checking if repository \"" + repo_name + "\" exists.")
    ecr_client = user_session.client("ecr")
    try:
        repo_details = ecr_client.describe_images(
            repositoryName=repo_name
        )
        #print("The repository \"" + repo_name + "\" exists.")
        # print details all images in repo
        #print(repo_details)
        return True
    except:
        #print("The repository \"" + repo_name + "\" does not exist.")
        return False

# delete statement with specific Sid in ECR
def delete_registry_permission(user_session, statement_id):
    ecr_client = user_session.client("ecr")
    try:
        registry_policy = ecr_client.get_registry_policy()
        policy_json = json.loads(registry_policy['policyText'])
        for statement in policy_json["Statement"]:
            if(statement["Sid"]==statement_id):
                policy_json["Statement"].remove(statement)
                response = ecr_client.put_registry_policy(json.dumps(policy_json))
                break
        time.sleep(5)
    except:
        None

# add statement with specific Sid in ECR
def add_registry_permission(user_session, statement_id, source_account_id, account_id, region):
    ecr_client = user_session.client("ecr")
    policy=json.dumps({
        "Version":"2012-10-17",
        "Statement":[
            {
                "Sid": statement_id,
                "Principal": {
                    "AWS": "arn:aws:iam::" + source_account_id + ":root"
                },
                "Action": "ecr:*",
                "Resource": [ "arn:aws:ecr:"+region+":"+account_id+":repository/*"],
                "Effect": "Allow"
            }
        ]
    })
    response = ecr_client.put_registry_policy(policyText=policy)
    time.sleep(5)

# create base image having repository:image_tag identity from source account if image not present, update base image if update flag set to true
def clone_base_image(user_session, repository, image_tag, source_account_id, api_endpoint="", update=False):

    if(check_if_image_exists(user_session, repository, image_tag) and update==False):
        result = {"Status": 1, "Success" : "Base image \"" + repository + ":" + image_tag + "\" already exists on this user's account."}
    elif(len(api_endpoint)>0):
        sts_client = user_session.client("sts")

        account_id = sts_client.get_caller_identity()["Account"]
        region = user_session.region_name

        ecr_client = user_session.client('ecr')

        statement_id = source_account_id + "_image_clone_access"

        delete_registry_permission(user_session, statement_id)
        add_registry_permission(user_session, statement_id, source_account_id, account_id, region)

        bodydata = {
            "useraccountnumber": str(account_id),  # change this to first and last name
            "region": region,
            "repository": repository,
            "image_tag": image_tag
        }
        
        # datasets api
        headers_with_authentication = {'Content-Type': 'application/json',
                                       'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"),
                                       'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization',
                                       'Access-Control-Allow-Origin': '*'}

        # modeltoapi lambda function invoked through below url to return new prediction api in response
        response = requests.post(api_endpoint, json=bodydata, headers=headers_with_authentication)
        time.sleep(5)

        # Delete registry policy
        delete_registry_permission(user_session, statement_id)

        #client.delete_registry_policy()
        result = {"Status": 1, "Success" : "New base image loaded to user's account"}

        #print(result)
    else:
        result = {"Status": 0, "Success" : "API endpoint not valid/not provided and image does not exist either."}
    return result


def get_cloudformation_template():
    return """
        {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Creation of Role, Policy, Lambda, Lambda Permission",
            "Resources": {
                "Role": {
                    "Type" : "AWS::IAM::Role",
                    "Properties" : {
                        "AssumeRolePolicyDocument" : $TrustPolicy,
                        "RoleName" : "$RoleName"
                    }
                },
                "Policy": {
                    "DependsOn": "Role",
                    "Type" : "AWS::IAM::Policy",
                    "Properties" : {
                        "PolicyDocument" : $PolicyDocument,
                        "PolicyName" : "$PolicyName",
                        "Roles": ["$RoleName"]
                    }
                },
                "Lambda": {
                    "DependsOn": "Policy",
                    "Type" : "AWS::Lambda::Function",
                    "Properties" : {
                        "Code" : $Code,
                        "Environment" : $Environment,
                        "FunctionName" : "$FunctionName",
                        "MemorySize" : $MemorySize,
                        "PackageType" : "$PackageType",
                        "Role" : {
                            "Fn::GetAtt": [
                                "Role",
                                "Arn"
                            ]
                        },
                        "Timeout" : $Timeout
                    }
                },
                "LambdaInvokePermission1": {
                    "DependsOn": "Lambda",
                    "Type" : "AWS::Lambda::Permission",
                    "Properties" : {
                        "Action" : "$Action1",
                        "FunctionName" : "$FunctionName",
                        "Principal" : "$Principal1",
                        "SourceArn" : "$SourceArn1"
                    }
                }
            }
        }
    """