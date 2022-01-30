import time
import json
import boto3

delay=2

class AWSClient():

    def __init__(self, user_session):

        self.ecr_client = user_session.client('ecr')
        self.s3_client = user_session.client("s3")
        self.iam_client = user_session.client("iam")
        self.sts_client = user_session.client("sts")
        self.lambda_client = user_session.client("lambda")
        self.codebuild_client = user_session.client("codebuild")
        self.apigateway_client = user_session.client("apigateway")
        self.account_id = self.sts_client.get_caller_identity()["Account"]

    def get_repository_details(self, repository_name):
        try:
            response = self.ecr_client.describe_repositories(repositoryNames=[repository_name])
            repository_details = response['repositories']
        except:
            repository_details = []
        return repository_details

    def get_image_details(self, repository_name, image_tag):
        try:
            response = self.ecr_client.describe_images(repositoryName=repository_name, imageIds=[{'imageTag': image_tag}])
            image_details = response['imageDetails']
        except:
            image_details = []
        return image_details

    def get_role_details(self, role_name):
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            role_details = response['Role']
        except:
            role_details = {}
        return role_details

    def get_policy_details(self, policy_arn):
        try:
            response = self.iam_client.get_policy(PolicyArn=policy_arn)
            policy_details = response['Policy']
        except:
            policy_details = {}
        return policy_details

    def detach_policies_from_role(self, role_name):
        response = self.iam_client.list_attached_role_policies(RoleName=role_name)
        policies = response['AttachedPolicies']
        for policy in policies:
            response = self.iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
            time.sleep(delay)

    def delete_iam_role(self, role_name):
        if(len(self.get_role_details(role_name))):
            response = self.detach_policies_from_role(role_name)
            response = self.iam_client.delete_role(RoleName=role_name)
            time.sleep(delay)
        
    def delete_iam_policy(self, policy_name):
        policy_arn = "arn:aws:iam::" + self.account_id + ":policy/" + policy_name
        if(len(self.get_policy_details(policy_arn))):
            response = self.iam_client.delete_policy(PolicyArn=policy_arn)
            time.sleep(delay)

    def create_iam_role(self, role_name, trust_relationship):
        response = self.iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_relationship))
        time.sleep(delay)

    def create_iam_policy(self, policy_name, policy):
        policy_arn = "arn:aws:iam::" + self.account_id + ":policy/" + policy_name
        response = self.iam_client.create_policy(PolicyName=policy_name, PolicyDocument=json.dumps(policy))
        time.sleep(delay)

    def attach_policy_to_role(self, role_name, policy_name):
        policy_arn = "arn:aws:iam::" + self.account_id + ":policy/" + policy_name
        response = self.iam_client.attach_role_policy(RoleName = role_name, PolicyArn = policy_arn)
        time.sleep(delay)

    def create_repository(self, repository_name):
        response = self.ecr_client.create_repository(repositoryName=repository_name)
        time.sleep(delay)

    def delete_repository(self, repository_name):
        pass
        time.sleep(delay)

    def upload_file_to_s3(self, local_file_path, bucket_name, bucket_file_path):
        response = self.s3_client.upload_file(local_file_path, bucket_name, bucket_file_path)
        time.sleep(delay)

    def delete_file_from_s3(self, bucket_name, bucket_file_path):
        response = self.s3_client.delete_object(Bucket=bucket_name, Key=bucket_file_path)
        time.sleep(delay)

    def create_s3_bucket(self, bucket_name, region):
        try:
            response=self.s3_client.head_bucket(Bucket=bucket_name)
        except:
            if(region=="us-east-1"):
                response = self.s3_client.create_bucket(
                    ACL="private",
                    Bucket=bucket_name
                )
            else:
                location = {'LocationConstraint': region}
                response = self.s3_client.create_bucket(
                    ACL="private",
                    Bucket=bucket_name,
                    CreateBucketConfiguration=location
                )
        return response

    ####################################################################################################

    def add_invoke_resource_policy_to_lambda(self, lambda_function, statement_id, source_arn):
        response = self.lambda_client.add_permission(
            FunctionName=lambda_function,
            StatementId=statement_id,
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=source_arn
        )
        return response

    def get_api_resources(self, api_id):
        try:
            response = self.apigateway_client.get_resources(restApiId=api_id)
            resources = response['items']
        except:
            resources = []
        return resources

    def integration_setup(self, api_id, resource_id, uri_str, credentials, integration_response):
        response_1 = self.apigateway_client.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=uri_str,
            credentials=credentials
        )
        response_2 = self.apigateway_client.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            type='MOCK',
            requestTemplates={"application/json": '{"statusCode": 200}'},
            integrationHttpMethod='OPTIONS',
            uri=uri_str,
            credentials=credentials
        )
        response_3 = self.apigateway_client.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            type='MOCK',
            requestTemplates={"application/json": '{"statusCode": 200}'}
        )
        response_4 = self.apigateway_client.put_integration_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseParameters=integration_response,
            responseTemplates={"application/json": '{"statusCode": 200}'}
        )