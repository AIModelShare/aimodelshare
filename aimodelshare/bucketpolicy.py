def _custom_s3_policy(bucketname):
  this_policy = {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Action": "s3:*",
              "Resource": [
                "arn:aws:s3:::"+bucketname,
                "arn:aws:s3:::"+bucketname+"/*"
                      ]
                },{
              "Effect": "Allow",
              "Action": "apigateway:*",
              "Resource":"*"
                },{
              "Effect": "Allow",
              "Action": "lambda:*",
              "Resource":"*"
                },{
              "Effect": "Allow",
              "Action": "sts:GetCallerIdentity",
              "Resource":"*"
                },{
              "Effect": "Allow",
              "Action": ["iam:ListRoles","iam:PutRolePolicy","iam:CreateRole", "iam:ListRolePolicies", "iam:ListAttachedRolePolicies","iam:GetRole","iam:PassRole"],
              "Resource":[ "arn:aws:lambda:*:*:function:*", "arn:aws:iam::*:role/*","arn:aws:apigateway:*:*","arn:aws:execute-api:*:*"]
                }
                 ]

                }

  return this_policy


def _custom_upload_policy(bucket_name, unique_model_id):
  managed_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowUserToSeeBucketListInTheConsole"+str(unique_model_id),
            "Action": [
                "s3:ListAllMyBuckets",
                "s3:GetBucketLocation"
            ],
            "Effect": "Allow",
            "Resource": [
                "arn:aws:s3:::*"
            ]
        },
        {
            "Sid": "AllowRootAndHomeListingOfCompanyBucket"+str(unique_model_id),
            "Action": [
                "s3:ListBucket"
            ],
            "Effect": "Allow",
            "Resource": [
                "arn:aws:s3:::"+bucket_name
            ],
            "Condition": {
                "StringEquals": {
                    "s3:prefix": [
                        "",
                        unique_model_id+"/"
                    ],
                    "s3:delimiter": [
                        "/"
                    ]
                }
            }
        },
        {
            "Sid": "AllowListingOfUserFolder"+str(unique_model_id),
            "Action": [
                "s3:ListBucket"
            ],
            "Effect": "Allow",
            "Resource": [
                "arn:aws:s3:::"+bucket_name
            ],
            "Condition": {
                "StringLike": {
                    "s3:prefix": [
                        unique_model_id+"/*"
                    ]
                }
            }
        },
        {
            "Sid": "AllowAllS3ActionsInUserFolder"+str(unique_model_id),
            "Effect": "Allow",
            "Action": [
                "s3:*"
            ],
            "Resource": [
                "arn:aws:s3:::"+bucket_name+"/"+unique_model_id+"/*"
            ]
        }
    ]
    }
  return managed_policy

__all__ = [
    _custom_s3_policy,
    _custom_upload_policy,
]
