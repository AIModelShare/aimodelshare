import json
from model import handler

def lambda_handler(event, context):

    return handler(event, context)
