import sys
import math
import datetime
import requests

import jwt
import boto3
import botocore
import json
import logging
import os
import time
import uuid
import redis
from redisearch import Client, TextField, NumericField, IndexDefinition, Query

REDIS_HOST = 'data url removed'

REDIS_PORT = 15647
REDIS_DB = 0

db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username="", password="",decode_responses=True)

# Created a client with a given index name and indexed all fields in original dynamodb website data
client = Client("myIndexfulldatav2",conn=db)

# Fxn that Searches database using apiurl to get modelid for redis hset item updates:
def get_id_fromapiurl(apiurlstring="default"):
  res = client.search(apiurlstring.replace(".","/").split("/")[2])
  if res.total==0:
    modelidstring="model:"+str(db.dbsize()+1)
  else:
    modelidstring=res.docs[0].id  
  return modelidstring
  
# Notes: only allow end user with authenticated user name to change database
def update_modeldata(query_param_eventdict="default"):
  #add logic to extract authenticated user from context after sign in.
  model_idkey=query_param_eventdict['id']
  query_param_eventdict.pop('id','id')
  query_param_eventdict.pop('delete','delete not found in event')
  query_param_eventdict.pop('versionupdateget','versionupdateget not found in event')
  query_param_eventdict.pop('versionupdateput','versionupdateput not found in event')

  modeldataupdate_keys=list(query_param_eventdict.keys())
  modeldataupdate_vals=list(query_param_eventdict.values())

  for i, j in zip(modeldataupdate_keys, modeldataupdate_vals):
    db.hset(model_idkey,key=i, value=j)
  return "update fxn ran"

# now how do you add a new image to image url using similar function with randomly selected images?
def getimage_by_modeltype(modeltype=""):
      randomimagedata={'text': [
          'https://images.unsplash.com/photo-1556075798-4825dfaaf498',
          'https://images.unsplash.com/photo-1451226428352-cf66bf8a0317',
          'https://images.unsplash.com/photo-1572375992501-4b0892d50c69',
          'https://images.unsplash.com/photo-1575886889511-1a2d75aa58b2',
          'https://images.unsplash.com/photo-1550645612-83f5d594b671',
          'https://images.unsplash.com/photo-1573919779902-39c46fa3fbfb',
          'https://images.unsplash.com/photo-1588570208355-f93e85c15de4',
          'https://images.unsplash.com/photo-1576057405245-28ef83d5b50e',
          'https://images.unsplash.com/photo-1568716353609-12ddc5c67f04',
          'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5',
          'https://images.unsplash.com/photo-1534972195531-d756b9bfa9f2',
          'https://images.unsplash.com/photo-1531030874896-fdef6826f2f7',
          'https://images.unsplash.com/photo-1533709752211-118fcaf03312',
          'https://images.unsplash.com/photo-1518818608552-195ed130cdf4',
          'https://images.unsplash.com/photo-1517148815978-75f6acaaf32c',
          'https://images.unsplash.com/photo-1535551951406-a19828b0a76b',
          'https://images.unsplash.com/photo-1584949091882-158ebf933cda',
          'https://images.unsplash.com/photo-1514543250559-83867827ecce',
          'https://images.unsplash.com/photo-1603302576837-37561b2e2302',
          'https://images.unsplash.com/photo-1521185496955-15097b20c5fe'
        ],
        'image': [
          'https://images.unsplash.com/photo-1505739998589-00fc191ce01d',
          'https://images.unsplash.com/photo-1488928741225-2aaf732c96cc',
          'https://images.unsplash.com/photo-1481923387198-050ac1a2896e',
          'https://images.unsplash.com/photo-1490750967868-88aa4486c946',
          'https://images.unsplash.com/photo-1507290439931-a861b5a38200',
          'https://images.unsplash.com/photo-1464820453369-31d2c0b651af',
          'https://images.unsplash.com/photo-1475724017904-b712052c192a',
          'https://images.unsplash.com/photo-1532211387405-12202cb81d7b',
          'https://images.unsplash.com/photo-1573572042111-dcdf086047be',
          'https://images.unsplash.com/photo-1496483648148-47c686dc86a8',
          'https://images.unsplash.com/photo-1499018658500-b21c72d7172b',
          'https://images.unsplash.com/photo-1490349368154-73de9c9bc37c',
          'https://images.unsplash.com/photo-1491929007750-dce8ba76e610',
          'https://images.unsplash.com/photo-1559333421-1ffcd0f39500',
          'https://images.unsplash.com/photo-1597908755005-f3a199798270',
          'https://images.unsplash.com/photo-1576485248988-ea8a3cbf9f36',
          'https://images.unsplash.com/photo-1603734449646-b5c2b4a2336c',
          'https://images.unsplash.com/32/RgJQ82pETlKd0B7QzcJO_5912578701_92397ba76c_b.jpg',
          'https://images.unsplash.com/photo-1585139469968-012793d88874',
          'https://images.unsplash.com/photo-1549204712-94c58d2fe518',
          'https://images.unsplash.com/photo-1601758416326-e7f87ce9f222',
          'https://images.unsplash.com/photo-1594676979216-380a0257305d'
        ],
        'audio': [
          'https://images.unsplash.com/photo-1519874179391-3ebc752241dd',
          'https://images.unsplash.com/photo-1488376986648-2512dfc6f736',
          'https://images.unsplash.com/photo-1485579149621-3123dd979885',
          'https://images.unsplash.com/photo-1476136236990-838240be4859',
          'https://images.unsplash.com/photo-1565695340051-6ae3603dfa4d',
          'https://images.unsplash.com/photo-1588479839125-3a70c078d257',
          'https://images.unsplash.com/photo-1567456171026-a317912368c9',
          'https://images.unsplash.com/photo-1569120120122-02bbcf92462a',
          'https://images.unsplash.com/photo-1591105866700-cb5d708ccd93',
          'https://images.unsplash.com/photo-1532778489370-ffc5d2095091',
          'https://images.unsplash.com/photo-1543574494-27ea333caf56',
          'https://images.unsplash.com/photo-1484704849700-f032a568e944',
          'https://images.unsplash.com/photo-1525022404438-91321710652d',
          'https://images.unsplash.com/photo-1512429234305-12fe5b0b0f07',
          'https://images.unsplash.com/photo-1483689449536-2f4a21d712b2',
          'https://images.unsplash.com/photo-1487369760466-250247fd9a04',
          'https://images.unsplash.com/photo-1603619890744-ed995cb96219',
          'https://images.unsplash.com/photo-1587687747420-b45f676dca73',
          'https://images.unsplash.com/photo-1521899519970-aa0d00905f56',
          'https://images.unsplash.com/photo-1566153428059-c4c8edc63c3b'
        ],
        'video': [
          'https://images.unsplash.com/photo-1485846234645-a62644f84728',
          'https://images.unsplash.com/photo-1532436908675-8b2b1e9ca504',
          'https://images.unsplash.com/photo-1523286680734-c387ced73af5',
          'https://images.unsplash.com/photo-1555246384-9e2f17cafb6a',
          'https://images.unsplash.com/photo-1510282271343-fdc3dea55439',
          'https://images.unsplash.com/photo-1506434304575-afbb92660c28',
          'https://images.unsplash.com/photo-1525482555664-d9ead73ade78',
          'https://images.unsplash.com/photo-1496680272892-9dd620473ae5',
          'https://images.unsplash.com/photo-1554941068-a252680d25d9',
          'https://images.unsplash.com/photo-1606579335925-ea67bcbd09c7',
          'https://images.unsplash.com/photo-1532456164788-984c62717cf8',
          'https://images.unsplash.com/photo-1501281668745-f7f57925c3b4',
          'https://images.unsplash.com/photo-1542404687-47593128ef38',
          'https://images.unsplash.com/photo-1593205303646-f7d0fa3dfc98',
          'https://images.unsplash.com/photo-1593207124518-d17f82d83fe3',
          'https://images.unsplash.com/photo-1531178625044-cc2a0fb353a9',
          'https://images.unsplash.com/photo-1513880010670-b9d4433deca7',
          'https://images.unsplash.com/photo-1553446472-8f625ae729ad',
          'https://images.unsplash.com/photo-1575664463429-bf4d3c296604',
          'https://images.unsplash.com/photo-1530903503232-1d106837148d'
        ],
        'tabular': [
          'https://images.unsplash.com/photo-1558588942-930faae5a389',
          'https://images.unsplash.com/photo-1581092580497-e0d23cbdf1dc',
          'https://images.unsplash.com/photo-1423666523292-b458da343f6a',
          'https://images.unsplash.com/photo-1516383274235-5f42d6c6426d',
          'https://images.unsplash.com/photo-1485796826113-174aa68fd81b',
          'https://images.unsplash.com/photo-1483736762161-1d107f3c78e1',
          'https://images.unsplash.com/photo-1560472354-b33ff0c44a43',
          'https://images.unsplash.com/photo-1526628953301-3e589a6a8b74',
          'https://images.unsplash.com/photo-1569396116180-210c182bedb8',
          'https://images.unsplash.com/photo-1488229297570-58520851e868',
          'https://images.unsplash.com/photo-1551650992-ee4fd47df41f',
          'https://images.unsplash.com/3/doctype-hi-res.jpg',
          'https://images.unsplash.com/photo-1446776653964-20c1d3a81b06',
          'https://images.unsplash.com/photo-1597852074816-d933c7d2b988',
          'https://images.unsplash.com/photo-1523961131990-5ea7c61b2107',
          'https://images.unsplash.com/photo-1529078155058-5d716f45d604',
          'https://images.unsplash.com/photo-1506555191898-a76bacf004ca',
          'https://images.unsplash.com/photo-1585401738582-41c80d8f7b10',
          'https://images.unsplash.com/photo-1502570149819-b2260483d302',
          'https://images.unsplash.com/photo-1504607798333-52a30db54a5d'
        ],
        'timeseries': [
          'https://images.unsplash.com/photo-1501139083538-0139583c060f',
          'https://images.unsplash.com/photo-1468174829941-1d60ae85c487',
          'https://images.unsplash.com/photo-1431499012454-31a9601150c9',
          'https://images.unsplash.com/photo-1462885928573-b5d04c6855de',
          'https://images.unsplash.com/photo-1447015237013-0e80b2786ddc',
          'https://images.unsplash.com/photo-1494481524892-b1bf38423fd1',
          'https://images.unsplash.com/photo-1439754389055-9f0855aa82c2',
          'https://images.unsplash.com/photo-1499377193864-82682aefed04',
          'https://images.unsplash.com/photo-1482775907821-a56ec43344fc',
          'https://images.unsplash.com/photo-1508962914676-134849a727f0',
          'https://images.unsplash.com/photo-1541480601022-2308c0f02487',
          'https://images.unsplash.com/photo-1456574808786-d2ba7a6aa654',
          'https://images.unsplash.com/photo-1506452819137-0422416856b8',
          'https://images.unsplash.com/photo-1499290731724-12e120cfaef3',
          'https://images.unsplash.com/photo-1478947954920-36eb64d7d8aa',
          'https://images.unsplash.com/photo-1494858723852-d3cc2477e12c',
          'https://images.unsplash.com/photo-1495857000853-fe46c8aefc30',
          'https://images.unsplash.com/photo-1505394021165-c7a7a3ba1307',
          'https://images.unsplash.com/photo-1527434284315-fadc3143d9f2',
          'https://images.unsplash.com/photo-1571251455805-f50e70e3cee3'
        ],
        'unknown': [
          'https://images.unsplash.com/photo-1553901767-12be7b9840e2',
          'https://images.unsplash.com/photo-1578083386368-dcb6d17181da',
          'https://images.unsplash.com/photo-1495055154266-57bbdeada43e',
          'https://images.unsplash.com/photo-1485827404703-89b55fcc595e',
          'https://images.unsplash.com/flagged/photo-1558313728-a3ffd746bd23',
          'https://images.unsplash.com/photo-1573496528681-9b0f4fb0c660',
          'https://images.unsplash.com/photo-1532178324009-6b6adeca1741',
          'https://images.unsplash.com/photo-1494869042583-f6c911f04b4c',
          'https://images.unsplash.com/photo-1528030084539-15fe52c20f1b',
          'https://images.unsplash.com/photo-1559153290-13861520d6b7',
          'https://images.unsplash.com/photo-1543941869-11da6518d88f',
          'https://images.unsplash.com/photo-1531747118685-ca8fa6e08806',
          'https://images.unsplash.com/photo-1516192518150-0d8fee5425e3',
          'https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1',
          'https://images.unsplash.com/photo-1561557944-6e7860d1a7eb',
          'https://images.unsplash.com/photo-1451187863213-d1bcbaae3fa3',
          'https://images.unsplash.com/photo-1455165814004-1126a7199f9b',
          'https://images.unsplash.com/photo-1485795959911-ea5ebf41b6ae',
          'https://images.unsplash.com/photo-1589254065878-42c9da997008',
          'https://images.unsplash.com/photo-1554965650-378bcfce5cac'
        ],
        'custom': [
          'https://images.unsplash.com/photo-1553901767-12be7b9840e2',
          'https://images.unsplash.com/photo-1578083386368-dcb6d17181da',
          'https://images.unsplash.com/photo-1495055154266-57bbdeada43e',
          'https://images.unsplash.com/photo-1485827404703-89b55fcc595e',
          'https://images.unsplash.com/flagged/photo-1558313728-a3ffd746bd23',
          'https://images.unsplash.com/photo-1573496528681-9b0f4fb0c660',
          'https://images.unsplash.com/photo-1532178324009-6b6adeca1741',
          'https://images.unsplash.com/photo-1494869042583-f6c911f04b4c',
          'https://images.unsplash.com/photo-1528030084539-15fe52c20f1b',
          'https://images.unsplash.com/photo-1559153290-13861520d6b7',
          'https://images.unsplash.com/photo-1543941869-11da6518d88f',
          'https://images.unsplash.com/photo-1531747118685-ca8fa6e08806',
          'https://images.unsplash.com/photo-1516192518150-0d8fee5425e3',
          'https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1',
          'https://images.unsplash.com/photo-1561557944-6e7860d1a7eb',
          'https://images.unsplash.com/photo-1451187863213-d1bcbaae3fa3',
          'https://images.unsplash.com/photo-1455165814004-1126a7199f9b',
          'https://images.unsplash.com/photo-1485795959911-ea5ebf41b6ae',
          'https://images.unsplash.com/photo-1589254065878-42c9da997008',
          'https://images.unsplash.com/photo-1554965650-378bcfce5cac'
        ],
        'customized': [
          'https://images.unsplash.com/photo-1553901767-12be7b9840e2',
          'https://images.unsplash.com/photo-1578083386368-dcb6d17181da',
          'https://images.unsplash.com/photo-1495055154266-57bbdeada43e',
          'https://images.unsplash.com/photo-1485827404703-89b55fcc595e',
          'https://images.unsplash.com/flagged/photo-1558313728-a3ffd746bd23',
          'https://images.unsplash.com/photo-1573496528681-9b0f4fb0c660',
          'https://images.unsplash.com/photo-1532178324009-6b6adeca1741',
          'https://images.unsplash.com/photo-1494869042583-f6c911f04b4c',
          'https://images.unsplash.com/photo-1528030084539-15fe52c20f1b',
          'https://images.unsplash.com/photo-1559153290-13861520d6b7',
          'https://images.unsplash.com/photo-1543941869-11da6518d88f',
          'https://images.unsplash.com/photo-1531747118685-ca8fa6e08806',
          'https://images.unsplash.com/photo-1516192518150-0d8fee5425e3',
          'https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1',
          'https://images.unsplash.com/photo-1561557944-6e7860d1a7eb',
          'https://images.unsplash.com/photo-1451187863213-d1bcbaae3fa3',
          'https://images.unsplash.com/photo-1455165814004-1126a7199f9b',
          'https://images.unsplash.com/photo-1485795959911-ea5ebf41b6ae',
          'https://images.unsplash.com/photo-1589254065878-42c9da997008',
          'https://images.unsplash.com/photo-1554965650-378bcfce5cac'
        ],
        'neural style transfer': [
          'https://images.unsplash.com/photo-1553901767-12be7b9840e2',
          'https://images.unsplash.com/photo-1578083386368-dcb6d17181da',
          'https://images.unsplash.com/photo-1495055154266-57bbdeada43e',
          'https://images.unsplash.com/photo-1485827404703-89b55fcc595e',
          'https://images.unsplash.com/flagged/photo-1558313728-a3ffd746bd23',
          'https://images.unsplash.com/photo-1573496528681-9b0f4fb0c660',
          'https://images.unsplash.com/photo-1532178324009-6b6adeca1741',
          'https://images.unsplash.com/photo-1494869042583-f6c911f04b4c',
          'https://images.unsplash.com/photo-1528030084539-15fe52c20f1b',
          'https://images.unsplash.com/photo-1559153290-13861520d6b7',
          'https://images.unsplash.com/photo-1543941869-11da6518d88f',
          'https://images.unsplash.com/photo-1531747118685-ca8fa6e08806',
          'https://images.unsplash.com/photo-1516192518150-0d8fee5425e3',
          'https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1',
          'https://images.unsplash.com/photo-1561557944-6e7860d1a7eb',
          'https://images.unsplash.com/photo-1451187863213-d1bcbaae3fa3',
          'https://images.unsplash.com/photo-1455165814004-1126a7199f9b',
          'https://images.unsplash.com/photo-1485795959911-ea5ebf41b6ae',
          'https://images.unsplash.com/photo-1589254065878-42c9da997008',
          'https://images.unsplash.com/photo-1554965650-378bcfce5cac'
        ]
      }


      import random
      imageurl=random.choice(randomimagedata[modeltype])
      return imageurl


def create(event, context):
    print("Lambda function ARN:", context.invoked_function_arn)
    print("CloudWatch log stream name:", context.log_stream_name)
    print("CloudWatch log group name:",  context.log_group_name)
    print("Lambda Request ID:", context.aws_request_id)
    print("Lambda function memory limits in MB:", context.memory_limit_in_mb)
    # Need to use username to ensure that developers do not update other peoples data or receive on get calls
    print("Lambda function authorizer context data:", event['requestContext']['authorizer']['username'])
    user_authenticated=event['requestContext']['authorizer']['username']
    print(event)
    body=event['body']
    import six
    if isinstance(event["body"], six.string_types):
        body = json.loads(event["body"])


    timestamp = int(time.time() * 1000)
    
    modelidstring=get_id_fromapiurl(body["apiurl"])
    if all([body.get("apideveloper","ALL")==user_authenticated]):
        model_developer_name=body.get("apideveloper")
    else: 
        model_developer_name=db.hget(modelidstring,"apideveloper")
        
    if user_authenticated==model_developer_name:
        if body.get('delete','ALL')=='TRUE':
            #TODO: redis search for apiurl.  Return necessary values (such as model id).
            #Use model id to update item values.
            
            
            db.hset(modelidstring,key="delete",value="TRUE")
            db.hset(modelidstring,key="updatedAt",value=-timestamp)

            # Function to divide data into pages
            # Yield successive n-sized 
            # chunks from l. 
            print(db.hgetall(modelidstring))
    
            # put (idempotent)
            resultfinal=[db.hgetall(modelidstring)]

        
        elif body.get('versionupdateget','ALL')=='TRUE':    
            modelidstring=get_id_fromapiurl(body["apiurl"])
    
            result=db.hgetall(modelidstring)
    
            resultfinal=[str(result['version']), result['bucket_name'],result['unique_model_id']]
    
        elif body.get('versionupdateput','ALL')=='TRUE':    
            modelidstring=get_id_fromapiurl(body["apiurl"])
            body['id']=modelidstring
            #TODO: add      'updatedAt': -timestamp to update_modeldata() fxn
            
            update_modeldata(body)
            db.hset(modelidstring,key="updatedAt",value=-timestamp)

            itemtoupdate=db.hgetall(modelidstring)
    
            resultfinal=[itemtoupdate]
        elif body.get('accessrequestcheck','ALL')=='TRUE':
            resultfinal=["Cognito auth check completed"]
        elif body.get('modifyaccess','ALL')=='TRUE':
            operation = body['operation']
            if operation == "Add":
              result = db.hgetall(modelidstring)
              result = str(result['emaillist'])
              item = {
                'id': modelidstring,
                'emaillist': result + ',' + body['email_list']
              }
              result = db.hset(item['id'], mapping=item)
              resultfinal = item['id']
              pass
            elif operation == 'Remove':
              result = db.hgetall(modelidstring)
              result = str(result['emaillist'])
              # Always leave the owner.
              item = {
                'id': modelidstring,
                'emaillist': result.split(',')[0]
              }
              result = db.hset(item['id'], mapping=item)
              resultfinal = item['id']
            elif operation == 'Replace':
              result = db.hgetall(modelidstring)
              result = str(result['emaillist'])
              # Always leave the owner.
              item = {
                'id': modelidstring,
                'emaillist': result.split(',')[0] + ',' + body['email_list']
              }
              result = db.hset(item['id'], mapping=item)
              resultfinal = item['id']
        elif body.get('getaccess','ALL')=='TRUE':
            # modelidstring = get_id_fromapiurl(body["apiurl"])
            result = db.hgetall(modelidstring)
            resultfinal = str(result['emaillist'])
        else:
            item = {
                'id': "model:"+str(db.dbsize()+1),
                'createdAt': timestamp,
                'updatedAt': -timestamp,
                'unique_model_id':body['unique_model_id'],
                'bucket_name':body['bucket_name'],
                'apideveloper':body['apideveloper'],
                'apimodeldescription':body['apimodeldescription'],
                'apimodelevaluation':body['apimodelevaluation'],
                'apimodeltype':body['apimodeltype'],
                'apiurl':body['apiurl'],
                'modelname':body['modelname'],
                'Private':body['Private'],
                'Categorical':body['Categorical'],
                'delete':body['delete'],
                'tags':body['tags'],
                'version':int(body['version']),
                'input_feature_dtypes':str(body.get('input_feature_dtypes','')),
                'input_feature_names':str(body.get('input_feature_names','')),
                'apicalls': int(0),
                'input_shape': str(body.get('input_shape','')),
                'curlcolabhtmllinkcode': "",
                'rcolabhtmllinkcode': "",
                'pythoncolabhtmllinkcode': "",
                'searchitems': str(body['apideveloper'].lower())+'_' + str(body['apimodeldescription'].lower())+'_'+str(body['apimodelevaluation'].lower())+'_'+str(body['apimodeltype'].lower())+'_'+str(body['apiurl'].lower())+'_'+str(body['tags'].lower()),
                'imageUrl':getimage_by_modeltype(modeltype=body['apimodeltype'].lower()),
                'exampledata':str(body.get('exampledata','')),
                'emaillist':str(body.get('email_list',''))
            }
        
            # write the todo to the database
            result=db.hset(item['id'],mapping=item)
            resultfinal=item['id']
            
    else:
            resultfinal="Request denied: Request username does not match model owner name."
    # create a response
    response = {
        "statusCode": 200,
        "headers": {
          "Access-Control-Allow-Origin" : "*",
          "Access-Control-Allow-Credentials": True,
          "Allow" : "GET, OPTIONS, POST",
          "Access-Control-Allow-Methods" : "GET, OPTIONS, POST",
          "Access-Control-Allow-Headers" : "*"
        },
        "body": json.dumps(resultfinal)
    }

    return response