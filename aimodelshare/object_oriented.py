# import packages 
import os
import contextlib
import boto3
from aimodelshare.api import get_api_json
import tempfile
import torch
import onnx
from aimodelshare.utils import HiddenPrints
import signal
from aimodelshare.aimsonnx import model_to_onnx, model_to_onnx_timed
from aimodelshare.tools import extract_varnames_fromtrainingdata, _get_extension_from_filepath
import time
import pandas
import requests


# import packages 
import os
import contextlib
import boto3
from aimodelshare.api import get_api_json
import tempfile
import torch
import onnx
from aimodelshare.aws import get_aws_token

class ModelPlayground:
    """
    Parameters:
    ----------
    `model_type` : ``string``
          values - [ 'text' , 'image' , 'tabular' , 'video', 'audio','timeseries' ] 
          type of model data     
    `classification`:    ``bool, default=True``
        True [DEFAULT] if model is of Classification type with categorical target variables
        False if model is of Regression type with continuous target variables
    `private` :   ``bool, default = False``
        True if model and its corresponding data is not public
        False [DEFAULT] if model and its corresponding data is public 
    `email_list`: ``list of string values``
                values - list including all emails of users who have access the private playground.
                list should contain same emails that were used by users to sign up for modelshare.org account.
                [OPTIONAL] set by the playground owner for private playgrounds.  Can also be updated by editing deployed 
                playground page at www.modelshare.org.
    """
    def __init__(self, model_type=None, classification=None, private=None, playground_url=None, email_list=[]):
        # confirm correct args are provided
        if playground_url != None or all([model_type !=None, classification !=None, private!=None]):
            pass
        elif playground_url == None and any([model_type ==None, classification ==None, private==None]):
            return print("Error. To instantiate a ModelPlayground instance, please provide either a playground_url or \n the model_type, classification, and private arguments.")
        
        self.model_type = model_type
        self.categorical = classification 
        self.private = private
        self.playground_url = playground_url
        self.email_list = email_list
        def codestring(self):
            if self.playground_url==None:
              return   "ModelPlayground(model_type="+"'"+str(self.model_type)+"'"+",classification="+str(self.categorical)+",private="+str(self.private)+",playground_url="+str(self.playground_url)+",email_list="+str(self.email_list)+")"
            else:
              return   "ModelPlayground(model_type="+"'"+str(self.model_type)+"'"+",classification="+str(self.categorical)+",private="+str(self.private)+",playground_url="+"'"+str(self.playground_url)+"'"+",email_list="+str(self.email_list)+")"
        self.class_string=codestring(self)
    
    
    def __str__(self):
        return f"ModelPlayground(self.model_type,self.categorical,self.private = private,self.playground_url,self.email_list)"



    def activate(self, model_filepath=None, preprocessor_filepath=None, y_train=None, example_data=None, 
        custom_libraries = "FALSE", image="", reproducibility_env_filepath=None, memory=None, timeout=None, 
        onnx_timeout=60, pyspark_support=False, model_input=None): 

        """
        Launches a live model playground to the www.modelshare.org website. The playground can optionally include a live prediction REST API for deploying ML models using model parameters and user credentials, provided by the user.
        Inputs : 7
        Output : model launched to an API
                detailed API info printed out
        Parameters: 
        ----------
        `model_filepath` :  ``string`` ends with '.onnx'
              value - Absolute path to model file 
              .onnx is the only accepted model file extension
              "example_model.onnx" filename for file in directory.
              "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
              if no value is set the playground will be launched with only a placeholder prediction API.
        `preprocessor_filepath`:  ``string``
            value - absolute path to preprocessor file 
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated using export_preprocessor function from the AI Modelshare library
            if no value is set the playground will be launched with only a placeholder prediction API.
        `y_train` : training labels for classification models.
            expects pandas dataframe of one hot encoded y train data
            if no value is set ... #TODO 
        `example_data`: ``Example of X data that will be shown on the online Playground page. 
            if no example data is submitted, certain functionalities may be limited, including the deployment of live prediction APIs.
            Example data can be updated at a later stage, using the update_example_data() method.``
        `custom_libraries`: ``string``
            "TRUE" if user wants to load custom Python libraries to their prediction runtime
            "FALSE" if user wishes to use AI Model Share base libraries including latest versions of most common ML libs.
        `reproducibility_env_filepath`: ``TODO``
        `memory`: ``TODO``
        `timeout`: ``TODO``
        `onnx_timeout`: ``int``
            Time in seconds after which ONNX conversion should be interrupted.
            Set to False if you want to force ONNX conversion.
        `pyspark_support`: ``TODO``
        `model_input`: ``array_like``
            Required only when framework="pytorch" 
            One example of X training data in correct format.
         
        Returns:
        --------
        print_api_info : prints statements with generated model playground page and live prediction API details
                        also prints steps to update the model submissions by the user/team 
        """


        # test whether playground is already active
        if self.playground_url: 
            print(self.playground_url)
            def ask_user():
                print("Playground is already active. Would you like to overwrite?")
                response = ''
                while response not in {"yes", "no"}:
                    response = input("Please enter yes or no: ").lower()
                return response != "yes"

            r = ask_user()

            if r: 
                return

        # convert model to onnx 
        if onnx_timeout == False:
            force_onnx=True
        else:
            force_onnx=False
        model_filepath = model_to_onnx_timed(model_filepath, timeout = onnx_timeout, 
            force_onnx=force_onnx, model_input=model_input)

        # keep track of submitted artifacts
        if isinstance(y_train, pandas.Series): 
            y_train_bool = True
        else: 
            y_train_bool = bool(y_train)

        if isinstance(example_data, (pandas.Series, pandas.DataFrame)): 
            example_data_bool = True
        else: 
            example_data_bool = bool(y_train)


        track_artifacts = {"model_filepath": bool(model_filepath),
                            "preprocessor_filepath": bool(preprocessor_filepath), 
                            "y_train": y_train_bool,
                            "example_data": example_data_bool,
                            "custom_libraries": bool(custom_libraries),
                            "image": bool(image),
                            "reproducibility_env_filepath": bool(reproducibility_env_filepath),
                            "memory": bool(memory),
                            "timeout": bool(timeout),
                            "pyspark_support": bool(pyspark_support)
                            }

        import pkg_resources

        # insert placeholders into empty arguments
        if model_filepath == None:
            model_filepath = pkg_resources.resource_filename(__name__, "placeholders/model.onnx")

        if preprocessor_filepath == None:
            preprocessor_filepath = pkg_resources.resource_filename(__name__, "placeholders/preprocessor.zip")

        if y_train_bool == False:
            y_train = []

        if example_data_bool == False and self.model_type=="tabular":
            example_data = pandas.DataFrame()

        import json, tempfile
        tfile = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(track_artifacts, tfile)
        tfile.flush()

        input_dict = {"requirements": "",
                      "model_name": "Default Model Playground",
                      "model_description": "",
                      "tags": ""}

        from aimodelshare.generatemodelapi import model_to_api
        self.playground_url = model_to_api(model_filepath=model_filepath, 
                                      model_type = self.model_type, 
                                      private = self.private, 
                                      categorical = self.categorical,
                                      y_train = y_train, 
                                      preprocessor_filepath = preprocessor_filepath, 
                                      example_data = example_data,
                                      custom_libraries = custom_libraries,
                                      image=image,
                                      reproducibility_env_filepath = reproducibility_env_filepath,
                                      memory=memory,
                                      timeout=timeout,
                                      email_list=self.email_list,
                                      pyspark_support=pyspark_support,
                                      input_dict=input_dict, 
                                      print_output=False)
        #remove extra quotes
        self.playground_url = self.playground_url[1:-1]

        # upload track artifacts
        from aimodelshare.aws import get_s3_iam_client
        s3, iam, region = get_s3_iam_client(os.environ.get("AWS_ACCESS_KEY_ID_AIMS"), os.environ.get("AWS_SECRET_ACCESS_KEY_AIMS"), os.environ.get("AWS_REGION_AIMS"))


        unique_model_id = self.playground_url.split(".")[0].split("//")[-1]

        s3["client"].upload_file(tfile.name, os.environ.get("BUCKET_NAME"), unique_model_id + "/track_artifacts.json") 


    def deploy(self, model_filepath, preprocessor_filepath, y_train, example_data=None, custom_libraries = "FALSE", 
        image="", reproducibility_env_filepath=None, memory=None, timeout=None, onnx_timeout=60, pyspark_support=False,
        model_input=None, input_dict=None):

        """
        Launches a live prediction REST API for deploying ML models using model parameters and user credentials, provided by the user
        Inputs : 7
        Output : model launched to an API
                detailed API info printed out
        Parameters: 
        ----------
        `model_filepath` :  ``string`` ends with '.onnx'
              value - Absolute path to model file 
              [REQUIRED] to be set by the user
              .onnx is the only accepted model file extension
              "example_model.onnx" filename for file in directory.
              "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `preprocessor_filepath`:  ``string``
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated using export_preprocessor function from the AI Modelshare library  
        `y_train` : training labels for classification models.
              [REQUIRED] for classification type models
              expects pandas dataframe of one hot encoded y train data
        `example_data`: ``Example of X data that will be shown on the online Playground page. 
            if no example data is submitted, certain functionalities may be limited, including the deployment of live prediction APIs.
            Example data can be updated at a later stage, using the update_example_data() method.``
        `custom_libraries`:   ``string``
            "TRUE" if user wants to load custom Python libraries to their prediction runtime
            "FALSE" if user wishes to use AI Model Share base libraries including latest versions of most common ML libs.
        `image`: ``TODO``
        `reproducibility_env_filepath`: ``TODO``
        `memory`: ``TODO``
        `timeout`: ``TODO``
        `onnx_timeout`: ``int``
            Time in seconds after which ONNX conversion should be interrupted.
            Set to False if you want to force ONNX conversion.
        `pyspark_support`: ``TODO``
        `model_input`: ``array_like``
            Required only when framework="pytorch" 
            One example of X training data in correct format.
        `input_dict`:   ``dictionary``
             Use to bypass text input boxes Example: {"model_name": "My Model Playground",
                      "model_description": "My Model Description",
                      "tags": "model, classification, awesome"}
         
        Returns:
        --------
        print_api_info : prints statements with generated live prediction API details
                        also prints steps to update the model submissions by the user/team
        """

        # check whether playground url exists
        if self.playground_url: 
            print(self.playground_url)
            print("Trying to deploy to active playground. Would you like to overwrite prediction API?")
            response = ''
            while response not in {"yes", "no"}:
                response = input("Please enter yes or no: ").lower()

            if response == "no":

                print("Please instantiate a new playground and try again.")
                return
        # model deployment files (plus ytrain object)

        # convert model to onnx
        if onnx_timeout == False:
            force_onnx=True
        else:
            force_onnx=False
        model_filepath = model_to_onnx_timed(model_filepath, timeout = onnx_timeout, 
            force_onnx=force_onnx, model_input=model_input)

        if "model_share"==os.environ.get("cloud_location"):
            print("Creating your Model Playground...\nEst. completion: ~1 minute\n")
            def upload_playground_zipfile(model_filepath=None, preprocessor_filepath=None, y_train=None, example_data=None):
              """
              minimally requires model_filepath, preprocessor_filepath 
              """
              zipfilelist=[model_filepath,preprocessor_filepath]

              import json
              import os
              import requests
              import pandas as pd
              if any([isinstance(example_data, pd.DataFrame),example_data==None]):
                  pass
              else:
                  zipfilelist.append(example_data)

              #need to save dict pkl file with arg name and filepaths to add to zipfile



              apiurl="https://1l2z4k1gce.execute-api.us-east-2.amazonaws.com/prod/m"

              apiurl_eval=apiurl[:-1]+"eval"

              headers = { 'Content-Type':'application/json', 'authorizationToken': json.dumps({"token":os.environ.get("AWS_TOKEN"),"eval":"TEST"}), } 
              post_dict = {"return_zip": "True"}
              zipfile = requests.post(apiurl_eval,headers=headers,data=json.dumps(post_dict)) 

              zipfileputlistofdicts=json.loads(zipfile.text)['put']

              zipfilename=list(zipfileputlistofdicts.keys())[0]

              from zipfile import ZipFile
              import os
              from os.path import basename
              import tempfile

              wkingdir=os.getcwd()

              tempdir=tempfile.gettempdir() 

              zipObj = ZipFile(tempdir+"/"+zipfilename, 'w')
              # Add multiple files to the zip
              for i in zipfilelist:
                zipObj.write(i)

              # add object to pkl file pathway here. (saving y label data)
              import pickle

              if y_train==None:
                pass
              else:
                with open(tempdir+"/"+'ytrain.pkl', 'wb') as f:
                  pickle.dump(y_train, f)

                os.chdir(tempdir)
                zipObj.write('ytrain.pkl')

              if isinstance(example_data, pd.DataFrame):
                with open(tempdir+"/"+'exampledata.pkl', 'wb') as f:
                  pickle.dump(example_data, f)

                os.chdir(tempdir)
                zipObj.write('exampledata.pkl')
              else:
                pass


              # close the Zip File
              os.chdir(wkingdir)

              zipObj.close()



              import ast

              finalzipdict=ast.literal_eval(zipfileputlistofdicts[zipfilename])

              url=finalzipdict['url']
              fields=finalzipdict['fields']

              #### save files from model deploy to zipfile in tempdir before loading to s3



              ### Load zipfile to s3
              with open(tempdir+"/"+zipfilename, 'rb') as f:
                files = {'file': (tempdir+"/"+zipfilename, f)}
                http_response = requests.post(url, data=fields, files=files)
              return zipfilename
            deployzipfilename=upload_playground_zipfile(model_filepath, preprocessor_filepath, y_train, example_data)   
            #if aws arg = false, do this, otherwise do aws code
            #create deploy code_string
            def nonecheck(objinput=""):
                if objinput==None:
                  objinput="None"
                else:
                  objinput="'/tmp/"+objinput+"'"
                return objinput

            deploystring=self.class_string.replace(",aws=False","")+"."+"deploy('/tmp/"+model_filepath+"','/tmp/"+preprocessor_filepath+"',"+'y_train'+","+nonecheck(example_data)+",input_dict="+str(input_dict)+')'
            import base64
            import requests
            import json

            api_url = "https://7yo5bckp5bz6l657hl52kao66u0jdlal.lambda-url.us-east-2.on.aws/"

            data = json.dumps({"code": """from aimodelshare import ModelPlayground;myplayground="""+deploystring, "zipfilename": deployzipfilename,"username":os.environ.get("username"), "password":os.environ.get("password"),"token":os.environ.get("JWT_AUTHORIZATION_TOKEN"),"s3keyid":"diays4ugz5"})

            headers = {"Content-Type": "application/json"}

            response = requests.request("POST", api_url, headers = headers, data=data)
            # Print response
            result=json.loads(response.text)

            modelplaygroundurlid=json.loads(result['body'])[-7].replace("Playground Url: ","").strip()

            print(json.loads(result['body'])[-8]+"\n")
            print("View live playground now at:\n"+json.loads(result['body'])[-1])
            
            print("\nConnect to your playground in Python:\n")
            print("myplayground=ModelPlayground(playground_url="+json.loads(result['body'])[-7].replace("Playground Url: ","").strip()+")")
            self.playground_url=modelplaygroundurlid

        else:    
        
            #aws pathway begins here
            from aimodelshare.generatemodelapi import model_to_api
            self.playground_url = model_to_api(model_filepath=model_filepath, 
                                          model_type = self.model_type, 
                                          private = self.private, 
                                          categorical = self.categorical,
                                          y_train = y_train, 
                                          preprocessor_filepath = preprocessor_filepath, 
                                          example_data = example_data,
                                          custom_libraries = custom_libraries,
                                          image=image,
                                          reproducibility_env_filepath = reproducibility_env_filepath,
                                          memory=memory,
                                          timeout=timeout,
                                          email_list=self.email_list,
                                          pyspark_support=pyspark_support,
                                          input_dict=input_dict, 
                                          print_output=False)
            #remove extra quotes
            self.playground_url = self.playground_url[1:-1]

    def get_apikey(self):
        import os
        import requests
        import json
        if all(["username" in os.environ, 
               "password" in os.environ]):
            pass
        else:
            return print("'get_apikey()' unsuccessful. Please provide credentials with set_credentials().")

        post_dict = {"return_apikey":"True"}

        headers = { 'Content-Type':'application/json', 'authorizationToken': os.environ.get("AWS_TOKEN"),} 

        apiurl_eval=self.playground_url[:-1]+"eval"

        api_json = requests.post(apiurl_eval,headers=headers,data=json.dumps(post_dict)) 

        return json.loads(api_json.text)['apikey']
    
    def create_competition(self, data_directory, y_test, eval_metric_filepath=None, email_list = [], public=True, public_private_split=0.5, input_dict=None):
        """
        Creates a model competition for a deployed prediction REST API
        Inputs : 4
        Output : Create ML model competition and allow authorized users to submit models to resulting leaderboard/competition
        
        Parameters:
        -----------
        `y_test` :  ``list`` of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
            [REQUIRED] to generate eval metrics in competition leaderboard
                                
        `data_directory` : folder storing training data and test data (excluding Y test data)
        `eval_metric_filepath`: [OPTIONAL] file path of zip file with custon evaluation functions
        `email_list`: [OPTIONAL] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
        `public`: [REQUIRED] True/false. Defaults to False.  If True, competition is public and ANY AIMODELSHARE USER CAN SUBMIT MODELS.  USE WITH CAUTION b/c one model and 
            one preprocessor file will be be saved to your AWS S3 folder for each model submission.
        `public_private_split`: [REQUIRED] Float between 0 and 1. Defaults to 0.5. Porportion of test data that is allocated to private hold-out set.
        
        Returns:
        -----------
        finalmessage : Information such as how to submit models to competition
        
        """

        # catch email list error
        if public==False and email_list == []:
            raise ValueError("Please submit valid email list for private competition.")
        if "model_share"==os.environ.get("cloud_location"):
            print("Creating your Model Playground...\nEst. completion: ~1 minute\n")
            if input_dict==None:
                print("\n--INPUT COMPETITION DETAILS--\n")

                aishare_competitionname = input("Enter competition name:")
                aishare_competitiondescription = input("Enter competition description:")

                print("\n--INPUT DATA DETAILS--\n")
                print("Note: (optional) Save an optional LICENSE.txt file in your competition data directory to make users aware of any restrictions on data sharing/usage.\n")

                aishare_datadescription = input(
                    "Enter data description (i.e.- filenames denoting training and test data, file types, and any subfolders where files are stored):")

                aishare_datalicense = input(
                    "Enter optional data license descriptive name (e.g.- 'MIT, Apache 2.0, CC0, Other, etc.'):")
 
                input_dict={"competition_name":aishare_competitionname,"competition_description":aishare_competitiondescription,"data_description":aishare_datadescription,"data_license":aishare_datalicense}
            else:
               pass
        
            
            # model competition files
            def upload_comp_exp_zipfile(data_directory, y_test=None, eval_metric_filepath=None, email_list=[]):
                """
                minimally requires model_filepath, preprocessor_filepath 
                """
                zipfilelist=[data_directory]

                import json
                import os
                import requests
                import pandas as pd
                if eval_metric_filepath==None:
                    pass
                else:
                    zipfilelist.append(eval_metric_filepath)

                #need to save dict pkl file with arg name and filepaths to add to zipfile




                apiurl="https://1l2z4k1gce.execute-api.us-east-2.amazonaws.com/prod/m"

                apiurl_eval=apiurl[:-1]+"eval"

                headers = { 'Content-Type':'application/json', 'authorizationToken': json.dumps({"token":os.environ.get("AWS_TOKEN"),"eval":"TEST"}), } 
                post_dict = {"return_zip": "True"}
                zipfile = requests.post(apiurl_eval,headers=headers,data=json.dumps(post_dict)) 

                zipfileputlistofdicts=json.loads(zipfile.text)['put']

                zipfilename=list(zipfileputlistofdicts.keys())[0]

                from zipfile import ZipFile
                import os
                from os.path import basename
                import tempfile

                wkingdir=os.getcwd()

                tempdir=tempfile.gettempdir() 

                zipObj = ZipFile(tempdir+"/"+zipfilename, 'w')
                # Add multiple files to the zip


                for i in zipfilelist:
                  for dirname, subdirs, files in os.walk(i):
                    zipObj.write(dirname)
                    for filename in files:
                        zipObj.write(os.path.join(dirname, filename))
                  #zipObj.write(i)

                # add object to pkl file pathway here. (saving y label data)
                import pickle

                if y_test==None:
                  pass
                else:
                  with open(tempdir+"/"+'ytest.pkl', 'wb') as f:
                    pickle.dump(y_test, f)

                  os.chdir(tempdir)
                  zipObj.write('ytest.pkl')

                if isinstance(email_list, list):
                  with open(tempdir+"/"+'emaillist.pkl', 'wb') as f:
                    pickle.dump(email_list, f)

                  os.chdir(tempdir)
                  zipObj.write('emaillist.pkl')
                else:
                  pass


                # close the Zip File
                os.chdir(wkingdir)

                zipObj.close()



                import ast

                finalzipdict=ast.literal_eval(zipfileputlistofdicts[zipfilename])

                url=finalzipdict['url']
                fields=finalzipdict['fields']

                #### save files from model deploy to zipfile in tempdir before loading to s3



                ### Load zipfile to s3
                with open(tempdir+"/"+zipfilename, 'rb') as f:
                  files = {'file': (tempdir+"/"+zipfilename, f)}
                  http_response = requests.post(url, data=fields, files=files)
                return zipfilename                                                 
            compzipfilename=upload_comp_exp_zipfile(data_directory, y_test, eval_metric_filepath, email_list)
            #if aws arg = false, do this, otherwise do aws code
            #create deploy code_string
            def nonecheck(objinput=""):
                if objinput==None:
                  objinput="None"
                else:
                  objinput="'/tmp/"+objinput+"'"
                return objinput
            playgroundurlcode="playground_url="+self.playground_url
            compstring=self.class_string.replace(",aws=False","").replace("playground_url=None",playgroundurlcode)+"."+"create_competition('/tmp/"+data_directory+"',"+'y_test'+","+nonecheck(eval_metric_filepath)+","+'email_list'+",input_dict="+str(input_dict)+')'
            print(compstring)
            import base64
            import requests
            import json

            api_url = "https://7yo5bckp5bz6l657hl52kao66u0jdlal.lambda-url.us-east-2.on.aws/"

            data = json.dumps({"code": """from aimodelshare import ModelPlayground;myplayground="""+compstring, "zipfilename": compzipfilename,"username":os.environ.get("username"), "password":os.environ.get("password"),"token":os.environ.get("JWT_AUTHORIZATION_TOKEN"),"s3keyid":"diays4ugz5"})

            headers = {"Content-Type": "application/json"}

            response = requests.request("POST", api_url, headers = headers, data=data)
            print(response.text)

            return(response.text) 
        else:    

                from aimodelshare.generatemodelapi import create_competition

                competition = create_competition(self.playground_url, 
                                            data_directory, 
                                            y_test, 
                                            eval_metric_filepath,
                                            email_list, 
                                            public,
                                            public_private_split, input_dict=input_dict)
                return competition
        
    def create_experiment(self, data_directory, y_test, eval_metric_filepath=None, email_list = [], public=False, public_private_split=0):
        """
        Creates an experiment for a deployed prediction REST API
        Inputs : 4
        Output : Create ML model experiment and allows authorized users to submit models to resulting experiment tracking leaderboard
        
        Parameters:
        -----------
        `y_test` :  ``list`` of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
            [REQUIRED] to generate eval metrics in experiment leaderboard
                                
        `data_directory` : folder storing training data and test data (excluding Y test data)
        `eval_metric_filepath`: [OPTIONAL] file path of zip file with custon evaluation functions
        `email_list`: [OPTIONAL] list of comma separated emails for users who are allowed to submit models to experiment leaderboard.  Emails should be strings in a list.
        `public`: [REQUIRED] True/false. Defaults to False.  If True, experiment is public and ANY AIMODELSHARE USER CAN SUBMIT MODELS.  USE WITH CAUTION b/c one model and 
            one preprocessor file will be be saved to your AWS S3 folder for each model submission.
        `public_private_split`: [REQUIRED] Float between 0 and 1. Defaults to 0. Porportion of test data that is allocated to private hold-out set.
        
        
        Returns:
        -----------
        finalmessage : Information such as how to submit models to experiment
        
        """

        # catch email list error
        if public==False and email_list == []:
            raise ValueError("Please submit valid email list for private experiment.")


        from aimodelshare.generatemodelapi import create_experiment

        experiment = create_experiment(self.playground_url, 
                                    data_directory, 
                                    y_test, 
                                    eval_metric_filepath,
                                    email_list, 
                                    public,
                                    public_private_split)
        return experiment

    def quick_submit(self, model_filepath, preprocessor_filepath, prediction_submission, y_test, 
        data_directory=None, eval_metric_filepath=None, email_list = [], public=True, public_private_split=0.5,
        model_input=None, timeout=None, onnx_timeout=60, example_data=None):
        """
        Submits model/preprocessor to machine learning experiment leaderboard and model architecture database using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated 
        
        Parameters:
        -----------
        `model_filepath`:  ``string`` ends with '.onnx'
            value - Absolute path to model file [REQUIRED] to be set by the user
            .onnx is the only accepted model file extension
            "example_model.onnx" filename for file in directory.
            "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `preprocessor_filepath`:   ``string``, default=None
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
        `prediction_submission`: [REQUIRED] list of predictions from X test data that will be used to evaluate model prediction error against y test data.
            Use mycompetition.inspect_y_test() to view example of list expected by competition.
        `y_test` :  ``list`` of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
            [REQUIRED] to generate eval metrics in experiment leaderboard   
        `eval_metric_filepath`: [OPTIONAL] file path of zip file with custon evaluation functions
        `data_directory` : folder storing training data and test data (excluding Y test data)                   
        `email_list`: [OPTIONAL] list of comma separated emails for users who are allowed to submit models to experiment leaderboard.  Emails should be strings in a list.
        `public`: [REQUIRED] True/false. Defaults to False.  If True, experiment is public and ANY AIMODELSHARE USER CAN SUBMIT MODELS.  USE WITH CAUTION b/c one model and 
            one preprocessor file will be be saved to your AWS S3 folder for each model submission.
        `public_private_split`: [REQUIRED] Float between 0 and 1. Defaults to 0. Porportion of test data that is allocated to private hold-out set.
        `model_input`: ``array_like``
            Required only when framework="pytorch" 
            One example of X training data in correct format.
        `timeout`: ``TODO``
        `onnx_timeout`: ``int``
            Time in seconds after which ONNX conversion should be interrupted.
            Set to False if you want to force ONNX conversion.
        `example_data`: ``Example of X data that will be shown on the online Playground page. 
            if no example data is submitted, certain functionalities may be limited, including the deployment of live prediction APIs.
            Example data can be updated at a later stage, using the update_example_data() method.``

        Returns:
        -------
        response:   Model version if the model is submitted sucessfully
                    error  if there is any error while submitting models
        """

        # catch email list error
        if public==False and email_list == []:
            raise ValueError("Please submit valid email list for private competition/experiment.")

        # catch missing model_input for pytorch 
        if isinstance(model_filepath, torch.nn.Module) and model_input==None:
            raise ValueError("Please submit valid model_input for pytorch model.")

        # convert model to onnx 
        if onnx_timeout == False:
            force_onnx=True
        else:
            force_onnx=False
        model_filepath = model_to_onnx_timed(model_filepath, timeout = onnx_timeout, 
            force_onnx=force_onnx, model_input=model_input)


        # test whether playground is active, activate if that is not the case
        if not self.playground_url:
            #self.activate(example_data=example_data)
            self.activate(model_filepath, preprocessor_filepath, example_data=example_data,
                onnx_timeout=onnx_timeout, y_train=y_train, custom_libraries=custom_libraries, 
                image=image, reproducibility_env_filepath=reproducibility_env_filepath, 
                memory=memory, pyspark_support=pyspark_support, timeout=timeout)
            print()

        # if playground is active, ask whether user wants to overwrite 
        else:

            print("The Model Playground is already active. Do you want to overwrite existing competitions and experiments?")
            response = ''
            while response not in {"yes", "no"}:
                response = input("Please enter yes or no: ").lower()

            if response == "no":

                print("Please instantiate a new playground and try again.")
                return

        # get model id from playground url
        unique_model_id = self.playground_url.split(".")[0].split("//")[-1]

        comp_input_dict = {"competition_name": "Default Competition "+ unique_model_id,
                            "competition_description": "",
                            "data_description": "",
                            "data_license": ""}


        with HiddenPrints():

            from aimodelshare.generatemodelapi import create_competition
            create_competition(apiurl=self.playground_url,
                                    data_directory=data_directory, 
                                    y_test = y_test,
                                    eval_metric_filepath = eval_metric_filepath,
                                    email_list=email_list,
                                    public=public,
                                    public_private_split=public_private_split,
                                    input_dict=comp_input_dict,
                                    print_output=False)

            competition = Competition(self.playground_url)

            if len(prediction_submission):           
                version_comp = competition.submit_model(model_filepath = model_filepath,
                                        preprocessor_filepath=preprocessor_filepath,
                                        prediction_submission=prediction_submission,
                                        input_dict={"tags":"", "description":""},
                                        print_output=False)
        if len(prediction_submission):           
            print(f"Your model has been submitted to competition as model version {version_comp}.")
           

        with HiddenPrints():

            exp_input_dict = {"experiment_name": "Default Experiment "+unique_model_id,
                                "experiment_description": "",
                                "data_description": "",
                                "data_license": ""}

            from aimodelshare.generatemodelapi import create_experiment
            create_experiment(apiurl=self.playground_url,
                                    data_directory=data_directory, 
                                    y_test = y_test,
                                    eval_metric_filepath = eval_metric_filepath,
                                    email_list=email_list,
                                    public=public,
                                    public_private_split=public_private_split,
                                    input_dict=exp_input_dict,
                                    print_output=False)

            experiment = Experiment(self.playground_url)
            
            if len(prediction_submission):           
                version_exp = experiment.submit_model(model_filepath = model_filepath,
                                     preprocessor_filepath=preprocessor_filepath,
                                     prediction_submission=prediction_submission,
                                     input_dict={"tags":"", "description":""},
                                     print_output=False)

        if len(prediction_submission):           
            print(f"Your model has been submitted to experiment as model version {version_exp}.")

        print("Check out your Model Playground page for more.")

        try:    
            temp.close()
        except:
            pass

    def submit_model(self, model_filepath, preprocessor_filepath, prediction_submission, submit_to="all",
        sample_data=None, reproducibility_env_filepath=None, custom_metadata=None, input_dict=None, onnx_timeout=60, model_input=None):
        """
        Submits model/preprocessor to machine learning competition using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated
        
        Parameters:
        -----------
        `model_filepath`:  ``string`` ends with '.onnx'
            value - Absolute path to model file [REQUIRED] to be set by the user
            .onnx is the only accepted model file extension
            "example_model.onnx" filename for file in directory.
            "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `prediction_submission`:   one hot encoded y_pred
            value - predictions for test data
            [REQUIRED] for evaluation metrics of the submitted model
        `preprocessor_filepath`:   ``string``, default=None
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
 
        Returns:
        --------
        response:   Model version if the model is submitted sucessfully
        """

        from aimodelshare.model import submit_model

        # convert model to onnx
        if onnx_timeout == False:
            force_onnx=True
        else:
            force_onnx=False
        model_filepath = model_to_onnx_timed(model_filepath, timeout = onnx_timeout, 
            force_onnx=force_onnx, model_input=model_input)

        # create input dict
        input_dict = {}
        input_dict["tags"] = input("Insert search tags to help users find your model (optional): ")
        input_dict["description"] = input("Provide any useful notes about your model (optional): ")

 
        if submit_to == "competition" or submit_to == "all": 

            with HiddenPrints():
                competition = Competition(self.playground_url)

                version_comp = competition.submit_model(model_filepath = model_filepath, 
                                      prediction_submission = prediction_submission, 
                                      preprocessor_filepath = preprocessor_filepath,
                                      reproducibility_env_filepath = reproducibility_env_filepath,
                                      custom_metadata = custom_metadata, 
                                      input_dict=input_dict,
                                      print_output=False)


            print(f"Your model has been submitted to competition as model version {version_comp}.")

        if submit_to == "experiment" or submit_to == "all": 

            with HiddenPrints():
                experiment = Experiment(self.playground_url)

                version_exp = experiment.submit_model(model_filepath = model_filepath, 
                                      prediction_submission = prediction_submission, 
                                      preprocessor_filepath = preprocessor_filepath,
                                      reproducibility_env_filepath = reproducibility_env_filepath,
                                      custom_metadata = custom_metadata, 
                                      input_dict=input_dict,
                                      print_output=False)

            print(f"Your model has been submitted to experiment as model version {version_exp}.")
            print(f"Visit your Model Playground Page for more.")


        return 

    
    def update_runtime_model(self, model_version=None, submission_type="competition"):
        """
        Updates the prediction API behind the Model Playground with a new model from the leaderboard and verifies Model Playground performance metrics.
        Parameters:
        -----------
        `model_version`: ``int``
            model version number from competition leaderboard
        Returns:
        --------
        response:   success message when the model and preprocessor are updated successfully
        
        """
        from aimodelshare.model import update_runtime_model as update
        update = update(apiurl = self.playground_url, model_version = model_version, submission_type=submission_type)
        return update
        
    def instantiate_model(self, version=None, trained=False, reproduce=False, submission_type="competition"): 
        """
        Import a model previously submitted to a leaderboard to use in your session
        Parameters:
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
        `trained`: ``bool, default=False``
            if True, a trained model is instantiated, if False, the untrained model is instantiated
        Returns:
        --------
        model: model chosen from leaderboard
        """
        raise AssertionError("You are trying to Instantiate model with ModelPlayground Object, Please use the competition object to Instantiate model")
        from aimodelshare.aimsonnx import instantiate_model
        model = instantiate_model(apiurl=self.playground_url, trained=trained, version=version, reproduce=reproduce, submission_type=submission_type)
        return model
    
    def replicate_model(self,version=None, submission_type="competition"): 
        """
        Instantiate an untrained model with reproducibility environment setup. 
        
        Parameters: 
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
          
        Returns:
        --------
        model:  model chosen from leaderboard     
        """
        
        model = self.instantiate_model(version=version,trained = False,reproduce=True,submission_type=submission_type)
        
        return model
    
   
    def delete_deployment(self, playground_url=None):
        """
        Delete all components of a Model Playground, including: AWS s3 bucket & contents,
        attached competitions, prediction REST API, and interactive Model Playground web dashboard.
        Parameters:
        -----------
        `playground_url`: ``string`` of API URL the user wishes to delete
        WARNING: User must supply high-level credentials in order to delete an API.
        Returns:
        --------
        Success message when deployment is deleted.
        """
        from aimodelshare.api import delete_deployment
        if playground_url == None:
            playground_url = self.playground_url
        deletion = delete_deployment(apiurl = playground_url)
        return deletion

    def import_reproducibility_env(self):
        from aimodelshare.reproducibility import import_reproducibility_env_from_model
        import_reproducibility_env_from_model(apiurl=self.playground_url)

    def update_access_list(self, email_list=[], update_type="Replace_list"):
        """
        Updates list of authenticated participants who can submit new models to a competition.
        Parameters:
        -----------
        `apiurl`: string
                URL of deployed prediction API 
          
        `email_list`: [REQUIRED] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
        `update_type`:[REQUIRED] options, ``string``: 'Add', 'Remove', 'Replace_list','Get. Add appends user emails to original list, Remove deletes users from list, 
                  'Replace_list' overwrites the original list with the new list provided, and Get returns the current list.    
        Returns:
        --------
        response:   "Success" upon successful request
        """
        from aimodelshare.generatemodelapi import update_access_list as update_list
        update = update_list(apiurl = self.playground_url, email_list=email_list, update_type=update_type)
        return update

    def update_model(self): 
        return


    def update_preprocessor(self): 
        return


    def update_example_data(self, example_data): 

        """
        Updates example data associated with a model playground prediction API.

        Parameters:
        -----------

        `example_data`: ``Example of X data that will be shown on the online Playground page``
        If no example data is submitted, certain functionalities may be limited, including the deployment of live prediction APIs.

        Returns:
        --------
        response:   "Success" upon successful request
        """
        
        from aimodelshare.generatemodelapi import _create_exampledata_json

        _create_exampledata_json(self.model_type, example_data)
        
        temp_dir = tempfile.gettempdir()
        exampledata_json_filepath = temp_dir+ "/exampledata.json"


        from aimodelshare.aws import get_s3_iam_client
        s3, iam, region = get_s3_iam_client(os.environ.get("AWS_ACCESS_KEY_ID_AIMS"), os.environ.get("AWS_SECRET_ACCESS_KEY_AIMS"), os.environ.get("AWS_REGION_AIMS"))

        unique_model_id = self.playground_url.split(".")[0].split("//")[-1]

        s3["client"].upload_file(exampledata_json_filepath, os.environ.get("BUCKET_NAME"), unique_model_id + "/exampledata.json") 


        variablename_and_type_data = extract_varnames_fromtrainingdata(example_data)

        bodydata = {"apiurl":self.playground_url,
                  "apideveloper":os.environ.get("username"),
                  "versionupdateput":"TRUE",
                  "input_feature_dtypes": variablename_and_type_data[0],
                  "input_feature_names": variablename_and_type_data[1],
                  "exampledata":"TRUE"}

        headers_with_authentication = {'Content-Type': 'application/json', 'authorizationToken': os.environ.get("JWT_AUTHORIZATION_TOKEN"), 'Access-Control-Allow-Headers':
                                        'Content-Type,X-Amz-Date,authorizationToken,Access-Control-Allow-Origin,X-Api-Key,X-Amz-Security-Token,Authorization', 'Access-Control-Allow-Origin': '*'}
        response = requests.post("https://bhrdesksak.execute-api.us-east-1.amazonaws.com/dev/modeldata",
                                  json=bodydata, headers=headers_with_authentication)

        return



class Competition:
    """
    Parameters:
    ----------
    `playground_url`: playground_url attribute of ModelPlayground class or ``string``
        of existing ModelPlayground URL
    """

    submission_type = "competition"

    def __init__(self, playground_url):
        self.playground_url = playground_url
    
    def __str__(self):
        return f"Competition class instance for playground: {self.playground_url}"
        
    def submit_model(self, model_filepath, preprocessor_filepath, prediction_submission, 
        sample_data=None, reproducibility_env_filepath=None, custom_metadata=None, input_dict=None, print_output=True):
        """
        Submits model/preprocessor to machine learning competition using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated
        
        Parameters:
        -----------
        `model_filepath`:  ``string`` ends with '.onnx'
            value - Absolute path to model file [REQUIRED] to be set by the user
            .onnx is the only accepted model file extension
            "example_model.onnx" filename for file in directory.
            "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `prediction_submission`:   one hot encoded y_pred
            value - predictions for test data
            [REQUIRED] for evaluation metrics of the submitted model
        `preprocessor_filepath`:   ``string``, default=None
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
 
        Returns:
        --------
        response:   Model version if the model is submitted sucessfully
        """

        from aimodelshare.model import submit_model
        submission = submit_model(model_filepath = model_filepath, 
                              apiurl = self.playground_url,
                              prediction_submission = prediction_submission, 
                              preprocessor = preprocessor_filepath,
                              reproducibility_env_filepath = reproducibility_env_filepath,
                              custom_metadata = custom_metadata, 
                              submission_type = self.submission_type,
                              input_dict=input_dict,
                              print_output=print_output)


    def instantiate_model(self, version=None, trained=False, reproduce=False): 
        """
        Import a model previously submitted to the competition leaderboard to use in your session
        Parameters:
        -----------
        `version`: ``int``
            Model version number from competition leaderboard
        `trained`: ``bool, default=False``
            if True, a trained model is instantiated, if False, the untrained model is instantiated
       
        Returns:
        --------
        model: model chosen from leaderboard
        """
        from aimodelshare.aimsonnx import instantiate_model
        model = instantiate_model(apiurl=self.playground_url, trained=trained, version=version, 
            reproduce=reproduce, submission_type=self.submission_type)
        return model

    def replicate_model(self,version=None): 
        """
        Instantiate an untrained model previously submitted to the competition leaderboard with its reproducibility environment setup. 
        
        Parameters: 
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
          
        Returns:
        --------
        model:  model chosen from leaderboard     
        """
        
        model = self.instantiate_model(version=version,trained = False,reproduce=True) 
        return model 
    
    def set_model_reproducibility_env(self,version=None): 
        """
        Set the reproducibility environment prior to instantiating an untrained model previously submitted to the competition leaderboard.
        
        Parameters: 
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
          
        Returns:
        --------
        Sets environment according to reproducibility.json from model if present.  
        """    
        from aimodelshare.reproducibility import import_reproducibility_env_from_competition_model
        import_reproducibility_env_from_competition_model(apiurl=self.playground_url,version = version,submission_type=self.submission_type)


    def inspect_model(self, version=None, naming_convention=None):
        """
        Examine structure of model submitted to a competition leaderboard
        Parameters:
        ----------
        `version` : ``int``
            Model version number from competition leaderboard
      
        Returns:
        --------
        inspect_pd : dictionary of model summary & metadata
        """
        from aimodelshare.aimsonnx import inspect_model

        inspect_pd = inspect_model(apiurl=self.playground_url, version=version, 
            naming_convention=naming_convention, submission_type = self.submission_type)

        return inspect_pd

    def compare_models(self, version_list="None", by_model_type=None, best_model=None, verbose=1, naming_convention=None):
        """
        Compare the structure of two or more models submitted to a competition leaderboard.
        Use in conjuction with stylize_compare to visualize data. 
        
        Parameters:
        -----------
        `version_list` = ``list of int``
            list of model version numbers to compare (previously submitted to competition leaderboard) 
        `verbose` = ``int``
            controls the verbosity: the higher, the more detail 
        
        Returns:
        --------
        data : dictionary of model comparison information
        """
        from aimodelshare.aimsonnx import compare_models as compare
        data = compare(apiurl = self.playground_url, 
                      version_list = version_list, 
                      by_model_type = by_model_type,
                      best_model = best_model, 
                      verbose = verbose,
                      naming_convention=naming_convention,
                      submission_type = self.submission_type)
        return data

    def stylize_compare(self, compare_dict, naming_convention="keras"):
        """
        Stylizes data received from compare_models to highlight similarities & differences.
        Parameters:
        -----------
        `compare_dict` = dictionary of model data from compare_models
 
        Returns:
        --------
        formatted table of model comparisons 
        """
        from aimodelshare.aimsonnx import stylize_model_comparison
        stylized_compare = stylize_model_comparison(comp_dict_out=compare_dict, naming_convention=naming_convention)
        return(stylized_compare)

    def inspect_y_test(self):
        """
        Examines structure of y-test data to hep users understand how to submit models to the competition leaderboad.
        Parameters:
        ------------
        None
 
        Returns:
        --------
        dictionary of a competition's y-test metadata
        """
        from aimodelshare.aimsonnx import inspect_y_test
        data = inspect_y_test(apiurl = self.playground_url, submission_type=self.submission_type)
        return data
    
    def get_leaderboard(self, verbose=3, columns=None):
        """
        Get current competition leaderboard to rank all submitted models.
        Use in conjuction with stylize_leaderboard to visualize data. 
        
        Parameters:
        -----------
        `verbose` : optional, ``int``
            controls the verbosity: the higher, the more detail 
        `columns` : optional, ``list of strings``
            list of specific column names to include in the leaderboard, all else will be excluded
            performance metrics will always be displayed
        
        Returns:
        --------
        dictionary of leaderboard data 
        """
        from aimodelshare.leaderboard import get_leaderboard
        data = get_leaderboard(verbose=verbose,
                 columns=columns, 
                 apiurl = self.playground_url, 
                 submission_type=self.submission_type)
        return data
    
    def stylize_leaderboard(self, leaderboard, naming_convention="keras"):
        """
        Stylizes data received from get_leaderbord.
        Parameters:
        -----------
        `leaderboard` : data dictionary object returned from get_leaderboard
        Returns:
        --------
        Formatted competition leaderboard
        """
        from aimodelshare.leaderboard import stylize_leaderboard as stylize_lead
        stylized_leaderboard = stylize_lead(leaderboard = leaderboard, naming_convention=naming_convention)
        return stylized_leaderboard
    
    def update_access_list(self, email_list=[],update_type="Replace_list"):
        """
        Updates list of authenticated participants who can submit new models to a competition.
        Parameters:
        -----------
        `apiurl`: string
                URL of deployed prediction API 
          
        `email_list`: [REQUIRED] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
        `update_type`:[REQUIRED] options, ``string``: 'Add', 'Remove', 'Replace_list','Get. Add appends user emails to original list, Remove deletes users from list, 
                  'Replace_list' overwrites the original list with the new list provided, and Get returns the current list.    
        Returns:
        --------
        response:   "Success" upon successful request
        """
        from aimodelshare.generatemodelapi import update_access_list as update_list
        update = update_list(apiurl = self.playground_url, 
            email_list=email_list,update_type=update_type,
            submission_type=self.submission_type)
        return update



class Experiment(Competition):
    """
    Parameters:
    ----------
    `playground_url`: playground_url attribute of ModelPlayground class or ``string``
        of existing ModelPlayground URL
    """

    submission_type = "experiment"

    def __init__(self, playground_url):
        self.playground_url = playground_url
    
    def __str__(self):
        return f"Experiment class instance for playground: {self.playground_url}"
        
    

class Data: 
    def __init__(self, data_type, playground_url=None):
        self.data_type = data_type
        self.playground_url = playground_url 
    
    def __str__(self):
        return f"This is a description of the Data class."

    def share_dataset(self, data_directory="folder_file_path", classification="default", private="FALSE"): 
        from aimodelshare.data_sharing.share_data import share_dataset as share
        response = share(data_directory=data_directory, classification=classification, private=private)
        return response
    
    def download_data(self, repository):
        from aimodelshare.data_sharing.download_data import download_data as download
        datadownload = download(repository)
        return datadownload
