Supporting Functions
====================

.. _configure_credentials:

configure_credentials()
-----------------------

.. py:function:: aimodelshare.aws.configure_credentials()

   Return a formatted credentials file built with user inputs.

Combine your AI Model Share & AWS credentials into a single ‘credentials.txt' file with the `configure_credentials` function. You only have to make the file once, then you can use it whenever you use the aimodelshare library. 

Credentials files must follow this format: 
	
	.. image:: images/creds_file_example.png
   			:width: 600

The following code will prompt you to provide your credentials one at a time and pre-format a txt file for you to use in the future: 


Example ::

	#install aimodelshare library
	! pip install aimodelshare

	# Generate credentials file
	import aimodelshare as ai 
	from aimodelshare.aws import configure_credentials 
	configure_credentials()
	

.. _set_credentials:

set_credentials()
-----------------

Set credentials for all AI Model Share functions with
the ``aimodelshare.aws.set_credentials()`` function:

.. py:function:: aimodelshare.aws.set_credentials(credential_file="credentials.txt", type="submit_model", apiurl)

   Set credentials for AI Model Share and Amazon Web Services (AWS). 

   :param credential_file: Path to formatted credentials txt file.
   :type credential_file: string

   :param type: set to "deploy_model" to deploy a ModelPlayground.
   :type type: string

   :param apiurl: unique api_url that powers a specific Model Playground. 
   :type apiurl: string

   :return: Success Message.

Example :: 

	# Deploying ModelPlaygrounds - Requires AWS credentials
	from aimodelshare.aws import set_credentials
	set_credentials(credential_file="credentials.txt", type="deploy_model")

	# Submitting Models to Competition - No AWS credentials required 
	from aimodelshare.aws import set_credentials
	apiurl="https://example.execute-api.us-east-1.amazonaws.com/prod/m"
	set_credentials(apiurl=apiurl)

.. _download_data:

download_data()
---------------

Download data sets that have been shared to AI ModelShare with the ``aimodelshare.data_sharing.download_data()`` function: 

.. py:function:: aimodelshare.data_sharing.download_data(repository)

   Download data that has been shared to the AI ModelShare website.

   :param repository: URI & image_tag of uploaded data (provided with the create_competition method of the Model Playground class) 
   :type repository: string
   :return: Success Message & downloaded data directory

Example :: 

	from aimodelshare import download_data
	download_data('example-repository:image_tag') 

.. export_eval_metric:

export_eval_metric()
--------------------

.. py:function:: aimodelshare.custom_eval_metrics.export_eval_metric(eval_metric_fxn, directory, name) 
   
   Export evaluation metric and related objects into zip file for model deployment

   :param eval_metric_fxn: name of eval metric function (should always be named "eval_metric" to work properly)
   :type eval_metric_fxn: string
   :param directory: folderpath to eval metric function
               use "" to reference current working directory
   :type directory: string
   :param name: name of the custom eval metric
   :type name: string
   :return: file named 'name.zip' in the correct format for model deployment

Example :: 

	from aimodelshare import export_eval_metric
	export_eval_metric(eval_metric_fxn, directory, name) 

.. export_reproducibility_env:

export_reproducibility_env()
----------------------------

.. py:function:: aimodelshare.reproducibility.export_reproducibility_env(seed, directory, mode) 

   Export development environment to enable reproducibility of your model.

   :param seed: Random Seed 
   :type seed: Int
   :directory: Directory for completed json file 
   :type directory: string
   :param mode: Processor - either "gpu" or "cpu"
   :type mode: string
   :return: “./reproducibility.json” file to use with submit_model() 

Example :: 

	from aimodelshare import export_reproducibility_env
	export_eval_metric(seed, directory, mode) 

.. _share_dataset:

share_dataset()
---------------

Upload data sets to AI ModelShare with the ``aimodelshare.data_sharing.share_dataset()`` function: 

.. py:function:: aimodelshare.data_sharing.share_dataset(data_directory="folder_file_path",classification="default", private="FALSE")

   Upload data to the AI ModelShare website.

   :param data_directory: path to the file directory to upload.
   :type data_directory: string
   :return: Success Message 

Example :: 

	from aimodelshare.data_sharing.share_data import share_dataset
	share_dataset(data_directory = "example_path", classification="default", private="FALSE")
