.. _aimodelshare_tutorial: 

AI Model Share Tutorial
#######################

This tutorial will take you through the core functionality of the AI Model Share library, with the Titanic passenger data set. You will import the Titanic data set, build a machine learning model for tabular classification, deploy an interactive web dashboard (ModelPlayground) powered by a REST API, create a competition, and explore how to learn from the model collaboration process. 

This tutorial is applicable for all new users, especially those that are interested in publishing their machine learning models on a live webpage to generate predictions, improving on existing models through the model collaboration process, and/or hosting a competition for other users. 

If you have been invited to participate in an existing competition (or would like to submit a model to any of our ongoing public competitions), you may wish to skip to one of the Model Submission Guides included in the :ref:`example_notebooks`. 

The only thing to need to complete the tutorial is a computer with an internet connection.


.. _getting_started:

Getting Started 
***************

.. _cred_configure:

Credential Configuration
========================

To complete this tutorial, you will need to have a pre-formatted credentials file. Follow the directions :ref:`HERE <create_credentials>` to create one. 

.. _set_environment:

Set Up Environment
==================

Use your credentials file to set your credentials for all aimodelshare functions. This will give you access to your AI Model Share account and your AWS resources in order to deploy a Model Playground.

.. code-block::

	# Set credentials 
	from aimodelshare.aws import set_credentials
	set_credentials(credential_file="credentials.txt", type="deploy_model")
	
	# Get materials for tutorial
	import aimodelshare as ai
	X_train, X_test, y_train_labels, y_test, example_data, y_test_labels = ai.import_quickstart_data("titanic")


.. _part_one:

Part One: Deploy a Model Playground
***********************************

This tutorial will use data from the Titanic passenger data set. We will use attributes of the passengers on board to determine whether they survived or died in the 1912 Shipwreck.

	At the end of part one, you will have built a model for tabular classification which will take passenger characteristics and predict if they survived or died in the Titanic shipwreck. You will have deployed that model into a "Model Playground", which is an interactive web application that will use your model to generate predictions in a user-friendly dashboard. Additionally, users will have access to customized code to use the background REST API to generate predictions programatically. 

.. _step_one:

Step 1: Preprocessor Function & Setup
=====================================

Preprocessor functions are used to preprocess data into the precise format your model requires to generate predictions. An example preprocessor using sklearn's Column Transformer is included below, but you can customize your preprocessor however you see fit. 

.. note::
    Preprocessor functions should always be named "preprocessor".

    You can use any Python library in a preprocessor function, but all libraries should be imported inside your preprocessor function.

    For tabular prediction, models users should minimally include function inputs for an unpreprocessed pandas dataframe. Any categorical features should be preprocessed to one hot encoded numeric values.

Set Up Preprocessor:: 

	# In this case we use Sklearn's Column transformer in our preprocessor function
	from sklearn.compose import ColumnTransformer
	from sklearn.pipeline import Pipeline
	from sklearn.impute import SimpleImputer
	from sklearn.preprocessing import StandardScaler, OneHotEncoder

	#Preprocess data using sklearn's Column Transformer approach

	# We create the preprocessing pipelines for both numeric and categorical data.
	numeric_features = ['age', 'fare']
	numeric_transformer = Pipeline(steps=[
    		('imputer', SimpleImputer(strategy='median')), #'imputer' names the step
    		('scaler', StandardScaler())])

	categorical_features = ['embarked', 'sex', 'pclass']

	# Replacing missing values with Modal value and then one-hot encoding.
	categorical_transformer = Pipeline(steps=[
    		('imputer', SimpleImputer(strategy='most_frequent')),
    		('onehot', OneHotEncoder(handle_unknown='ignore'))])

	# Final preprocessor object set up with ColumnTransformer...
	preprocess = ColumnTransformer(
    		transformers=[
       		('num', numeric_transformer, numeric_features),
        	('cat', categorical_transformer, categorical_features)])

	# fit preprocessor to your data
	preprocess = preprocess.fit(X_train)

Preprocessor Function:: 
	
	# Here is where we actually write the preprocessor function:

	# Write function to transform data with preprocessor 
	# In this case we use sklearn's Column transformer in our preprocessor function

	def preprocessor(data):
    		preprocessed_data=preprocess.transform(data)
    		return preprocessed_data

Check X Data::

	# check shape of X data after preprocessing it using our new function
	preprocessor(X_train).shape

One Hot Encode y_data::

	# Create one hot encoded data from list of y_train category labels
	#...to allow `ModelPlayground.deploy()` to extract correct labels for predictions in your deployed API
	import pandas as pd
	y_train = pd.get_dummies(y_train_labels)

	#ensure column names are correct in one hot encoded target for correct label extraction
	list(y_train.columns)

.. _step_two:

Step 2 - Build Model
====================

Build Model Using sklearn (or your preferred Machine Learning Library). This is the model that will ultimately power your REST API and Model Playground. The model and preprocessor can be updated at any time by the Model Playground owner. 

.. code-block::

	from sklearn.linear_model import LogisticRegression

	model = LogisticRegression(C=10, penalty='l1', solver = 'liblinear')
	model.fit(preprocessor(X_train), y_train_labels) # Fitting to the training set.
	model.score(preprocessor(X_train), y_train_labels) # Fit score, 0-1 scale. 

.. _step_three:

Step 3 - Save Preprocessor
==========================

Save preprocessor function to "preprocessor.zip" file. The preprocessor code will be included in the Model Playground and executed to preprocess data submitted for predictions. 

.. code-block:: 

	import aimodelshare as ai
	ai.export_preprocessor(preprocessor,"")

.. code-block:: 

	#  Now let's import and test the preprocessor function to see if it is working...

	import aimodelshare as ai
	prep=ai.import_preprocessor("preprocessor.zip")
	prep(example_data).shape

.. _step_four:

Step 4 - Save sklearn Model to Onnx File Format
===============================================

.. code-block:: 

	# Save sklearn model to local ONNX file
	from aimodelshare.aimsonnx import model_to_onnx

	# Check how many preprocessed input features there are
	from skl2onnx.common.data_types import FloatTensorType
	initial_type = [('float_input', FloatTensorType([None, 10]))]  # Insert correct number of features in preprocessed data

	onnx_model = model_to_onnx(model, framework='sklearn',
                     	initial_types=initial_type,
                        transfer_learning=False,deep_learning=False)

	with open("model.onnx", "wb") as f:
    		f.write(onnx_model.SerializeToString())

.. _step_five:

Step 5 - Create your Model Playground and Deploy REST API/Live Web-Application
==============================================================================

.. code-block::  

	#Set up arguments for Model Playground deployment
	import pandas as pd 

	model_filepath="model.onnx"
	preprocessor_filepath="preprocessor.zip"
	exampledata = example_data

.. code-block::  

	from aimodelshare import ModelPlayground

	#Instantiate ModelPlayground() Class

	myplayground=ModelPlayground(model_type="tabular", classification=True, private=False)

	# Create Model Playground (generates live rest api and web-app for your model/preprocessor)

	myplayground.deploy(model_filepath, preprocessor_filepath, y_train_labels, exampledata)


Use your new Model Playground!
==============================

Follow the link in the output above to:

* Generate predictions with your interactive web dashboard.
* Access example code in Python, R, and Curl.

Or, follow the rest of the tutorial to create a competition for your Model Playground and:

* Access verified model performance metrics.
* Upload multiple models to a leaderboard.
* Easily compare model performance & structure.


.. _part_two: 

Part Two: Create a Competition 
******************************

After deploying your Model Playground, you can now create a competition. Creating a competition allows you to:

* Verify the model performance metrics on aimodelshare.org.
* Submit models to a leaderboard.
* Grant access to other users to submit models to the leaderboard.
* Easily compare model performance and structure.

.. code-block:: 

	# Create list of authorized participants for competition
	# Note that participants should use the same email address when creating modelshare.org account

	emaillist=["emailaddress1@email.com", "emailaddress2@email.com", "emailaddress3@email.com"]

.. code-block:: 

	# Create Competition
	# Note -- Make competition public (allow any AI Model Share user to submit models) 
	# .... by excluding the email_list argument and including the 'public=True' argument 

	myplayground.create_competition(data_directory='titanic_competition_data', 
                               		 y_test = y_test_labels, 
                          	     #   email_list=emaillist)
                          	         public=True)

.. code-block:: 

	#Instantiate Competition
	#--Note: If you start a new session, the first argument should be the Model Playground url in quotes. 
	#--e.g.- mycompetition= ai.Competition("https://2121212.execute-api.us-east-1.amazonaws.com/prod/m)
	#See Model Playground "Compete" tab for example model submission code.

	mycompetition= ai.Competition(myplayground.playground_url)

.. code-block:: 

	# Add, remove, or completely update authorized participants for competition later
	emaillist=["emailaddress4@email.com"]
	mycompetition.update_access_list(email_list=emaillist,update_type="Add")

.. _submit_models_to_comp:

Submit Models
=============

After a competition is created, users can submit models to be tracked in the competition leaderboard. When models are submitted, model metadata is extracted and model performance metrics are generated. 

.. note::
	There may be two leaderboards associated with every competition: a "public" leaderboard, visible to everyone with access to the competition, and a "private" leaderboard, visible to only the competition owner. Competition owners may choose to create the private leaderboard for the purpose of evaluating models with a special subset of held out y-test data. This encourages the development of models that are generalizable to additional real-world data, and not overfit to a specific split of data. 


.. code-block:: 

	#Authorized users can submit new models after setting credentials using modelshare.org username/password
	from aimodelshare.aws import set_credentials

	apiurl=myplayground.playground_url # example url from deployed playground: apiurl= "https://123456.execute-api.us-east-1.amazonaws.com/prod/m

	set_credentials(apiurl=apiurl)

.. code-block:: 

	#Submit Model 1: 

	#-- Generate predicted values (a list of predicted labels "survived" or "died") (Model 1)
	prediction_labels = model.predict(preprocessor(X_test))

	# Submit Model 1 to Competition Leaderboard
	mycompetition.submit_model(model_filepath = "model.onnx",
                                 preprocessor_filepath="preprocessor.zip",
                                 prediction_submission=prediction_labels)

Create, save, and submit a second model::  

	# Create model 2 (L2 Regularization - Ridge)
	from sklearn.linear_model import LogisticRegression

	model_2 = LogisticRegression(C=.01, penalty='l2')
	model_2.fit(preprocessor(X_train), y_train_labels) # Fitting to the training set.
	model_2.score(preprocessor(X_train), y_train_labels) # Fit score, 0-1 scale.

.. code-block::  

	# Save Model 2 to .onnx file

	# How many preprocessed input features there are
	from skl2onnx.common.data_types import FloatTensorType
	initial_type = [('float_input', FloatTensorType([None, 10]))]  

	onnx_model = model_to_onnx(model_2, framework='sklearn',
                          initial_types=initial_type,
                          transfer_learning=False,
                          deep_learning=False)

	# Save model to local .onnx file
	with open("model_2.onnx", "wb") as f:
    		f.write(onnx_model.SerializeToString())

.. code-block:: 

	# Submit Model 2

	#-- Generate predicted y values (Model 2)
	prediction_labels = model_2.predict(preprocessor(X_test))

	# Submit Model 2 to Competition Leaderboard
	mycompetition.submit_model(model_filepath = "model_2.onnx",
                                 prediction_submission=prediction_labels,
                                 preprocessor_filepath="preprocessor.zip")

.. _learn:

Learn From Submitted Models
===========================

The leaderboard is a helpful tool for not only examining your model's current standing in an active competition, but also for learning about which model structures most and least effective for a particular data set. Authorized competition users can download the current leaderboard for an overall understanding of model metadata and ranking, and then compare certain models to examine their metadata more closely. 


Get Leaderboard:: 

	data = mycompetition.get_leaderboard()
	mycompetition.stylize_leaderboard(data)

Compare Models:: 

	# Compare two or more models
	data=mycompetition.compare_models([1,2], verbose=1)
	mycompetition.stylize_compare(data)

.. note::
	``Competition.compare_models()`` is maximally useful for comparing models with the same basic structure.

Users can also check the structure of the y test data. This helps users understand how to submit predicted values to leaderboard. 

Check Structure of y-test data:: 	

	mycompetition.inspect_y_test()

.. _part_three:

Part Three: Maintaining your Model Playground
*********************************************

Update Runtime model

Use this function to: 

#. Update the prediction API behind your Model Playground with a new model, chosen from the leaderboard, and. 
#. Verify the model performance metrics in your Model Playground.

.. code-block:: 

	myplayground.update_runtime_model(model_version=1)


Delete Deployment

Use this function to delete the entire Model Playground, including the REST API, web dashboard, competition, and all submitted models

.. code-block:: 

	myplayground.delete_deployment()
