from aimodelshare.playground import ModelPlayground, Experiment, Competition
from aimodelshare.aws import set_credentials, get_aws_token
import aimodelshare as ai
from aimodelshare.data_sharing.utils import redo_with_write

from unittest.mock import patch

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

import pandas as pd
import shutil
import os




# def test_set_credentials():

#	set_credentials(credential_file="../../credentials.txt", type="deploy_model")


# def test_quickstart_sklearn():

# 	X_train, X_test, y_train, y_test, example_data, y_test_labels = ai.import_quickstart_data("titanic")

# 	assert isinstance(X_train, pd.DataFrame)
# 	assert isinstance(X_test, pd.DataFrame)
# 	assert isinstance(y_train, pd.Series)
# 	assert isinstance(y_test, pd.Series)
# 	assert isinstance(example_data, pd.DataFrame)
# 	assert isinstance(y_test_labels, list)


def test_configure_credentials():

	# mock user input
	inputs = [os.environ.get('USERNAME'),
			  os.environ.get('PASSWORD'),
			  os.environ.get('AWS_ACCESS_KEY_ID'),
			  os.environ.get('AWS_SECRET_ACCESS_KEY'),
			  os.environ.get('AWS_REGION')]

	with patch("getpass.getpass", side_effect=inputs):
		from aimodelshare.aws import configure_credentials
		configure_credentials()

	# clean up credentials file
	os.remove("credentials.txt")


def test_playground_sklearn():

	# mock user input
	inputs = [os.environ.get('USERNAME'),
			  os.environ.get('PASSWORD'),
			  os.environ.get('AWS_ACCESS_KEY_ID'),
			  os.environ.get('AWS_SECRET_ACCESS_KEY'),
			  os.environ.get('AWS_REGION')]

	with patch("getpass.getpass", side_effect=inputs):
		from aimodelshare.aws import configure_credentials
		configure_credentials()

	# set credentials
	set_credentials(credential_file="credentials.txt", type="deploy_model")
	#os.environ["AWS_TOKEN"]=get_aws_token()

	# clean up credentials file
	os.remove("credentials.txt")

	# Get materials for tutorial
	X_train, X_test, y_train, y_test, example_data, y_test_labels = ai.import_quickstart_data("titanic")

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

	# Write function to transform data with preprocessor 
	# In this case we use sklearn's Column transformer in our preprocessor function
	def preprocessor(data):
	    preprocessed_data=preprocess.transform(data)
	    return preprocessed_data

	# check shape of X data after preprocessing it using our new function
	assert preprocessor(X_train).shape == (1047, 10)

	# build model 1
	model = LogisticRegression(C=10, penalty='l1', solver = 'liblinear')
	model.fit(preprocessor(X_train), y_train)
	model.score(preprocessor(X_train), y_train)

	# generate predictions
	prediction_labels = model.predict(preprocessor(X_test))

	# Instantiate ModelPlayground() Class
	myplayground=ModelPlayground(input_type="tabular", task_type="classification", private=True)

	# Create Model Playground page
	myplayground.create(eval_data = y_test_labels)

	# Submit Model to Experiment Leaderboard
	myplayground.submit_model(model = model,
							  preprocessor=preprocessor,
							  prediction_submission=prediction_labels, 
							  input_dict={"description": "", "tags": ""},
							  submission_type="all")

	# build model 2
	model_2 = LogisticRegression(C=.01, penalty='l2')
	model_2.fit(preprocessor(X_train), y_train) # Fitting to the training set.
	model_2.score(preprocessor(X_train), y_train) # Fit score, 0-1 scale.

	# generate predictions
	prediction_labels = model_2.predict(preprocessor(X_test))

	# Submit Model 2 to Experiment
	myplayground.submit_model(model= model_2,
	                          preprocessor=preprocessor,
	                          prediction_submission=prediction_labels,
	                          input_dict={"description": "", "tags": ""},
	                          submission_type="all")

	#submit model through competition
	mycompetition = ai.playground.Competition(myplayground.playground_url)
	mycompetition.submit_model(model=model_2,
							   preprocessor=preprocessor,
							   prediction_submission=prediction_labels,
							   input_dict={"description": "", "tags": ""}
							   )

	#submit model through experiment
	myexperiment = ai.playground.Experiment(myplayground.playground_url)
	myexperiment.submit_model(model=model_2,
							   preprocessor=preprocessor,
							   prediction_submission=prediction_labels,
							  input_dict={"description": "", "tags": ""}
							  )

	# Check Competition Leaderboard
	data = myplayground.get_leaderboard()
	myplayground.stylize_leaderboard(data)
	assert isinstance(data, pd.DataFrame)

	# Compare two or more models
	data = myplayground.compare_models([1,2], verbose=1)
	myplayground.stylize_compare(data)
	assert isinstance(data, (pd.DataFrame, dict))

	# Check structure of evaluation data
	data = myplayground.inspect_eval_data()
	assert isinstance(data, dict)

	# deploy model
	myplayground.deploy_model(model_version=1, example_data=example_data, y_train=y_train)

	# update example data
	myplayground.update_example_data(example_data)

	# swap out runtime model
	myplayground.update_runtime_model(model_version=1)

	# delete
	myplayground.delete_deployment(confirmation=False)

	# local cleanup 
	shutil.rmtree("titanic_competition_data", onerror=redo_with_write)
	shutil.rmtree("titanic_quickstart", onerror=redo_with_write)



def test_playground_keras():

	# mock user input
	inputs = [os.environ.get('USERNAME'),
			  os.environ.get('PASSWORD'),
			  os.environ.get('AWS_ACCESS_KEY_ID'),
			  os.environ.get('AWS_SECRET_ACCESS_KEY'),
			  os.environ.get('AWS_REGION')]

	with patch("getpass.getpass", side_effect=inputs):
		from aimodelshare.aws import configure_credentials
		configure_credentials()

	# set credentials
	set_credentials(credential_file="credentials.txt", type="deploy_model")
	# os.environ["AWS_TOKEN"]=get_aws_token()

	# clean up credentials file
	os.remove("credentials.txt")

	# # Download flower image data and and pretrained Keras models
	from aimodelshare.data_sharing.download_data import import_quickstart_data
	keras_model, y_train_labels = import_quickstart_data("flowers")
	keras_model_2, y_test_labels = import_quickstart_data("flowers", "competition")

	# Here is a pre-designed preprocessor, but you could also build your own to prepare the data differently
	def preprocessor(image_filepath, shape=(192, 192)):
	        """
	        This function preprocesses reads in images, resizes them to a fixed shape and
	        min/max transforms them before converting feature values to float32 numeric values
	        required by onnx files.
	        
	        params:
	            image_filepath
	                full filepath of a particular image
	                      
	        returns:
	            X
	                numpy array of preprocessed image data
	        """
	           
	        import cv2
	        import numpy as np

	        "Resize a color image and min/max transform the image"
	        img = cv2.imread(image_filepath) # Read in image from filepath.
	        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads in images in order of blue green and red, we reverse the order for ML.
	        img = cv2.resize(img, shape) # Change height and width of image.
	        img = img / 255.0 # Min-max transform.


	        # Resize all the images...
	        X = np.array(img)
	        X = np.expand_dims(X, axis=0) # Expand dims to add "1" to object shape [1, h, w, channels] for keras model.
	        X = np.array(X, dtype=np.float32) # Final shape for onnx runtime.
	        return X

	# Preprocess X_test image data to generate predictions from models 
	import numpy as np

	# Generate filenames
	file_names = [('flower_competition_data/test_images/' + str(i) + '.jpg') for i in range(1, 735)]

	# Apply preprocessor to image data
	preprocessed_image_data = [preprocessor(x) for x in file_names]

	# Create single X_test array from preprocessed images
	X_test = np.vstack(preprocessed_image_data) 

	# One-hot encode y_train labels (y_train.columns used to generate prediction labels below)
	import pandas as pd
	y_train = pd.get_dummies(y_train_labels)

	# Generate predicted y values
	prediction_column_index=keras_model.predict(X_test).argmax(axis=1)

	# Extract correct prediction labels 
	prediction_labels = [y_train.columns[i] for i in prediction_column_index]

	# Instantiate Model Playground object
	from aimodelshare.playground import ModelPlayground
	myplayground=ModelPlayground(input_type="image", task_type="classification", private=False)
	# Create Model Playground Page on modelshare.ai website
	myplayground.create(eval_data=y_test_labels)

	# Submit Model to Experiment Leaderboard
	myplayground.submit_model(model=keras_model,
	                          preprocessor=preprocessor,
	                          prediction_submission=prediction_labels,
	                          input_dict={"description": "", "tags": ""},
	                          submission_type="all")

	# Deploy model by version number
	myplayground.deploy_model(model_version=1, example_data="quickstart_materials/example_data", y_train=y_train)

	# example url from deployed playground: apiurl= "https://123456.execute-api.us-east-1.amazonaws.com/prod/m
	apiurl=myplayground.playground_url 


	# Submit Model 2
	# Generate predicted y values (Model 2)
	prediction_column_index=keras_model_2.predict(X_test).argmax(axis=1)

	# extract correct prediction labels (Model 2)
	prediction_labels = [y_train.columns[i] for i in prediction_column_index]

	# Submit Model 2 to Experiment Leaderboard
	myplayground.submit_model(model=keras_model_2,
	                            preprocessor=preprocessor,
	                            prediction_submission=prediction_labels,
	                            input_dict={"description": "", "tags": ""},
	                            submission_type="all")

	#submit model through competition
	mycompetition = ai.playground.Competition(myplayground.playground_url)
	mycompetition.submit_model(model=keras_model_2,
							   preprocessor=preprocessor,
							   prediction_submission=prediction_labels,
							   input_dict={"description": "", "tags": ""}
							   )

	#submit model through experiment
	myexperiment = ai.playground.Experiment(myplayground.playground_url)
	myexperiment.submit_model(model=keras_model_2,
							   preprocessor=preprocessor,
							   prediction_submission=prediction_labels,
							  input_dict={"description": "", "tags": ""}
							  )

	# Check experiment leaderboard
	data = myplayground.get_leaderboard()
	myplayground.stylize_leaderboard(data)
	assert isinstance(data, pd.DataFrame)

	# Compare two or more models
	data = myplayground.compare_models([1,2], verbose=1)
	myplayground.stylize_compare(data)
	assert isinstance(data, (pd.DataFrame, dict))

	# Check structure of evaluation data
	data = myplayground.inspect_eval_data()
	assert isinstance(data, dict)

	# Update runtime model
	myplayground.update_runtime_model(model_version=2)

	# delete
	myplayground.delete_deployment(confirmation=False)

	# local cleanup 
	shutil.rmtree("flower_competition_data", onerror=redo_with_write)
	shutil.rmtree("quickstart_materials", onerror=redo_with_write)
	shutil.rmtree("quickstart_flowers_competition", onerror=redo_with_write)
