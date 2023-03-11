from aimodelshare.playground import ModelPlayground, Experiment, Competition
from aimodelshare.aws import set_credentials
import aimodelshare as ai
from aimodelshare.data_sharing.utils import redo_with_write

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

import pandas as pd
import shutil




def test_set_credentials():

	set_credentials(credential_file="../../credentials_private.txt", type="deploy_model")


# def test_quickstart_sklearn():

# 	X_train, X_test, y_train, y_test, example_data, y_test_labels = ai.import_quickstart_data("titanic")

# 	assert isinstance(X_train, pd.DataFrame)
# 	assert isinstance(X_test, pd.DataFrame)
# 	assert isinstance(y_train, pd.Series)
# 	assert isinstance(y_test, pd.Series)
# 	assert isinstance(example_data, pd.DataFrame)
# 	assert isinstance(y_test_labels, list)


def test_playground_sklearn(): 

	# set credentials
	set_credentials(credential_file="../../credentials_private.txt", type="deploy_model")

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
	preprocessor(X_train).shape

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
							  input_dict={"description": "", "tags": ""})

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

	# Check Competition Leaderboard
	data = myplayground.get_leaderboard()
	myplayground.stylize_leaderboard(data)

	# Compare two or more models
	data=myplayground.compare_models([1,2], verbose=1)
	myplayground.stylize_compare(data)

	# Check structure of evaluation data
	myplayground.inspect_eval_data()

	# deploy model
	myplayground.deploy_model(model_version=1, example_data=example_data, y_train=y_train)

	# myplayground.update_example_data(example_data)

	# # swap out runtime model
	# myplayground.update_runtime_model(model_version=1)

	# delete
	myplayground.delete_deployment(confirmation=False)

	# local cleanup 
	shutil.rmtree("titanic_competition_data", onerror=redo_with_write)
	shutil.rmtree("titanic_quickstart", onerror=redo_with_write)


