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



def test_configure_credentials():

	# when testing locally, we can set credentials from file
	try:
		set_credentials(credential_file="../../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

	try:
		set_credentials(credential_file="../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

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

	# when testing locally, we can set credentials from file
	try:
		set_credentials(credential_file="../../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

	try:
		set_credentials(credential_file="../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

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

	# when testing locally, we can set credentials from file
	try:
		set_credentials(credential_file="../../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

	try:
		set_credentials(credential_file="../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

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


def test_playground_pytorch():

	# when testing locally, we can set credentials from file
	try:
		set_credentials(credential_file="../../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

	try:
		set_credentials(credential_file="../../credentials.txt", type="deploy_model")
	except Exception as e:
		print(e)

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

	# # Download flower image data # Download flower image file (jpg) dataset
	import aimodelshare as ai
	ai.download_data("public.ecr.aws/y2e2a1d6/flower-competition-data-repository:latest")

	# Extract filepaths to use to import and preprocess image files...
	base_path = 'flower-competition-data/train_images'
	categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

	# Load file paths to fnames list object...
	fnames = []

	for category in categories:
		flower_folder = os.path.join(base_path, category)
		file_names = os.listdir(flower_folder)
		full_path = [os.path.join(flower_folder, file_name) for file_name in file_names]
		fnames.append(full_path)

	# Here is a pre-designed preprocessor, but you could also build your own to prepare the data differently

	def preprocessor(data, shape=(128, 128)):
		"""
        This function preprocesses reads in images, resizes them to a fixed shape and
        min/max transforms them before converting feature values to float32 numeric values
        required by onnx files.

        params:
            data
                list of unprocessed images

        returns:
            X
                numpy array of preprocessed image data

        """

		import cv2
		import numpy as np

		"Resize a color image and min/max transform the image"
		img = cv2.imread(data)  # Read in image from filepath.
		img = cv2.cvtColor(img,
						   cv2.COLOR_BGR2RGB)  # cv2 reads in images in order of blue green and red, we reverse the order for ML.
		img = cv2.resize(img, shape)  # Change height and width of image.
		img = img / 255.0  # Min-max transform.

		# Resize all the images...
		X = np.array(img)
		X = np.expand_dims(X, axis=0)  # Expand dims to add "1" to object shape [1, h, w, channels].
		X = np.array(X, dtype=np.float32)  # Final shape for onnx runtime.

		# transpose image to pytorch format
		X = np.transpose(X, (0, 3, 1, 2))

		return X

	# Import image, load to array of shape height, width, channels, then min/max transform...

	# Read in all images from filenames...
	preprocessed_image_data = [preprocessor(x) for x in fnames[0] + fnames[1] + fnames[2] + fnames[3] + fnames[4]]

	# models require object to be an array rather than a list. (vstack converts above list to array object.)
	import numpy as np
	X = np.vstack(
		preprocessed_image_data)  # Assigning to X to highlight that this represents feature input data for our model.

	# Create y training label data made up of correctly ordered labels from file folders...
	from itertools import repeat

	daisy = list(repeat("daisy", 507))  # i.e.: 507 filenames in daisy folder
	dandelion = list(repeat("dandelion", 718))
	roses = list(repeat("roses", 513))
	sunflowers = list(repeat("sunflowers", 559))
	tulips = list(repeat("tulips", 639))

	# Combine into single list of y labels...
	y_labels = daisy + dandelion + roses + sunflowers + tulips

	# Check length, same as X above...
	len(y_labels)

	# get numerical representation of y labels
	import pandas as pd
	y_labels_num = pd.DataFrame(y_labels)[0].map(
		{'daisy': 4, 'dandelion': 1,  # `data_paths` has 'daisy', 'dandelion', 'sunflowers', 'roses', 'tulips'...
		 'sunflowers': 2, 'roses': 3, 'tulips': 0})  # ...but `image_paths` has 'tulips' first, and 'daisy' last.

	y_labels_num = list(y_labels_num)

	# train_test_split resized images...
	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y_labels_num,
														stratify=y_labels_num,
														test_size=0.20,
														random_state=1987)

	import torch

	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	from torch.utils.data import DataLoader, TensorDataset

	# prepare datasets for pytorch dataloader
	tensor_X_train = torch.Tensor(X_train)
	tensor_y_train = torch.tensor(y_train, dtype=torch.long)
	train_ds = TensorDataset(tensor_X_train, tensor_y_train)

	tensor_X_test = torch.Tensor(X_test)
	tensor_y_test = torch.tensor(y_test, dtype=torch.long)
	test_ds = TensorDataset(tensor_X_test, tensor_y_test)

	# set up dataloaders
	batch_size = 50
	train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
	test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

	from torch import nn

	# Define pytorch model
	class NeuralNetwork(nn.Module):
		def __init__(self):
			super(NeuralNetwork, self).__init__()
			self.flatten = nn.Flatten()
			self.linear_relu_stack = nn.Sequential(
				nn.Linear(128 * 128 * 3, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 5)
			)

		def forward(self, x):
			x = self.flatten(x)
			logits = self.linear_relu_stack(x)
			return logits

	model = NeuralNetwork().to(device)
	print(model)

	# set up loss function and optimizer
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

	# define training function
	def train(dataloader, model, loss_fn, optimizer):
		size = len(dataloader.dataset)
		model.train()
		for batch, (X, y) in enumerate(dataloader):
			X, y = X.to(device), y.to(device)

			# Compute prediction error
			pred = model(X)
			loss = loss_fn(pred, y)

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 100 == 0:
				loss, current = loss.item(), batch * len(X)
				print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

	# define testing function
	def test(dataloader, model, loss_fn):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		model.eval()
		test_loss, correct = 0, 0
		with torch.no_grad():
			for X, y in dataloader:
				X, y = X.to(device), y.to(device)
				pred = model(X)
				test_loss += loss_fn(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()
		test_loss /= num_batches
		correct /= size
		print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

	epochs = 2
	for t in range(epochs):
		print(f"Epoch {t + 1}\n-------------------------------")
		train(train_dataloader, model, loss_fn, optimizer)
		test(test_dataloader, model, loss_fn)
	print("Done!")

	# -- Generate predicted y values (Model 1)
	# Note: returns the predicted column index location for classification models
	if torch.cuda.is_available():
		prediction_column_index = model(tensor_X_test.cuda()).argmax(axis=1)
	else:
		prediction_column_index = model(tensor_X_test).argmax(axis=1)

	# extract correct prediction labels
	prediction_labels = [['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'][i] for i in prediction_column_index]

	# Create labels for y_test
	y_test_labels = [['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'][i] for i in y_test]

	# Create labels for y_train
	y_train_labels = [['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'][i] for i in y_train]

	# Instantiate Model Playground object
	from aimodelshare.playground import ModelPlayground
	myplayground = ModelPlayground(input_type="image", task_type="classification", private=False)

	# Create Model Playground Page on modelshare.ai website
	myplayground.create(eval_data=y_test_labels)

	if torch.cuda.is_available():
		example_input = torch.randn(1, 3, 128, 128, requires_grad=True).cuda()
	else:
		example_input = torch.randn(1, 3, 128, 128, requires_grad=True)


	# Submit Model to Experiment Leaderboard
	myplayground.submit_model(model=model,
							  preprocessor=preprocessor,
							  prediction_submission=prediction_labels,
							  input_dict={"description": "", "tags": ""},
							  submission_type="all",
							  model_input = example_input)


	# Create example data folder to provide on model playground page
	#     for users to test prediction REST API
	import shutil
	os.mkdir('example_data')
	example_images = ["flower-competition-data/train_images/daisy/100080576_f52e8ee070_n.jpg",
					  "flower-competition-data/train_images/dandelion/10200780773_c6051a7d71_n.jpg",
					  "flower-competition-data/train_images/roses/10503217854_e66a804309.jpg",
					  "flower-competition-data/train_images/sunflowers/1022552002_2b93faf9e7_n.jpg",
					  "flower-competition-data/train_images/tulips/100930342_92e8746431_n.jpg"]

	for image in example_images:
		shutil.copy(image, 'example_data')

	# Deploy model by version number
	myplayground.deploy_model(model_version=1, example_data="example_data", y_train=y_train)

	# example url from deployed playground: apiurl= "https://123456.execute-api.us-east-1.amazonaws.com/prod/m
	apiurl = myplayground.playground_url

	# Submit Model 2
	# Define model
	class NeuralNetwork(nn.Module):
		def __init__(self):
			super(NeuralNetwork, self).__init__()
			self.flatten = nn.Flatten()
			self.linear_relu_stack = nn.Sequential(
				nn.Linear(128 * 128 * 3, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 256),
				nn.ReLU(),
				nn.Linear(256, 5)
			)

		def forward(self, x):
			x = self.flatten(x)
			logits = self.linear_relu_stack(x)
			return logits

	model2 = NeuralNetwork().to(device)
	print(model2)

	# set up loss function and optimizer
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model2.parameters(), lr=1e-3)

	# train model
	epochs = 2
	for t in range(epochs):
		print(f"Epoch {t + 1}\n-------------------------------")
		train(train_dataloader, model2, loss_fn, optimizer)
		test(test_dataloader, model2, loss_fn)
	print("Done!")

	# Submit Model 2 to Experiment Leaderboard
	myplayground.submit_model(model=model2,
							  preprocessor=preprocessor,
							  prediction_submission=prediction_labels,
							  input_dict={"description": "", "tags": ""},
							  submission_type="all",
							  model_input = example_input)

	# submit model through competition
	mycompetition = ai.playground.Competition(myplayground.playground_url)
	mycompetition.submit_model(model=model2,
							   preprocessor=preprocessor,
							   prediction_submission=prediction_labels,
							   input_dict={"description": "", "tags": ""},
							   model_input=example_input)

	# submit model through experiment
	myexperiment = ai.playground.Experiment(myplayground.playground_url)
	myexperiment.submit_model(model=model2,
							  preprocessor=preprocessor,
							  prediction_submission=prediction_labels,
							  input_dict={"description": "", "tags": ""},
							  model_input=example_input)

	# Check experiment leaderboard
	data = myplayground.get_leaderboard()
	myplayground.stylize_leaderboard(data)
	assert isinstance(data, pd.DataFrame)

	# Compare two or more models
	data = myplayground.compare_models([1, 2], verbose=1)
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
	shutil.rmtree("flower-competition-data", onerror=redo_with_write)
	shutil.rmtree("example_data", onerror=redo_with_write)
