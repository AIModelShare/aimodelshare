import os
import sys
import gzip
from io import BytesIO
import json
import shutil
import requests
import tempfile
import tarfile
import urllib3
import re

urllib3.disable_warnings()

def get_auth_head_no_aws_auth(auth_url, registry, repository, type):
	resp = requests.get('{}?service={}&scope=repository:{}:pull'.format(auth_url, registry, repository), verify=False)
	access_token = resp.json()['token']
	auth_head = {'Authorization':'Bearer '+ access_token, 'Accept': type}
	return auth_head

def progress_bar(layer_label, nb_traits):
	sys.stdout.write('\r' + layer_label + 'Downloading [')
	for i in range(0, nb_traits):
		if i == nb_traits - 1:
			sys.stdout.write('>')
		else:
			sys.stdout.write('=')
	for i in range(0, 49 - nb_traits):
		sys.stdout.write(' ')
	sys.stdout.write(']')
	sys.stdout.flush()

def get_auth_url(registry): # to do with auth
	return 'https://' + registry + '/token/' # no aws auth

def get_auth_head(auth_url, registry, repository): # to do with auth
	return get_auth_head_no_aws_auth(auth_url, registry, repository, 'application/vnd.docker.distribution.manifest.v2+json') # no aws auth

def download_layer(layer, layer_count, tmp_img_dir, blobs_resp):

	ublob = layer['digest']
	layer_id = 'layer_' + str(layer_count) + '_' + ublob[7:]
	layer_label = ""
	layer_dir = tmp_img_dir + '/' + layer_id

	# Creating layer.tar file
	sys.stdout.write(layer_label + 'Downloading...')
	sys.stdout.flush()

	# Stream download and follow the progress
	unit = int(blobs_resp.headers['Content-Length']) / 50
	acc = 0
	nb_traits = 0
	progress_bar(layer_label, nb_traits)

	os.mkdir(layer_dir)
	with open(layer_dir + '/layer_gzip.tar', "wb") as file:
		for chunk in blobs_resp.iter_content(chunk_size=8192): 
			if chunk:
				file.write(chunk)
				acc = acc + 8192
				if acc > unit:
					nb_traits = nb_traits + 1
					progress_bar(layer_label, nb_traits)
					acc = 0

	sys.stdout.flush()

	with open(layer_dir + '/layer.tar', "wb") as file:
		unzip_layer = gzip.open(layer_dir + '/layer_gzip.tar','rb')
		shutil.copyfileobj(unzip_layer, file)
		unzip_layer.close()
	os.remove(layer_dir + '/layer_gzip.tar')

	return layer_id, layer_dir

def pull_image(image_uri):

	image_uri_parts = image_uri.split('/')

	registry = image_uri_parts[0]	
	image, tag = image_uri_parts[2].split(':')
	repository = '/'.join([image_uri_parts[1], image])

	auth_url = get_auth_url(registry)

	auth_head = get_auth_head(auth_url, registry, repository)

	resp = requests.get('https://{}/v2/{}/manifests/{}'.format(registry, repository, tag), headers=auth_head, verify=False)
    
	config = resp.json()['config']['digest']
	config_resp = requests.get('https://{}/v2/{}/blobs/{}'.format(registry, repository, config), headers=auth_head, verify=False)

	tmp_img_dir = tempfile.gettempdir() + '/' + 'tmp_{}_{}'.format(image, tag)
	os.mkdir(tmp_img_dir)

	file = open('{}/{}.json'.format(tmp_img_dir, config[7:]), 'wb')
	file.write(config_resp.content)
	file.close()

	content = [{
		'Config': config[7:] + '.json',
		'RepoTags': [],
		'Layers': []
	}]
	content[0]['RepoTags'].append(image_uri)

	layer_count=0
	layers = resp.json()['layers'][6:]

	for layer in layers:

		layer_count += 1

		auth_head = get_auth_head(auth_url, registry, repository) # done to keep from expiring
		blobs_resp = requests.get('https://{}/v2/{}/blobs/{}'.format(registry, repository, layer['digest']), headers=auth_head, stream=True, verify=False)

		layer_id, layer_dir = download_layer(layer, layer_count, tmp_img_dir, blobs_resp)
		content[0]['Layers'].append(layer_id + '/layer.tar')

		# Creating json file
		file = open(layer_dir + '/json', 'w')

		# last layer = config manifest - history - rootfs
		if layers[-1]['digest'] == layer['digest']:
			json_obj = json.loads(config_resp.content)
			del json_obj['history']
			del json_obj['rootfs']
		else: # other layers json are empty
			json_obj = json.loads('{}')
		
		json_obj['id'] = layer_id
		file.write(json.dumps(json_obj))
		file.close()

	file = open(tmp_img_dir + '/manifest.json', 'w')
	file.write(json.dumps(content))
	file.close()

	content = {
		'/'.join(image_uri_parts[:-1]) + '/' + image : { tag : layer_id }
	}

	file = open(tmp_img_dir + '/repositories', 'w')
	file.write(json.dumps(content))
	file.close()

	# Create image tar and clean tmp folder
	docker_tar = tempfile.gettempdir() + '/' + '_'.join([repository.replace('/', '_'), tag]) + '.tar'
	sys.stdout.flush()

	tar = tarfile.open(docker_tar, "w")
	tar.add(tmp_img_dir, arcname=os.path.sep)
	tar.close()

	shutil.rmtree(tmp_img_dir)

	return docker_tar

def extract_data_from_image(image_name, file_name, location):
    tar = tarfile.open(image_name, 'r')
    files = []
    for t in tar.getmembers():
        if('.tar' not in t.name):
            continue
        tar_layer = tarfile.open(fileobj=tar.extractfile(t))
        for tl in tar_layer.getmembers():
            if(re.match("var/task/"+file_name, tl.name)):
                files.append(tl)
        if(len(files)>0):
            break
    tar_layer.extractall(members=files, path=tempfile.gettempdir())
    if(os.path.isdir(file_name)):
        shutil.rmtree(file_name)
    shutil.copytree(tempfile.gettempdir()+'/var/task/'+file_name, os.path.join(location, file_name))
    shutil.rmtree(tempfile.gettempdir()+'/var')

def download_data(repository, location="./"):
	data_zip_name = repository.split('/')[2].split('-repository')[0]
	docker_tar = pull_image(repository)
	extract_data_from_image(docker_tar, data_zip_name, location)
	os.remove(docker_tar)
	print('\n\nData downloaded successfully.')


def import_quickstart_data(tutorial, section="modelplayground"):
    from aimodelshare.data_sharing.download_data import download_data
    import tensorflow as tf
    import os
    import pickle
    import shutil 
    import pandas as pd
    
   
    #Download Quick Start materials
    if all([tutorial == "flowers", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_materials-repository:latest"   
        existing_folder = 'flower_competition_data'
    if all([tutorial == "flowers", section == "competition"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_flowers_competition-repository:latest"
        existing_folder = 'flower_competition_data'
    
    if all([tutorial == "mnist", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/fashion_mnist_quickstart_materials-repository:latest"   
        existing_folder = 'fashion_mnist_competition_data'
    if all([tutorial == "mnist", section == "competition"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_mnist_competition-repository:latest"
        existing_folder = 'fashion_mnist_competition_data'
        
    if all([tutorial == "titanic", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/titanic_quickstart-repository:latest" 
        existing_folder = 'titanic_competition_data'
        
    if all([tutorial == "cars", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_car_sales_competition-repository:latest" 
        existing_folder = 'used_car_competition_data'
        
    if all([tutorial == "clickbait", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_clickbait_materials-repository:latest" 
        existing_folder = 'clickbait_competition_data'

    if all([tutorial == "covid_tweets", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_covid_competition-repository:latest" 
        existing_folder = 'covid_tweet_competition_data'
        
    if all([tutorial == "sports", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/sports_quick_start_materials-repository:latest" 
        existing_folder = 'sports_clips_competition_data'
    if all([tutorial == "sports", section == "competition"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_sports_competition-repository:latest"
        existing_folder = 'sports_clips_competition_data'

    if all([tutorial == "dogs", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/dog_breed_quickstart_materials-repository:latest"   
        existing_folder = 'dog_competition_data'
    if all([tutorial == "dogs", section == "competition"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/quickstart_dog_breed_competition-repository:latest"
        existing_folder = 'dog_competition_data'

    if all([tutorial == "imdb", section == "modelplayground"]):
        quickstart_repository = "public.ecr.aws/y2e2a1d6/imdb_quickstart_materials-repository:latest"   
        existing_folder = 'imdb_competition_data'

    download_data(quickstart_repository)
    
    #Delete pre-existing tutorial folders
    if os.path.exists(existing_folder):
        shutil.rmtree(existing_folder)

    #{{{ Prepare modelplayground materials
    if section == "modelplayground": 
        print("\nPreparing downloaded files for use...")
        
        if tutorial == "dogs":
           #instantiate model
            model = tf.keras.models.load_model('dog_breed_quickstart_materials/model.h5')
            
            #unpack data
            y_train = pd.read_csv("dog_breed_quickstart_materials/y_train.csv")
        
        if tutorial == "covid_tweets":
            #unpack data 
            X_train = pd.read_csv("quickstart_covid_competition/X_train.csv", squeeze=True)
            X_test = pd.read_csv("quickstart_covid_competition/X_test.csv", squeeze=True)
            y_test_labels = pd.read_csv("quickstart_covid_competition/y_test_labels.csv", squeeze=True)
            y_train_labels = pd.read_csv("quickstart_covid_competition/y_train_labels.csv", squeeze=True)
            # example data
            example_data = X_train[50:55]

            #move data files to cometition folder
            os.mkdir('covid_tweet_competition_data')
                
            files = ['quickstart_covid_competition/X_train.csv', 
                      'quickstart_covid_competition/X_test.csv',
                     'quickstart_covid_competition/y_train_labels.csv']
                
            for f in files:
                shutil.move(f, 'covid_tweet_competition_data')

        if tutorial == "imdb":
            #unpack data 
            X_train = pd.read_csv("imdb_quickstart_materials/X_train.csv", squeeze=True)
            X_test = pd.read_csv("imdb_quickstart_materials/X_test.csv", squeeze=True)
            y_test_labels = pd.read_csv("imdb_quickstart_materials/y_test_labels.csv", squeeze=True)
            y_train_labels = pd.read_csv("imdb_quickstart_materials/y_train_labels.csv", squeeze=True)
            # example data
            example_data = X_train[50:55]

            #instantiate models
            lstm_model = tf.keras.models.load_model('imdb_quickstart_materials/model_1.h5')
            lstm_model2 = tf.keras.models.load_model('imdb_quickstart_materials/model_2.h5')

            #move data files to cometition folder
            os.mkdir('imdb_competition_data')
                
            files = ['imdb_quickstart_materials/X_train.csv', 
                      'imdb_quickstart_materials/X_test.csv',
                      'imdb_quickstart_materials/y_train_labels.csv']
                
            for f in files:
                shutil.move(f, 'imdb_competition_data')

        if tutorial == "flowers":
           #instantiate model
            model = tf.keras.models.load_model('quickstart_materials/flowermodel.h5')
            
            #unpack data
            with open("quickstart_materials/y_train_labels.txt", "rb") as fp:  
                y_train_labels = pickle.load(fp)

        if tutorial == "mnist":
           #instantiate model
            model = tf.keras.models.load_model('fashion_mnist_quickstart_materials/mnist_model_1.h5')
            
            #unpack data
            with open("fashion_mnist_quickstart_materials/y_train_labels.pkl", "rb") as fp:  
                y_train_labels = pickle.load(fp)
        
        if tutorial == "sports":
           #instantiate model
            model = tf.keras.models.load_model('sports_quick_start_materials/video_1.h5')
            
            #unpack data
            y_train_labels = pd.read_csv("sports_quick_start_materials/y_train.csv")
                
        if tutorial == "clickbait":
            import pandas as pd
            # suppress tf warning
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
            #instantiate models
            lstm_model = tf.keras.models.load_model('quickstart_clickbait_materials/lstm_model1.h5')
            lstm_model2 = tf.keras.models.load_model('quickstart_clickbait_materials/lstm_model2.h5')
           
            # bring in data 
            clickbait = pd.read_csv('quickstart_clickbait_materials/clickbait_data',  sep="\n", header = None)
            clickbait['label'] = "clickbait"
            clickbait.columns = ['headline', 'label']
        
            not_clickbait = pd.read_csv('quickstart_clickbait_materials/non_clickbait_data',  sep="\n", header = None)
            not_clickbait['label'] = "not clickbait"
            not_clickbait.columns = ['headline', 'label']

            # train/test/split
            from sklearn.model_selection import train_test_split
            X = clickbait.append(not_clickbait)
            y = X['label']
            X = X.drop(['label'], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=942)
            X_test.reset_index(drop=True, inplace=True)
            X_test = X_test.squeeze()
            X_train.reset_index(drop=True, inplace=True)
            X_train = X_train.squeeze()

            # exampledata
            example_data = X_train[0:5]
            
            # Create data directory for competition 
            X_train.to_csv("X_train.csv", index=False)
            X_test.to_csv("X_test.csv", index=False)
            y_train.to_csv("y_train.csv", index=False)
            
            os.mkdir('clickbait_competition_data')
            
            files = ['X_train.csv', 'X_test.csv', 'y_train.csv']
                        
            for f in files:
                shutil.move(f, 'clickbait_competition_data')

        if tutorial == "titanic":
            from sklearn.model_selection import train_test_split
            import pandas as pd
            #read in data
            data = pd.read_csv("titanic_quickstart/titanic_data.csv")
            y = data['survived']
            y = y.map({0: 'died', 1: 'survived'}) 
            X = data.drop(['survived','sibsp','parch','ticket','name','cabin','boat','body','home.dest'], axis=1)
            #create subset as exampledata 
            example_data = pd.DataFrame(X[0:4])
            # create data directory for competition
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            training_data = pd.merge(X_train, y_train, left_index=True, right_index=True)
            training_data = training_data.drop(list(training_data.filter(like='Unnamed')), axis=1)
            training_data.to_csv("training_data.csv", index=False)

            test_data = X_test
            test_data = test_data.drop(list(test_data.filter(like='Unnamed')), axis=1)
            test_data.to_csv("test_data.csv", index=False)
            
            os.mkdir('titanic_competition_data')
            files = ['training_data.csv', 
                   'test_data.csv']
            
            for f in files:
                shutil.move(f, 'titanic_competition_data')
            
            #make y_test_labels for competition
            y_test_labels = y_test.to_list()

        if tutorial == "cars":
            from sklearn.model_selection import train_test_split

            # read in data
            import pandas as pd
            data = pd.read_csv("quickstart_car_sales_competition/used_car_dataset.csv")
            y = data['selling_price']
            X = data.drop(['selling_price', 'torque', 'name'], axis=1)

            #Data Prep:
                # convert rupees to $ (for smaller MSEs)
            y = y.mul(.014)
                # A: Split units from mileage and convert units 
            Correct_Mileage= []
            for i in X.mileage:
                if str(i).endswith('km/kg'):
                    i = i[:-6]
                    i = float(i)*1.40
                    Correct_Mileage.append(float(i))
                elif str(i).endswith('kmpl'):
                    i = i[:-5]
                    Correct_Mileage.append(float(i))
                else: 
                    Correct_Mileage.append(None)
            X['mileage']=Correct_Mileage

                  #B: Split units from engine, & max_pwer
            X['engine'] = X['engine'].str.replace(' CC', '')
            X['engine'] = pd.to_numeric(X['engine'])

            X['max_power'] = X['max_power'].str.replace(' bhp', '')
            X['max_power'] = pd.to_numeric(X['max_power'])

            #create subset as exampledata 
            example_data = pd.DataFrame(X[0:4])

            #create data directory for competition
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            training_data = X_train
            training_data = pd.merge(X_train, y_train, left_index=True, right_index=True)
            training_data.to_csv("training_data.csv", index=False)

            test_data = X_test
            test_data.to_csv("test_data.csv", index=False)

            os.mkdir('used_car_competition_data')
            files = ['training_data.csv', 
                        'test_data.csv']
            for f in files:
                shutil.move(f, 'used_car_competition_data')
    #}}}

    #{{{ prepare competition materials
    if section == "competition":
        print("\nPreparing downloaded files for use...")

        if tutorial == "flowers":
            #Instantiate Model 
            model_2 = tf.keras.models.load_model('quickstart_flowers_competition/flowermodel_2.h5')
        
            #unpack data
            with open("quickstart_flowers_competition/y_test_labels.txt", "rb") as fp:  
                y_test_labels = pickle.load(fp)
                
            #move data files to folder to upload with create_competiton
            os.mkdir('flower_competition_data')
                
            folders = ['quickstart_flowers_competition/test_images', 
                      'quickstart_flowers_competition/train_images']
                
            for f in folders:
                shutil.move(f, 'flower_competition_data')
        
        if tutorial == "mnist":
            #Instantiate Model 
            model_2 = tf.keras.models.load_model('quickstart_mnist_competition/mnist_model_2.h5')
        
            #unpack data
            with open("quickstart_mnist_competition/y_test_labels.pkl", "rb") as fp:  
                y_test_labels = pickle.load(fp)
                
            #move data files to folder to upload with create_competiton
            os.mkdir('fashion_mnist_competition_data')
                
            folders = ['quickstart_mnist_competition/test_data', 
                      'quickstart_mnist_competition/training_data', 
                       'fashion_mnist_quickstart_materials/y_train_labels.pkl']
                
            for f in folders:
                shutil.move(f, 'fashion_mnist_competition_data')

        if tutorial == "dogs":
            #Instantiate Model 
            model_2 = tf.keras.models.load_model('quickstart_dog_breed_competition/model_2.h5')
        
            #unpack data
            with open("quickstart_dog_breed_competition/y_test_labels.txt", "rb") as fp:  
                y_test_labels = pickle.load(fp)
                
            #move data files to folder to upload with create_competiton
            os.mkdir('dog_competition_data')
                
            folders = ['quickstart_dog_breed_competition/dog_breed_competition_data/test_images', 
                      'quickstart_dog_breed_competition/dog_breed_competition_data/train_images']
                
            for f in folders:
                shutil.move(f, 'dog_competition_data')
        
        if tutorial == "sports":
            model_2 = tf.keras.models.load_model('quickstart_sports_competition/video_2.h5')
            y_test = pd.read_csv("quickstart_sports_competition/y_test.csv")
            y_test_labels = y_test.idxmax(axis=1)
            os.mkdir('sports_clips_competition_data')
            folders = ['quickstart_sports_competition/clips_test.zip', 
                      'quickstart_sports_competition/clips_train.zip']
            for f in folders:
                shutil.move(f, 'sports_clips_competition_data')
    #}}}

    success_message = ("\nSuccess! Your Quick Start materials have been downloaded. \n"
                       "You are now ready to run the tutorial.")
    
    print(success_message)

    if all([tutorial == "flowers", section == "modelplayground"]):
        return model, y_train_labels

    if all ([tutorial == "flowers", section == "competition"]): 
        return model_2, y_test_labels
    
    if all([tutorial == "mnist", section == "modelplayground"]):
        return model, y_train_labels

    if all([tutorial == "mnist", section == "competition"]): 
        return model_2, y_test_labels

    if all([tutorial == "dogs", section == "modelplayground"]):
        return model, y_train

    if all ([tutorial == "dogs", section == "competition"]): 
        return model_2, y_test_labels
    
    if all([tutorial == "sports", section == "modelplayground"]):
        return model, y_train_labels
    
    if all ([tutorial == "sports", section == "competition"]): 
        return model_2, y_test, y_test_labels
    
    if tutorial == "titanic":
        return X_train, X_test, y_train, y_test, example_data, y_test_labels

    if tutorial == "cars":
        return X_train, X_test, y_train, y_test, example_data
    
    if tutorial == "clickbait":
        return X_train, X_test, y_train, y_test, example_data, lstm_model, lstm_model2	

    if tutorial == "imdb":
        return X_train, X_test, y_train_labels, y_test_labels, example_data, lstm_model, lstm_model2	

    if tutorial == "covid_tweets":
        return X_train, X_test, y_train_labels, y_test_labels, example_data
