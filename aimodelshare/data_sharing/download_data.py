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
	sys.stdout.write('\r' + layer_label + ': Downloading [')
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
	layer_label = str((layer_count/10)*100)+" Pct Complete"
	layer_dir = tmp_img_dir + '/' + layer_id

	# Creating layer.tar file
	sys.stdout.write(layer_label + ': Downloading...')
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
	print('Creating image structure in: ' + tmp_img_dir)

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
	layers = resp.json()['layers']

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
	sys.stdout.write("Creating archive...")
	sys.stdout.flush()

	tar = tarfile.open(docker_tar, "w")
	tar.add(tmp_img_dir, arcname=os.path.sep)
	tar.close()

	shutil.rmtree(tmp_img_dir)

	return docker_tar

def extract_data_from_image(image_name, file_name):
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
    shutil.copytree(tempfile.gettempdir()+'/var/task/'+file_name, file_name)
    shutil.rmtree(tempfile.gettempdir()+'/var')

def download_data(repository):
	data_zip_name = repository.split('/')[2].split('-repository')[0]
	docker_tar = pull_image(repository)
	extract_data_from_image(docker_tar, data_zip_name)
	os.remove(docker_tar)
	print('Data downloaded successfully.')
