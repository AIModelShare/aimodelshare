.. _example_notebooks: 

Example Notebooks
#################

After completing the AI Model Share Tutorial, you can deploy Model Playgrounds and submit models for a variety of live data classification & regression competitions. 

Choose a dataset that you would like to work with, and find everything you need included below. Notebooks can be downloaded or opened directly in Google Colab.  

(*Looking for PySpark?* Go :ref:`here. <tabular_class>`) 

.. _tabular_class:

Tabular Classification:
***********************

Titanic Dataset
================
	"The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others." Use data about the passengers on board to determine whether they were likely to survive the shipwreck or not. 

	*Dataset and description adapted from: Kaggle GettingStarted Prediction Competition. (2012, September). Titanic - Machine Learning from Disaster, Version 1. Retrieved September 7, 2021 from https://www.kaggle.com/c/titanic/data.* 

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:304>`_
	* `Quick Start Tutorial with PySpark <https://www.modelshare.org/notebooks/notebook:366>`_
	* `Model Submission Guide: sklearn <https://www.modelshare.org/notebooks/notebook:305>`_
	* `Model Submission Guide: Deep Learning <https://www.modelshare.org/notebooks/notebook:306>`_
	* `Model Submission Guide: Predictions Only (no model metadata extracted) <https://www.modelshare.org/notebooks/notebook:319>`_

.. _tabular_reg:

Tabular Regression:
*******************

Used Car Sales Price Dataset
============================

	Cars notoriously lose value as soon as they are purchased. However, the resell value of any particular car depends on many factors, including make, model, miles driven, transmission type, the number of owners, and more. Use this dataset to predict the resell value of used cars based on their features.

	*Dataset adapted from: Birla, Nehal, Nishant Verma, and Nikhil Kushwaha. (June, 2018). Vehicle dataset, Version 3. Retrieved September 14, 2021 from https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho.*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:285>`_
	* `Model Submission Guide: sklearn <https://www.modelshare.org/notebooks/notebook:286>`_
	* `Model Submission Guide: Deep Learning <https://www.modelshare.org/notebooks/notebook:287>`_

.. _text_class:

Text Classification:
********************

Covid Misinformation Identification
===================================

	"The significance of social media has increased manifold in the past few decades as it helps people from even the most remote corners of the world stay connected. With the COVID-19 pandemic raging, social media has become more relevant and widely used than ever before, and along with this, there has been a resurgence in the circulation of fake news and tweets that demand immediate attention." Use this dataset to read COVID-19 related tweets and determine if the information they present is "real" or "fake". 

	*Description and dataset adapted from: Sourya Dipta Das, Ayan Basak, and Saikat Dutta. "A Heuristic-driven Ensemble Framework for COVID-19 Fake News Detection”. arXiv preprint arXiv:2101.03545. 2021. Retrieved from https://github.com/diptamath/covid_fake_news/tree/main/data.*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:290>`_
	* `Model Submission Guide: sklearn <https://www.modelshare.org/notebooks/notebook:291>`_
	* `Model Submission Guide: Deep Learning <https://www.modelshare.org/notebooks/notebook:292>`_


Clickbait Identification
========================

	"In the online world, many media outlets have to generate revenue from the clicks made by their readers, and due to the presence of numerous such outlets, they have to compete with each other for reader attention. To attract the readers to click on an article and visit the media site, they often come up with catchy headlines accompanying the article links, which lure the readers to click on the link. Such headlines are known as Clickbaits. While these baits may trick the readers into clicking, in the long-run, clickbaits usually don’t live up to the expectation of the readers and leave them disappointed." Use this dataset to read headlines from multiple media outlets and identify whether they represent clickbait or substantive news.

	*Dataset and description adapted From: Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly. "Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media”. In Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), San Fransisco, US, August 2016.*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:288>`_
	* `Model Submission Guide <https://www.modelshare.org/notebooks/notebook:289>`_
	

IMDB Movie Review Identification
================================

	IMDb, also knows as the Internet Movie Database, is an online database of movies, TV shows, celebrities, and awards. Registered users can write reviews and rate content that they've seen. Use this dataset to classify 50,000 'highly polarized' movie reviews as positive or negative. 

	*Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:300>`_
	* `Model Submission Guide <https://www.modelshare.org/notebooks/notebook:301>`_


.. _image_class:

Image Classification:
*********************

Dog Breed Classification
========================

	This dataset contains pictures from 6 different dog breeds, adapted from the original dataset with 120 different dog breeds from around the world. Use this dataset to look at images of dogs and determine which breed they belong to. 

	*Dataset adapted from: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011. Retrieved from http://vision.stanford.edu/aditya86/ImageNetDogs/*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:293>`_
	* `Model Submission Guide <https://www.modelshare.org/notebooks/notebook:294>`_

Fashion MNIST Classification
============================

	An updated version of the iconic handwritten digits MNIST dataset. Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from one of 10 classes. 

	*Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:295>`_
	* `Model Submission Guide <https://www.modelshare.org/notebooks/notebook:296>`_

Flower Classification
=====================

	A dataset containing pictures of 5 different classes of flowers. 

	*The Tensorflow Team. (2019, January). Flowers. http://download.tensorflow.org/example_images/flower_photos.tgz*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:297>`_
	* `Model Submission Guide <https://www.modelshare.org/notebooks/notebook:299>`_

.. _video_class:

Video Classification:
*********************

Sports Clips Classification
===========================

	Video clips of people doing pull-ups, kayaking, and horseback riding. Use this dataset to watch video clips and determine which of three activities are taking place. 

	*Dataset adapted from: Soomro, K., Zamir, A. R., & Sha, M. (2012). UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild. Center for Research in Computer Vision, University of Central Florida. https://arxiv.org/pdf/1212.0402v1.pdf*

	* `Quick Start Tutorial (Start here to Deploy) <https://www.modelshare.org/notebooks/notebook:302>`_
	* `Model Submission Guide <https://www.modelshare.org/notebooks/notebook:303>`_
