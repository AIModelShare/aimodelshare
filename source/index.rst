.. AIModelShare documentation master file, created by
   sphinx-quickstart on Sun Feb  6 18:05:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AIModelShare's documentation!
========================================

Installing the Library
---------------------- 

To install the AIModelShare library using PyPI: 
***********************************************

::

	# Install aimodelshare library
	! pip install aimodelshare

To install the AIModelShare library on Anaconda: 
************************************************

*Conda/Mamba Install (For Mac and Linux Users Only, Windows Users should use pip method):*

Make sure you have conda >=4.9.

You can check your conda version with: :: 

	conda --version 

To update conda, use: :: 

	conda update conda
	
Installing aimodelshare from the ``conda-forge`` channel can be achieved by adding ``conda-forge`` to your channels with: :: 

	conda config --add channels conda-forge
	conda config --set channel_priority strict

Once the conda-forge channel has been enabled, aimodelshare can be installed with conda: :: 

	conda install aimodelshare

Or with mamba: :: 
	
	mamba install aimodelshare 


To fully utilize the library functionality, you will need to set up credentials with the `AI Model Share website <https://www.modelshare.org/login>`_ and/or with `Amazon Web Services. <https://aws.amazon.com/free>`_ See the credentials user guide :ref:`HERE. <create_credentials>`

AI Model Share Tutorial
-----------------------
.. toctree::

   gettingstarted

Generating Credentials
----------------------
.. toctree::

   create_credentials

AI Model Share Example Notebooks
--------------------------------
.. toctree::

   example_notebooks

AI Model Share Classes
----------------------
.. toctree::

   modelplayground
   competition


AI Model Share Supporting Functions 
-----------------------------------
.. toctree::

   functions

AI Model Share Advanced Features 
--------------------------------
.. toctree::

   advanced_features

About Us
---------
.. toctree::

   about


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

License
-------
AI | Model Share - All rights reserved. `Terms of Service <https://www.modelshare.org/terms-of-service>`_ and `Privacy Policy. <https://www.modelshare.org/privacy-policy>`_
