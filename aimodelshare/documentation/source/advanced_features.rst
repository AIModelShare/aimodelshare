.. _advanced_features: 

Advanced Features
#################

The AI Model Share library can help deploy interactive web dashboards for making predictions with your models in minutes. 

However, sometimes there is a need for higher levels of customization than the base features include. 

This page describes additional pathways to allow for the greatest programming flexibility, while also leveraging the power of the AI Model Share library. 

.. _custom_deployments:

Custom Deployments
******************

The base ``ModelPlayground.deploy`` method deploys a pre-written lambda handler optimized for efficiency with specific types of prediction domains. 

For projects requiring more flexibility in the lambda handler, the AI ModelShare library allows for "Custom Deployments". Custom Deployments allow users to leverage the AI ModelShare infrastructure through AWS, while also creating opportunities for additional customization. 

**Tutorial**

`Guide to Custom Deployments <https://www.modelshare.org/notebooks/notebook:365>`_

.. _PySpark:

Using PySpark
*************

AI Model Share supports PySpark. Note that the current prediction API runtime can only accept Pandas Dataframes, which may require additional steps for PySpark preprocessors. 


**Tutorial**

`Quick Start Tutorial with PySpark <https://www.modelshare.org/notebooks/notebook:366>`_
	

.. _webapps:

Connecting Custom Web Apps
**************************

AI ModelShare supports connecting your custom web apps to the AI ModelShare infrastructure deployment process in AWS. Web Apps can be featured on the AI ModelShare website and connected to model competitions & experiments.  

**Examples**

* `Text Classification: Webapp <https://share.streamlit.io/raudipra/streamlit-text-classification/main>`_
* `Text Classification: Github <https://github.com/raudipra/streamlit-text-classification>`_
* `Tabular Classification: Webapp <https://share.streamlit.io/raudipra/streamlit-tabular-classification/main>`_
* `Tabular Classification: Github <https://github.com/raudipra/streamlit-tabular-classification>`_
* `Image Classification: Webapp <https://share.streamlit.io/raudipra/streamlit-image-classification/main>`_
* `Image Classification: Github <https://github.com/raudipra/streamlit-image-classification>`_