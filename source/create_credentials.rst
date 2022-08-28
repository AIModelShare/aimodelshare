.. _create_credentials: 

Setting up AI Model Share Credentials
#####################################

In order to deploy a Model Playground, AI Model Share needs access to create & manage cloud-based resources on your behalf through Amazon Web Services (AWS). This guide shows you how to create an AWS account, access the proper credentials, and create a credentials file to use with the aimodelshare python library. 

Step 1: Create an AI Model Share Account
**************************************** 

*If you already have an AI Model Share account, proceed to the next step.* 

Create an AI Model Share account by going `HERE <https://www.modelshare.org/login>`_ and following the prompts.

*Are you just using your credentials to submit models to a pre-existing playground competition or experiment? You're done!* 
Get started with any of the :ref:`Model Submission notebooks <example_notebooks>`.  


Step 2: Create an AWS account
*****************************
 
*If you already have an AWS account, proceed to the next step.* 
 
Create an AWS account by going `HERE <https://portal.aws.amazon.com/billing/signup#/start/email>`_ and following the prompts.


Step 3: Create an IAM User & AWS Access Keys
********************************************

In order for aimodelshare to access your AWS resources, you will need to create IAM credentials for your AWS account. 

	* `Log in <https://signin.aws.amazon.com/signin>`_ to your AWS account. 
	* From the AWS Management Console, Navigate to “Security, Identity, & Compliance” 	and then “IAM”. 

		.. image:: images/creds1.png
   			:width: 300

	* In the side menu, navigate to “Access management”, click on “Users”, then click 	on “Add User” on the right side of the screen. 

		.. image:: images/creds2.png
   			:width: 600

	* Create a name that you’ll remember, like “aimodelshareadmin”, then enable 	“programmatic access” by checking the box. 

		.. image:: images/creds3.png
   			:width: 600

	* On the next screen, click “Attach existing policies directly”, then 	“AdministratorAccess”. 

		.. image:: images/creds4.png
   			:width: 600

	* Click Next: Review, then “Create User”. 	
	* Copy the Access key ID and Secret access key  and save them somewhere safe. 		These are the credentials you will use to link your AI Model Share account to the 	resources in your AWS account. 

Step 4: Create your credentials file 
************************************

Combine your AI Model Share & AWS credentials into a single ‘credentials.txt' file with the `configure_credentials` function. You only have to make the file once, then you can use it whenever you use the aimodelshare library. 

Credentials files must follow this format: 
	
	.. image:: images/creds_file_example.png
   			:width: 600

You can create this txt file manually or you can automatically create this file by inputting your credentials in response to simple prompts from the configure_credentials() function.  The following code will prompt you to provide your credentials one at a time and pre-format a txt file for you to use in the future: 

.. code-block::

	#install aimodelshare library
	! pip install aimodelshare

	# Generate credentials file
	import aimodelshare as ai 
	from aimodelshare.aws import configure_credentials 
	configure_credentials()


.. warning::

	Remember to keep your credentials secure! Handle your credentials file with the same level of security you handle your passwords. Do not share your file with anyone, send via email, or upload to Github.

Step 5: Get started! 
********************

Now that you have your credentials file, you are ready to work through the :ref:`AI Model Share Tutorial <aimodelshare_tutorial>`  or one of the :ref:`example_notebooks`. 
