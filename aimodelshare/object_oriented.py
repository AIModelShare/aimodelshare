# import packages 




class ModelPlayground:
    """
    Parameters:
    ----------
    `model_type` : ``string``
          values - [ 'text' , 'image' , 'tabular' , 'video', 'audio','timeseries' ] 
          type of model data     
    `classification`:    ``bool, default=True``
        True [DEFAULT] if model is of Classification type with categorical target variables
        False if model is of Regression type with continuous target variables
    `private` :   ``bool, default = False``
        True if model and its corresponding data is not public
        False [DEFAULT] if model and its corresponding data is public 
    `email_list`: ``list of string values``
                values - list including all emails of users who have access the private playground.
                list should contain same emails that were used by users to sign up for modelshare.org account.
                [OPTIONAL] set by the playground owner for private playgrounds.  Can also be updated by editing deployed 
                playground page at www.modelshare.org.
    """
    def __init__(self, model_type=None, classification=None, private=None, playground_url=None, email_list=[]):
        # confirm correct args are provided
        if playground_url != None or all([model_type !=None, classification !=None, private!=None]):
            pass
        elif playground_url == None and any([model_type ==None, classification ==None, private==None]):
            return print("Error. To instantiate a ModelPlayground instance, please provide either a playground_url or \n the model_type, classification, and private arguments.")
        
        self.model_type = model_type
        self.categorical = classification 
        self.private = private
        self.playground_url = playground_url
        self.email_list = email_list

    
    
    def __str__(self):
        return f"ModelPlayground instance of model type: {self.model_type}, classification: {self.categorical},  private: {self.private}"
    

    def deploy(self, model_filepath, preprocessor_filepath, y_train, example_data=None, custom_libraries = "FALSE", image="", reproducibility_env_filepath=None, memory=None, timeout=None, pyspark_support=False):

        """
        Launches a live prediction REST API for deploying ML models using model parameters and user credentials, provided by the user
        Inputs : 7
        Output : model launched to an API
                detailed API info printed out

        Parameters: 
        ----------
        `model_filepath` :  ``string`` ends with '.onnx'
              value - Absolute path to model file 
              [REQUIRED] to be set by the user
              .onnx is the only accepted model file extension
              "example_model.onnx" filename for file in directory.
              "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `preprocessor_filepath`:  ``string``
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated using export_preprocessor function from the AI Modelshare library  
        `y_train` : training labels for classification models.
              [REQUIRED] for classification type models
              expects pandas dataframe of one hot encoded y train data
        `custom_libraries`:   ``string``
            "TRUE" if user wants to load custom Python libraries to their prediction runtime
            "FALSE" if user wishes to use AI Model Share base libraries including latest versions of most common ML libs.
         
        Returns:
        --------
        print_api_info : prints statements with generated live prediction API details
                        also prints steps to update the model submissions by the user/team
        """
        from aimodelshare.generatemodelapi import model_to_api
        self.playground_url = model_to_api(model_filepath=model_filepath, 
                                      model_type = self.model_type, 
                                      private = self.private, 
                                      categorical = self.categorical,
                                      y_train = y_train, 
                                      preprocessor_filepath = preprocessor_filepath, 
                                      example_data = example_data,
                                      custom_libraries = custom_libraries,
                                      image=image,
                                      reproducibility_env_filepath = reproducibility_env_filepath,
                                      memory=memory,
                                      timeout=timeout,
                                      email_list=self.email_list,
                                      pyspark_support=pyspark_support)
        #remove extra quotes
        self.playground_url = self.playground_url[1:-1]
    
    def create_competition(self, data_directory, y_test, eval_metric_filepath=None, email_list = [], public=False, public_private_split=0.5):
        """
        Creates a model competition for a deployed prediction REST API
        Inputs : 4
        Output : Create ML model competition and allow authorized users to submit models to resulting leaderboard/competition
        
        Parameters:
        -----------
        `y_test` :  ``list`` of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
            [REQUIRED] to generate eval metrics in competition leaderboard
                                
        `data_directory` : folder storing training data and test data (excluding Y test data)
        `email_list`: [OPTIONAL] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
        `public`: [REQUIRED] True/false. Defaults to False.  If True, competition is public and ANY AIMODELSHARE USER CAN SUBMIT MODELS.  USE WITH CAUTION b/c one model and 
            one preprocessor file will be be saved to your AWS S3 folder for each model submission.
        `public_private_split`: [REQUIRED] Float between 0 and 1. Defaults to 0.5. Porportion of test data that is allocated to private hold-out set.
        
        Returns:
        -----------
        finalmessage : Information such as how to submit models to competition
        
        """
        from aimodelshare.generatemodelapi import create_competition

        competition = create_competition(self.playground_url, 
                                    data_directory, 
                                    y_test, 
                                    eval_metric_filepath,
                                    email_list, 
                                    public,
                                    public_private_split)
        return competition
        
    def create_experiment(self, data_directory, y_test, eval_metric_filepath=None, email_list = [], public=False, public_private_split=0):
        """
        Creates an experiment for a deployed prediction REST API
        Inputs : 4
        Output : Create ML model experiment and allows authorized users to submit models to resulting experiment tracking leaderboard
        
        Parameters:
        -----------
        `y_test` :  ``list`` of y values for test data used to generate metrics from predicted values from X test data submitted via the submit_model() function
            [REQUIRED] to generate eval metrics in experiment leaderboard
                                
        `data_directory` : folder storing training data and test data (excluding Y test data)
        `email_list`: [OPTIONAL] list of comma separated emails for users who are allowed to submit models to experiment leaderboard.  Emails should be strings in a list.
        `public`: [REQUIRED] True/false. Defaults to False.  If True, experiment is public and ANY AIMODELSHARE USER CAN SUBMIT MODELS.  USE WITH CAUTION b/c one model and 
            one preprocessor file will be be saved to your AWS S3 folder for each model submission.
        `public_private_split`: [REQUIRED] Float between 0 and 1. Defaults to 0. Porportion of test data that is allocated to private hold-out set.
        
        
        Returns:
        -----------
        finalmessage : Information such as how to submit models to experiment
        
        """
        from aimodelshare.generatemodelapi import create_experiment

        experiment = create_experiment(self.playground_url, 
                                    data_directory, 
                                    y_test, 
                                    eval_metric_filepath,
                                    email_list, 
                                    public,
                                    public_private_split)
        return experiment

    def submit_model(self, model_filepath, preprocessor_filepath, prediction_submission):
        """
        Submits model/preprocessor to machine learning experiment leaderboard and model architecture database using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated 
        
        Parameters:
        -----------
        `model_filepath`:  ``string`` ends with '.onnx'
            value - Absolute path to model file [REQUIRED] to be set by the user
            .onnx is the only accepted model file extension
            "example_model.onnx" filename for file in directory.
            "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `preprocessor_filepath`:   ``string``, default=None
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
        `prediction_submission`: [REQUIRED] list of predictions from X test data that will be used to evaluate model prediction error against y test data.
            Use mycompetition.inspect_y_test() to view example of list expected by competition.
        
        Returns:
        -------
        response:   Model version if the model is submitted sucessfully
                    error  if there is any error while submitting models
        
        """

        from aimodelshare.model import submit_model as submit
        submission = submit(model = model_filepath, 
                            apiurl = self.playground_url,
                            prediction_submission = prediction_submission, 
                            preprocessor = preprocessor_filepath)
        return submission
        

    
    def update_runtime_model(self, model_version=None, submission_type="competition"):
        """
        Updates the prediction API behind the Model Playground with a new model from the leaderboard and verifies Model Playground performance metrics.

        Parameters:
        -----------
        `model_version`: ``int``
            model version number from competition leaderboard

        Returns:
        --------
        response:   success message when the model and preprocessor are updated successfully
        
        """
        from aimodelshare.model import update_runtime_model as update
        update = update(apiurl = self.playground_url, model_version = model_version, submission_type=submission_type)
        return update
        
    def instantiate_model(self, version=None, trained=False, reproduce=False, submission_type="competition"): 
        """
        Import a model previously submitted to a leaderboard to use in your session

        Parameters:
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
        `trained`: ``bool, default=False``
            if True, a trained model is instantiated, if False, the untrained model is instantiated

        Returns:
        --------
        model: model chosen from leaderboard
        """
        raise AssertionError("You are trying to Instantiate model with ModelPlayground Object, Please use the competition object to Instantiate model")
        from aimodelshare.aimsonnx import instantiate_model
        model = instantiate_model(apiurl=self.playground_url, trained=trained, version=version, reproduce=reproduce, submission_type=submission_type)
        return model
    
    def replicate_model(self,version=None, submission_type="competition"): 
        """
        Instantiate an untrained model with reproducibility environment setup. 
        
        Parameters: 
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
          
        Returns:
        --------
        model:  model chosen from leaderboard     
        """
        
        model = self.instantiate_model(version=version,trained = False,reproduce=True,submission_type=submission_type)
        
        return model
    
   
    def delete_deployment(self, playground_url=None):
        """
        Delete all components of a Model Playground, including: AWS s3 bucket & contents,
        attached competitions, prediction REST API, and interactive Model Playground web dashboard.

        Parameters:
        -----------
        `playground_url`: ``string`` of API URL the user wishes to delete
        WARNING: User must supply high-level credentials in order to delete an API.

        Returns:
        --------
        Success message when deployment is deleted.
        """
        from aimodelshare.api import delete_deployment
        if playground_url == None:
            playground_url = self.playground_url
        deletion = delete_deployment(apiurl = playground_url)
        return deletion

    def import_reproducibility_env(self):
        from aimodelshare.reproducibility import import_reproducibility_env_from_model
        import_reproducibility_env_from_model(apiurl=self.playground_url)

    def update_access_list(self, email_list=[], update_type="Replace_list"):
        """
        Updates list of authenticated participants who can submit new models to a competition.

        Parameters:
        -----------
        `apiurl`: string
                URL of deployed prediction API 
          
        `email_list`: [REQUIRED] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
        `update_type`:[REQUIRED] options, ``string``: 'Add', 'Remove', 'Replace_list','Get. Add appends user emails to original list, Remove deletes users from list, 
                  'Replace_list' overwrites the original list with the new list provided, and Get returns the current list.    

        Returns:
        --------
        response:   "Success" upon successful request
        """
        from aimodelshare.generatemodelapi import update_access_list as update_list
        update = update_list(apiurl = self.playground_url, email_list=email_list, update_type=update_type)
        return update


class Competition:
    """
    Parameters:
    ----------
    `playground_url`: playground_url attribute of ModelPlayground class or ``string``
        of existing ModelPlayground URL
    """

    submission_type = "competition"

    def __init__(self, playground_url):
        self.playground_url = playground_url
    
    def __str__(self):
        return f"Competition class instance for playground: {self.playground_url}"
        
    def submit_model(self, model_filepath, preprocessor_filepath, prediction_submission, 
        sample_data=None, reproducibility_env_filepath=None, custom_metadata=None):
        """
        Submits model/preprocessor to machine learning competition using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated
        
        Parameters:
        -----------
        `model_filepath`:  ``string`` ends with '.onnx'
            value - Absolute path to model file [REQUIRED] to be set by the user
            .onnx is the only accepted model file extension
            "example_model.onnx" filename for file in directory.
            "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        `prediction_submission`:   one hot encoded y_pred
            value - predictions for test data
            [REQUIRED] for evaluation metrics of the submitted model
        `preprocessor_filepath`:   ``string``, default=None
            value - absolute path to preprocessor file 
            [REQUIRED] to be set by the user
            "./preprocessor.zip" 
            searches for an exported zip preprocessor file in the current directory
            file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
 
        Returns:
        --------
        response:   Model version if the model is submitted sucessfully
        """

        from aimodelshare.model import submit_model as submit
        submission = submit(model_filepath = model_filepath, 
                              apiurl = self.playground_url,
                              prediction_submission = prediction_submission, 
                              preprocessor = preprocessor_filepath,
                              reproducibility_env_filepath = reproducibility_env_filepath,
                              custom_metadata = custom_metadata, 
                              submission_type = self.submission_type)
        return submission

    def instantiate_model(self, version=None, trained=False, reproduce=False): 
        """
        Import a model previously submitted to the competition leaderboard to use in your session

        Parameters:
        -----------
        `version`: ``int``
            Model version number from competition leaderboard
        `trained`: ``bool, default=False``
            if True, a trained model is instantiated, if False, the untrained model is instantiated
       
        Returns:
        --------
        model: model chosen from leaderboard
        """
        from aimodelshare.aimsonnx import instantiate_model
        model = instantiate_model(apiurl=self.playground_url, trained=trained, version=version, 
            reproduce=reproduce, submission_type=self.submission_type)
        return model

    def replicate_model(self,version=None): 
        """
        Instantiate an untrained model previously submitted to the competition leaderboard with its reproducibility environment setup. 
        
        Parameters: 
        -----------
        `version`: ``int``
            Model version number from competition or experiment leaderboard
          
        Returns:
        --------
        model:  model chosen from leaderboard     
        """
        
        model = self.instantiate_model(version=version,trained = False,reproduce=True) 
        return model 
                        
    def inspect_model(self, version=None, naming_convention=None):
        """
        Examine structure of model submitted to a competition leaderboard

        Parameters:
        ----------
        `version` : ``int``
            Model version number from competition leaderboard
      
        Returns:
        --------
        inspect_pd : dictionary of model summary & metadata
        """
        from aimodelshare.aimsonnx import inspect_model

        inspect_pd = inspect_model(apiurl=self.playground_url, version=version, 
            naming_convention=naming_convention, submission_type = self.submission_type)

        return inspect_pd

    def compare_models(self, version_list="None", by_model_type=None, best_model=None, verbose=1, naming_convention=None):
        """
        Compare the structure of two or more models submitted to a competition leaderboard.
        Use in conjuction with stylize_compare to visualize data. 
        
        Parameters:
        -----------
        `version_list` = ``list of int``
            list of model version numbers to compare (previously submitted to competition leaderboard) 
        `verbose` = ``int``
            controls the verbosity: the higher, the more detail 
        
        Returns:
        --------
        data : dictionary of model comparison information
        """
        from aimodelshare.aimsonnx import compare_models as compare
        data = compare(apiurl = self.playground_url, 
                      version_list = version_list, 
                      by_model_type = by_model_type,
                      best_model = best_model, 
                      verbose = verbose,
                      naming_convention=naming_convention,
                      submission_type = self.submission_type)
        return data

    def stylize_compare(self, compare_dict, naming_convention="keras"):
        """
        Stylizes data received from compare_models to highlight similarities & differences.

        Parameters:
        -----------
        `compare_dict` = dictionary of model data from compare_models
 
        Returns:
        --------
        formatted table of model comparisons 
        """
        from aimodelshare.aimsonnx import stylize_model_comparison
        stylized_compare = stylize_model_comparison(comp_dict_out=compare_dict, naming_convention=naming_convention)
        return(stylized_compare)

    def inspect_y_test(self):
        """
        Examines structure of y-test data to hep users understand how to submit models to the competition leaderboad.

        Parameters:
        ------------
        None
 
        Returns:
        --------
        dictionary of a competition's y-test metadata
        """
        from aimodelshare.aimsonnx import inspect_y_test
        data = inspect_y_test(apiurl = self.playground_url, submission_type=self.submission_type)
        return data
    
    def get_leaderboard(self, verbose=3, columns=None):
        """
        Get current competition leaderboard to rank all submitted models.
        Use in conjuction with stylize_leaderboard to visualize data. 
        
        Parameters:
        -----------
        `verbose` : optional, ``int``
            controls the verbosity: the higher, the more detail 
        `columns` : optional, ``list of strings``
            list of specific column names to include in the leaderboard, all else will be excluded
            performance metrics will always be displayed
        
        Returns:
        --------
        dictionary of leaderboard data 
        """
        from aimodelshare.leaderboard import get_leaderboard
        data = get_leaderboard(verbose=verbose,
                 columns=columns, 
                 apiurl = self.playground_url, 
                 submission_type=self.submission_type)
        return data
    
    def stylize_leaderboard(self, leaderboard, naming_convention="keras"):
        """
        Stylizes data received from get_leaderbord.

        Parameters:
        -----------
        `leaderboard` : data dictionary object returned from get_leaderboard

        Returns:
        --------
        Formatted competition leaderboard
        """
        from aimodelshare.leaderboard import stylize_leaderboard as stylize_lead
        stylized_leaderboard = stylize_lead(leaderboard = leaderboard, naming_convention=naming_convention)
        return stylized_leaderboard
    
    def update_access_list(self, email_list=[],update_type="Replace_list"):
        """
        Updates list of authenticated participants who can submit new models to a competition.

        Parameters:
        -----------
        `apiurl`: string
                URL of deployed prediction API 
          
        `email_list`: [REQUIRED] list of comma separated emails for users who are allowed to submit models to competition.  Emails should be strings in a list.
        `update_type`:[REQUIRED] options, ``string``: 'Add', 'Remove', 'Replace_list','Get. Add appends user emails to original list, Remove deletes users from list, 
                  'Replace_list' overwrites the original list with the new list provided, and Get returns the current list.    

        Returns:
        --------
        response:   "Success" upon successful request
        """
        from aimodelshare.generatemodelapi import update_access_list as update_list
        update = update_list(apiurl = self.playground_url, 
            email_list=email_list,update_type=update_type,
            submission_type=self.submission_type)
        return update



class Experiment(Competition):
    """
    Parameters:
    ----------
    `playground_url`: playground_url attribute of ModelPlayground class or ``string``
        of existing ModelPlayground URL
    """

    submission_type = "experiment"

    def __init__(self, playground_url):
        self.playground_url = playground_url
    
    def __str__(self):
        return f"Experiment class instance for playground: {self.playground_url}"
        
    

class Data: 
    def __init__(self, data_type, playground_url=None):
        self.data_type = data_type
        self.playground_url = playground_url 
    
    def __str__(self):
        return f"This is a description of the Data class."

    def share_dataset(self, data_directory="folder_file_path", classification="default", private="FALSE"): 
        from aimodelshare.data_sharing.share_data import share_dataset as share
        response = share(data_directory=data_directory, classification=classification, private=private)
        return response
    
    def download_data(self, repository):
        from aimodelshare.data_sharing.download_data import download_data as download
        datadownload = download(repository)
        return datadownload

