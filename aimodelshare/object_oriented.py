# import packages 




class ModelPlayground:
    """
    Parameters:
    model_type :  string 
                  values - [ 'Text' , 'Image' , 'Tabular' , 'Timeseries' ] 
                  type of model data     
    classification:    bool, default=True
                        True [DEFAULT] if model is of Classification type with categorical target variables
                        False if model is of Regression type with continuous target variables
    private :   bool, default = False
                    True if model and its corresponding data is not public
                    False [DEFAULT] if model and its corresponding data is public 
    """
    def __init__(self, model_type=None, classification=None, private=None, playground_url=None):
        # confirm correct args are provided
        if playground_url != None or all([model_type !=None, classification !=None, private!=None]):
            pass
        elif playground_url == None and any([model_type ==None, classification ==None, private==None]):
            return print("Error. To instantiate a ModelPlayground instance, please provide either a playground_url or \n the model_type, classification, and private arguments.")
        
        self.model_type = model_type
        self.categorical = classification 
        self.private = private
        self.playground_url = playground_url
    
    
    def __str__(self):
        return f"ModelPlayground instance of model type: {self.model_type}, classification: {self.categorical},  private: {self.private}"
    
    def deploy(self, model_filepath, preprocessor_filepath, y_train, example_data=None, custom_libraries = "FALSE"):
        """
        Launches a live prediction REST API for deploying ML models using model parameters and user credentials, provided by the user
        Inputs : 8
        Output : model launched to an API
                detaled API info printed out

        -----------
        Parameters 
        
        model_filepath :  string ends with '.onnx'
                          value - Absolute path to model file 
                          [REQUIRED] to be set by the user
                          .onnx is the only accepted model file extension
                          "example_model.onnx" filename for file in directory.
                          "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        preprocessor_filepath:  string
                                value - absolute path to preprocessor file 
                                [REQUIRED] to be set by the user
                                "./preprocessor.zip" 
                                searches for an exported zip preprocessor file in the current directory
                                file is generated using export_preprocessor function from the AI Modelshare library  
        y_train : training labels of size of dataset
                  value - y values for model
                  [REQUIRED] for classification type models
                  expects a one hot encoded y train data format  
        custom_libraries:   string
                    "TRUE" if user wants to load custom Python libraries to their prediction runtime
                    "FALSE" if user wishes to use AI Model Share base libraries including latest versions of most common ML libs.
        example_data:  pandas DataFrame (for tabular & text data) OR filepath as string (image, audio, video data)
                      tabular data - pandas DataFrame in same structure expected by preprocessor function
                      other data types - absolute path to folder containing example data
                                          (first five files with relevent file extensions will be accepted)
                      [REQUIRED] for tabular data
        -----------
        Returns
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
                                      custom_libraries = custom_libraries)
        #remove extra quotes
        self.playground_url = self.playground_url[1:-1]
    
    def create_competition(self, data_directory, y_test, generate_credentials_file = False):
        """
        Creates a model competition for a deployed prediction REST API
        Inputs : 2
        Output : Submit credentials for model competition
        
        ---------
        Parameters
        y_test :  y labels for test data 
                [REQUIRED] for eval metrics
                expects a one hot encoded y test data format
                
        data_directory : folder storing training data and test data (excluding Y test data)
        generate_credentials_file (OPTIONAL): Default is True
                                              Function will output .txt file with new credentials
        ---------
        Returns
        finalresultteams3info : Submit_model credentials with access to S3 bucket
        (api_id)_credentials.txt : .txt file with submit_model credentials,
                                    formatted for use with set_credentials() function 
        """
        from aimodelshare.generatemodelapi import create_competition as to_competition
        competition = to_competition(self.playground_url, 
                                    data_directory, 
                                    y_test, 
                                    generate_credentials_file)
        return competition
        
    def submit_model(self, model_filepath, preprocessor_filepath, prediction_submission, sample_data=None):
        """
        Submits model/preprocessor to machine learning competition using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated 
        ---------------
        Parameters:
        model_filepath:  string ends with '.onnx'
                    value - Absolute path to model file [REQUIRED] to be set by the user
                    .onnx is the only accepted model file extension
                    "example_model.onnx" filename for file in directory.
                    "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        prediction_submission:   one hot encoded y_pred
                        value - predictions for test data
                        [REQUIRED] for evaluation metriicts of the submitted model
        preprocessor_filepath:   string,default=None
                        value - absolute path to preprocessor file 
                        [REQUIRED] to be set by the user
                        "./preprocessor.zip" 
                        searches for an exported zip preprocessor file in the current directory
                        file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
        -----------------
        Returns
        response:   Model version if the model is submitted sucessfully
                    error  if there is any error while submitting models
        
        """
        from aimodelshare.model import submit_model as submit
        submission = submit(model = model_filepath, 
                            apiurl = self.playground_url,
                            prediction_submission = prediction_submission, 
                            preprocessor = preprocessor_filepath,
                            sample_data = sample_data)
        return submission
        
    
    def update_runtime_model(self, model_version=None): 
        from aimodelshare.model import update_runtime_model as update
        update = update(apiurl = self.playground_url, model_version = model_version)
        return update
        
    def instantiate_model(self, version=None, trained=False): 
        from aimodelshare.aimsonnx import instantiate_model
        model = instantiate_model(apiurl=self.playground_url, trained=trained, version=version)
        return model
    
    def delete_deployment(self, playground_url=None):
        """
        Delete all components of a Model Playground, including: AWS s3 bucket & contents,
        attached competitions, prediction REST API, and interactive Model Playground web dashboard.
        ---------------
        playground_url: string of API URL the user wishes to delete

        WARNING: User must supply high-level credentials in order to delete an API. 
        """
        from aimodelshare.api import delete_deployment
        if playground_url == None:
            playground_url = self.playground_url
        deletion = delete_deployment(apiurl = playground_url)
        return deletion




class Competition: 
    def __init__(self, playground_url):
        self.playground_url = playground_url
    
    def __str__(self):
        return f"Competition class instance for playground: {self.playground_url}"
        
    def submit_model(self, model_filepath, preprocessor_filepath, prediction_submission, sample_data=None):
        """
        Submits model/preprocessor to machine learning competition using live prediction API url generated by AI Modelshare library
        The submitted model gets evaluated and compared with all existing models and a leaderboard can be generated 
        ---------------
        Parameters:
        model_filepath:  string ends with '.onnx'
                    value - Absolute path to model file [REQUIRED] to be set by the user
                    .onnx is the only accepted model file extension
                    "example_model.onnx" filename for file in directory.
                    "/User/xyz/model/example_model.onnx" absolute path to model file from local directory
        prediction_submission:   one hot encoded y_pred
                        value - predictions for test data
                        [REQUIRED] for evaluation metriicts of the submitted model
        preprocessor_filepath:   string,default=None
                        value - absolute path to preprocessor file 
                        [REQUIRED] to be set by the user
                        "./preprocessor.zip" 
                        searches for an exported zip preprocessor file in the current directory
                        file is generated from preprocessor module using export_preprocessor function from the AI Modelshare library 
        -----------------
        Returns
        response:   Model version if the model is submitted sucessfully
                    error  if there is any error while submitting models
        
        """
        from aimodelshare.model import submit_model as submit
        submission = submit(model = model_filepath, 
                              apiurl = self.playground_url,
                              prediction_submission = prediction_submission, 
                              preprocessor = preprocessor_filepath,
                              sample_data = sample_data)
        return submission
        
    def instantiate_model(self, version=None, trained=False): 
        from aimodelshare.aimsonnx import instantiate_model
        model = instantiate_model(apiurl=self.playground_url, trained=trained, version=version)
        return model

    def compare_models(self, version_list="None", by_model_type=None, best_model=None, verbose=3):
        from aimodelshare.aimsonnx import compare_models as compare
        data = compare(apiurl = self.playground_url, 
                      version_list = version_list, 
                      by_model_type = by_model_type,
                      best_model = best_model, 
                      verbose = verbose)
        return data
    
    def inspect_y_test(self):
        from aimodelshare.aimsonnx import inspect_y_test as inspect
        data = inspect(apiurl = self.playground_url)
        return data 
    
    def get_leaderboard(self, verbose=3, columns=None):
        from aimodelshare.leaderboard import get_leaderboard as get_lead
        data = get_lead(verbose=verbose,
                 columns=columns, 
                 apiurl = self.playground_url)
        return data
    
    def stylize_leaderboard(self, leaderboard):
        from aimodelshare.leaderboard import stylize_leaderboard as stylize_lead
        stylized_leaderboard = stylize_lead(leaderboard = leaderboard)
        return stylized_leaderboard







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





